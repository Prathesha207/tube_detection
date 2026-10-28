"""
tube_analyzer.py
================
Thin adapter between the backend services and the ``head_tail_analyzer`` wheel.

THE WHEEL DOES THE ML. This file only adds what a backend needs around it:

    * model cache + hot-reload (weights file changed on disk)
    * CUDA probe / CPU fallback
    * per-session ROI (drawn by the user) applied between frames
    * the dict shape the frontend expects (``build_frontend_stats``)

Per frame it makes ONE call into the wheel, exactly the call the wheel's own CLI /
``HeadTailAnalyzer`` makes::

    result, annotated = rvf.analyze_frame(entry, frame, state, roi, cfg, frame_number,
                                          annotate=..., masks=...)

so detection, tiling, merging, tracking, widths, Bigger/Smaller/Same, ``bigger_tube``,
counts, ``fps``, ``frame_total``, ``video_fps`` and the drawn picture all come from the
wheel. Nothing in this file computes or invents a detection, a width, a size verdict or a
frame rate. Settings come from config.yaml through the wheel's own loader
(``rvf.load_stream_config``), so the same config.yaml gives the same settings here and in
the CLI. ``TubeAnalyzer.describe_settings()`` prints what is actually in use.

The wheel is REQUIRED. If it is missing (or too old) startup fails instead of silently
running a different local copy. Set ``TUBE_ALLOW_LOCAL_RVF=1`` to allow the old fallback to
``app.ml.test_inference.run_video_frames`` (development only; it is logged as an ERROR).

``analysis.result`` is exactly what the wheel produces::

    {"frame", "heads_count", "tails_count", "heads_visible", "tails_visible",
     "heads_total", "tails_total", "heads_peak", "tails_peak", "frame_total", "fps",
     "video_fps", "bigger_tube",
     "detections": [{"id", "role", "conf", "bbox", "center", "width_px", "width_mm",
                     "width_now_px", "width_spread_px", "width_source", "size"}]}

One TubeAnalyzer = one video / stream of tracker state. It is NOT thread-safe:
call it from one thread at a time (the services do this with a lock / a single
worker thread).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

_CUDA_KERNELS_USABLE: Optional[bool] = None

_REQUIRED_WHEEL_FUNCS = (
    "detect", "update_tracks", "assign_sizes", "build_frame_result",
    "tubing_id_from_names", "role_ids_from_names", "in_roi", "cap_per_role", "draw",
    # the per-frame pipeline + state + loader the adapter now delegates to:
    "analyze_frame", "StreamState", "LoadedModel", "configure_for_model",
    "load_stream_config", "open_video", "as_bgr", "DEFAULTS",
)

logger = logging.getLogger("tube-analyzer")

rvf = None
_WHEEL_ERROR = None
WHEEL_INFO: Dict[str, Any] = {}

# 1. The installed wheel (required).
try:
    import head_tail_analyzer as _hta
    from head_tail_analyzer import run_video_frames as _wheel_rvf
    _missing = [n for n in _REQUIRED_WHEEL_FUNCS if not hasattr(_wheel_rvf, n)]
    if _missing:
        raise ImportError(f"installed head_tail_analyzer is too old, missing: {_missing}")
    rvf = _wheel_rvf
    WHEEL_INFO = {"source": "wheel", "version": getattr(_hta, "__version__", "?"), "file": rvf.__file__}
except Exception as e:
    _WHEEL_ERROR = e

# 2. Development-only escape hatch. Off by default: a silent fallback to a different copy of the
#    pipeline is exactly how "the session run differs from the CLI run" happens.
if rvf is None and os.environ.get("TUBE_ALLOW_LOCAL_RVF") == "1":
    try:
        from app.ml.test_inference import run_video_frames as _local_rvf
        _missing = [n for n in _REQUIRED_WHEEL_FUNCS if not hasattr(_local_rvf, n)]
        if _missing:
            raise ImportError(f"local run_video_frames is missing: {_missing}")
        rvf = _local_rvf
        _WHEEL_ERROR = None
        WHEEL_INFO = {"source": "LOCAL COPY (not the wheel)", "version": "?", "file": rvf.__file__}
        logger.error("TUBE_ALLOW_LOCAL_RVF=1: using a LOCAL run_video_frames, NOT the installed wheel: %s", rvf.__file__)
    except Exception as e:
        logger.error("Could not import local run_video_frames: %s", e)

if rvf is not None:
    logger.info("head_tail_analyzer pipeline: %s v%s  (%s)", WHEEL_INFO.get("source"),
                WHEEL_INFO.get("version"), WHEEL_INFO.get("file"))


_ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(_ML_DIR / "model" / "best.pt")
CONFIG_PATH = str(_ML_DIR / "config" / "config.yaml")
ROI_JSON = str(_ML_DIR / "config" / "roi.json")

#################### 1. Hardware & CUDA Helpers ####################
# --------------------------------------------------------------- small helpers
def cuda_is_available() -> bool:
    """Return true only when this PyTorch build can execute kernels on the GPU.

    ``torch.cuda.is_available()`` only reports that a CUDA device/driver exists;
    it does not detect an unsupported GPU architecture (for example sm_120 with
    an older wheel). Probe once so inference selects CPU instead of silently
    failing on every frame.
    """
    global _CUDA_KERNELS_USABLE
    if _CUDA_KERNELS_USABLE is not None:
        return _CUDA_KERNELS_USABLE
    if torch is None:
        _CUDA_KERNELS_USABLE = False
        return False
    try:
        with warnings.catch_warnings(record=True) as cuda_warnings:
            warnings.simplefilter("always")
            available = torch.cuda.is_available()
        for warning in cuda_warnings:
            logger.warning("CUDA initialization warning: %s", warning.message)
        if not available or torch.cuda.device_count() == 0:
            _CUDA_KERNELS_USABLE = False
            return False
        probe = torch.ones((1,), device="cuda:0")
        _ = probe + 1
        torch.cuda.synchronize(0)
        del probe
        # NOTE: cudnn.benchmark is deliberately NOT switched on. With benchmark=True cuDNN
        # picks the fastest convolution algorithm per input shape, and the choice can differ
        # between runs/processes. Different algorithms give slightly different float results,
        # which moves confidences and mask widths by a hair, and that is enough to flip a
        # detection sitting at the conf cutoff or a width ratio sitting at min_ratio. The
        # wheel's CLI does not enable it, so neither do we (opt in: TUBE_CUDNN_BENCHMARK=1).
        if os.environ.get("TUBE_CUDNN_BENCHMARK") == "1":
            torch.backends.cudnn.benchmark = True
        _CUDA_KERNELS_USABLE = True
    except Exception as e:
        logger.warning("CUDA is visible but this PyTorch build cannot run GPU kernels; using CPU: %s", e)
        _CUDA_KERNELS_USABLE = False
    return _CUDA_KERNELS_USABLE


def default_device() -> str:
    """"0" (first GPU) when CUDA works, otherwise "cpu"."""
    return "0" if cuda_is_available() else "cpu"


def inference_context():
    """torch.inference_mode() when torch is installed, otherwise a no-op context.
    inference_mode is per-thread, so enter it in the thread that runs the model."""
    return torch.inference_mode() if torch is not None else nullcontext()


def release_gpu_memory(clear_cache: bool = False) -> None:
    """Call AFTER the last reference to a model is dropped, to give VRAM back."""
    if clear_cache:
        _GLOBAL_YOLO_CACHE.clear()
        _GLOBAL_YOLO_MTIMES.clear()
        _GLOBAL_ENTRIES.clear()
        _GLOBAL_WARMED_SHAPES.clear()
    gc.collect()
    if cuda_is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def clear_yolo_cache() -> None:
    """Explicitly release all cached YOLO models to completely free GPU VRAM."""
    release_gpu_memory(clear_cache=True)

#################### 2. Region of Interest (ROI) Math ####################
def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if line segment (p1, p2) intersects with (p3, p4).
    _segments_intersect() & _is_self_intersecting() Complex math functions that prevent users from drawing "bowtie" shaped polygons where the lines cross over each other (which breaks the OpenCV drawing algorithms)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _is_self_intersecting(pts: np.ndarray) -> bool:
    """True if polygon has intersecting edges."""
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p3, p4 = pts[j], pts[(j + 1) % n]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    return False


def roi_from_points(points: Any, frame_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Validate and convert [[x, y], ...] in native frame pixels -> (N, 2) int32 polygon.
    Empty / None clears the ROI (whole frame counted).
    Raises ValueError on invalid input or geometry.  Takes the raw mouse-click coordinates [[x, y], ...] from the frontend and converts them into a strict Numpy Array. It enforces rules like "must have at least 3 points" and "cannot be smaller than 10x10 pixels".
    """
    if not points:
        return None
    try:
        arr = np.array(points, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("ROI points must be a list of [x, y] coordinates")

    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError("ROI points must be a list of 2D coordinates [[x, y], ...]")

    if len(arr) > 3 and np.array_equal(arr[0], arr[-1]):
        arr = arr[:-1]  # drop closing duplicate point if present

    if len(arr) < 3:
        raise ValueError("ROI needs at least 3 points")
    if len(arr) > 100:
        raise ValueError("ROI cannot have more than 100 points")

    w_span = arr[:, 0].max() - arr[:, 0].min()
    h_span = arr[:, 1].max() - arr[:, 1].min()
    if w_span < 10 or h_span < 10:
        raise ValueError("ROI must be at least 10x10 pixels")

    area = cv2.contourArea(arr.astype(np.float32))
    if area < 100:
        raise ValueError("ROI area is too small (minimum 100 square pixels)")

    if _is_self_intersecting(arr):
        raise ValueError("ROI polygon cannot be self-intersecting")

    if frame_size and frame_size[0] > 0 and frame_size[1] > 0:
        arr[:, 0] = arr[:, 0].clip(0, frame_size[0] - 1)
        arr[:, 1] = arr[:, 1].clip(0, frame_size[1] - 1)

    return np.round(arr).astype(np.int32)


def save_roi(points: Any, path: Optional[str] = None, frame_size: Optional[Tuple[int, int]] = None) -> None:
    """Validate, then write roi.json atomically with frame_width and frame_height.
    save_roi() & load_roi() Reads and writes the polygon points to roi.json so your drawn zones survive a server restart. If your camera changes resolution, load_roi automatically scales the polygon up or down to match!"""
    roi = roi_from_points(points, frame_size)
    target = Path(path or ROI_JSON)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"points": [] if roi is None else roi.tolist()}
    if frame_size and frame_size[0] > 0 and frame_size[1] > 0:
        payload["frame_width"] = int(frame_size[0])
        payload["frame_height"] = int(frame_size[1])

    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def load_roi(path: Optional[str] = None, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Polygon from roi.json as an (N, 2) int32 array, or None.
    If target_shape (height, width) is given and roi.json has saved dimensions,
    rescales the polygon so that switching resolutions maintains the correct relative region.
    """
    target = Path(path) if path else Path(ROI_JSON)
    if not target.exists():
        logger.info(f"No ROI file at {target} -- the whole frame is counted")
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        points = data.get("points")
        if not points:
            return None
        roi = np.array(points, dtype=np.float64)
        if roi.ndim != 2 or len(roi) < 3:
            return None

        saved_w = data.get("frame_width")
        saved_h = data.get("frame_height")
        if target_shape and saved_w and saved_h and saved_w > 0 and saved_h > 0:
            tgt_h, tgt_w = target_shape[0], target_shape[1]
            if tgt_w != saved_w or tgt_h != saved_h:
                sx = tgt_w / float(saved_w)
                sy = tgt_h / float(saved_h)
                roi = roi * [sx, sy]

        return np.round(roi).astype(np.int32)
    except Exception as e:
        logger.warning(f"Could not read ROI file {target}: {e} -- running without ROI")
        return None

#################### 4. Model Loading & Warming ####################
_GLOBAL_YOLO_CACHE: Dict[str, Any] = {}
_GLOBAL_YOLO_MTIMES: Dict[str, float] = {}
_GLOBAL_ENTRIES: Dict[str, Any] = {}          # cache_key -> rvf.LoadedModel (model + role ids + infer lock)
_GLOBAL_WARMED_SHAPES: Set[Tuple[int, int]] = set()
_ENTRY_SERIAL = 0

def _load_yolo_model(path: str, device: str = "cpu"):
    """Load the YOLO model (cached) for fast inference.
    Automatically detects if the model file was updated on disk and reloads it."""
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    cache_key = f"{path}:{device}"
    if cache_key in _GLOBAL_YOLO_CACHE:
        cached_mtime = _GLOBAL_YOLO_MTIMES.get(cache_key, 0.0)
        if cached_mtime == mtime:
            return _GLOBAL_YOLO_CACHE[cache_key]
        logger.info(f"Model file {path} was modified on disk (cached mtime {cached_mtime} -> new {mtime}). Evicting old model.")
        _GLOBAL_YOLO_CACHE.pop(cache_key, None)
        _GLOBAL_YOLO_MTIMES.pop(cache_key, None)
        release_gpu_memory(clear_cache=False)

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics / PyTorch failed to load. Check the PyTorch installation.") from e
    model = YOLO(path)
    if device != "cpu" and cuda_is_available():
        try:
            cuda_dev = f"cuda:{device}" if str(device).isdigit() else str(device)
            model.to(cuda_dev)
        except Exception as dev_err:
            logger.warning(f"Failed to place model on {device}: {dev_err}")
    _GLOBAL_YOLO_CACHE[cache_key] = model
    _GLOBAL_YOLO_MTIMES[cache_key] = mtime
    return model

def _load_entry(path: str, device: str = "cpu"):
    """The wheel's ``LoadedModel`` for this weights file: model + HEAD/TAIL class ids + tubing id +
    the per-model inference lock that ``rvf.analyze_frame`` takes around ``model.predict``.
    Cached with the model, so every analyzer sharing a model shares one lock."""
    global _ENTRY_SERIAL
    model = _load_yolo_model(path, device)
    key = f"{path}:{device}"
    entry = _GLOBAL_ENTRIES.get(key)
    if entry is not None and entry.model is model:
        return entry
    names = dict(model.names)
    try:
        role_ids = rvf.role_ids_from_names(names)
    except SystemExit as e:        # the wheel exits instead of raising; SystemExit is not an Exception
        raise RuntimeError(f"Model {path} is not usable for head/tail counting: {e}") from None
    _ENTRY_SERIAL += 1
    entry = rvf.LoadedModel(path=str(path), model=model, role_ids=role_ids,
                            tubing_id=rvf.tubing_id_from_names(names),
                            task=getattr(model, "task", "segment"), names=names, serial=_ENTRY_SERIAL)
    _GLOBAL_ENTRIES[key] = entry
    return entry


#################### 3. Data Formatting ####################
_WIDTH_DETECTION_KEYS = (
    "width_px", "width_mm", "width_now_px", "width_spread_px",
    "width_source", "size", "outline",
    "xyz_m", "xyz_now_m", "z_m", "z_mm", "depth_source", "width_mm_depth"
)

# ------------------------------------------------------ what the frontend gets
def build_frontend_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    """Wheel result dict -> the dict the frontend / API returns (one place, used by both services).
    These functions prepare the raw ML data so the React frontend can understand it.
    Adds: per-detection ``confidence`` / ``class`` / ``status`` aliases, ``total_count``
    (heads + tails visible in THIS frame; use heads_total + tails_total for the running
    total), and passes through ``bigger_tube`` from rvf.assign_sizes().

    Width fields (``width_px``, ``width_mm``, ``width_now_px``, ``width_spread_px``,
    ``width_source``, ``size``) and 3D depth fields (``xyz_m``, ``z_m``, ``z_mm``,
    ``width_mm_depth``, ``depth_source``) are guaranteed to exist on every detection
    (None when not measured), and ``bigger_tube`` is passed through from the wheel (None
    only for an empty frame). An empty result ({} for an empty frame) gives all-zero counts."""
    heads = result.get("heads_count", 0)
    tails = result.get("tails_count", 0)
    detections = []
    for d in result.get("detections", []):
        det = dict(d)
        det["confidence"] = det.get("conf")        # alias only; never a made-up default
        det["class"] = det.get("role")             # alias only
        # the wheel only emits CONFIRMED ends in ``detections``, so "OK" is true for every entry
        det["status"] = d.get("status", "OK")

        # Compute z_mm convenience distance for the frontend
        if det.get("z_m") is not None:
            det["z_mm"] = round(float(det["z_m"]) * 1000.0, 1)
        elif det.get("xyz_m") is not None and isinstance(det["xyz_m"], (list, tuple)) and len(det["xyz_m"]) >= 3 and det["xyz_m"][2] is not None:
            det["z_mm"] = round(float(det["xyz_m"][2]) * 1000.0, 1)
        else:
            det["z_mm"] = None

        # Prioritize true 3D width if width_mm is not yet set from mm_per_px
        if det.get("width_mm") is None and det.get("width_mm_depth") is not None:
            det["width_mm"] = det["width_mm_depth"]

        for key in _WIDTH_DETECTION_KEYS:          # stable schema even if nothing was measured
            det.setdefault(key, None)
        detections.append(det)
    return {
        **result,
        "heads_count": heads,
        "tails_count": tails,
        "heads_total": result.get("heads_total", 0),
        "tails_total": result.get("tails_total", 0),
        "heads_visible": result.get("heads_visible", heads),
        "tails_visible": result.get("tails_visible", tails),
        "total_count": heads + tails,
        "bigger_tube": result.get("bigger_tube"),   # from rvf.assign_sizes(); None for an empty frame
        "smaller_tube": result.get("smaller_tube"),   # the wheel has no such field -> None, kept for the frontend schema
        "detections": detections,
    }



def new_stats(**fields: Any) -> Dict[str, Any]:
    """Zeroed stats dict for a new session (same keys the frontend reads every frame)."""
    stats = {
        "status": "queued",
        "frame": 0,
        "frames_processed": 0,
        "fps": 0.0,
        "frame_time_ms": None,
        "latency_ms": None,           # legacy key, same value as frame_time_ms
        "video_width": 0,
        "video_height": 0,
        **build_frontend_stats({}),
    }
    stats.update(fields)
    return stats

#################### 5.The TubeAnalyzer Class & Frame Analysis ####################
@dataclass
class FrameAnalysis:
    """Everything one analyze_frame() call produced.

    result     the wheel's result dict ({} for an empty frame), untouched
    frame      the frame that was analysed (the same object, NOT a copy)
    annotated  the picture the wheel drew for THIS frame (only when annotate=True), else None.
               It is drawn inside the same call that produced ``result``, so its labels always
               agree with ``result`` (no later re-read of live tracker state).
    """
    result: Dict[str, Any]
    frame: Optional[np.ndarray]
    annotated: Optional[np.ndarray] = None


# Settings worth printing / diffing against the CLI when two runs disagree.
_SETTINGS_TO_REPORT = (
    "model_path", "device", "conf", "conf_head", "conf_tail", "iou", "scale", "imgsz", "tile",
    "tile_overlap", "merge_iou", "merge_contain", "merge_gap", "cross_role_contain", "gray",
    "stride", "min_frames", "min_frames_tail", "max_dist", "max_dist_tail", "max_missed",
    "max_heads", "max_tails", "width_method", "width_from", "min_ratio", "mm_per_px",
    "attach_gap", "max_disagree", "draw_masks", "draw_roi", "box_fill", "draw_scale",
    "camera_matrix", "calib_width", "calib_height", "dist_coeffs", "depth_scale",
    "depth_min_mm", "depth_max_mm", "depth_smooth", "depth_video",
)


# ---------------------------------------------------------------- the analyzer
def _build_settings(config_path: Optional[str], model_path: str, device: str,
                    draw_roi: bool, chosen: Dict[str, Any], extra: Dict[str, Any]):
    """wheel DEFAULTS <- config.yaml <- explicit arguments (later wins).

    This is the wheel's OWN loader (``rvf.load_stream_config``), not a copy of it, so a given
    config.yaml yields the same settings here as in the CLI / HeadTailAnalyzer."""
    if not config_path or not os.path.exists(config_path):
        # Falling back to the wheel's built-in defaults would silently drop the ML team's tuned
        # thresholds (conf_tail, min_frames, scale ...) and change the results.
        raise FileNotFoundError(f"config.yaml not found: {config_path!r} (it holds the ML team's settings)")

    overrides = {k: v for k, v in chosen.items() if v is not None}
    unknown = sorted(k for k in extra if k not in rvf.DEFAULTS)
    if unknown:
        logger.warning(f"TubeAnalyzer ignoring settings the wheel does not have: {unknown}")
    overrides.update({k: v for k, v in extra.items() if k in rvf.DEFAULTS and v is not None})
    overrides.update(model_path=model_path, device=device, draw_roi=draw_roi, print_frames=False)

    cfg, _roi = rvf.load_stream_config(config_path, use_roi=False, **overrides)   # roi: see __init__
    # The CLI overwrites a "latest frame" JSON file or history.jsonl when config.yaml names one.
    # The backend delivers results through the API / session folders instead, so never let a CLI
    # config make the server write these files into the code directory.
    cfg.latest_json = None
    cfg.history_jsonl = None
    return cfg


class TubeAnalyzer:
    """One instance = one video / stream worth of tracker state. Not thread-safe."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "0",
        conf: Optional[float] = None,
        tile: Optional[int] = None,
        tile_overlap: Optional[float] = None,
        gray: Optional[bool] = None,
        min_frames: Optional[int] = None,
        min_frames_tail: Optional[int] = None,
        max_dist: Optional[float] = None,
        max_dist_tail: Optional[float] = None,
        max_missed: Optional[int] = None,
        scale: Optional[float] = None,
        conf_head: Optional[float] = None,
        conf_tail: Optional[float] = None,
        width_method: Optional[str] = None,     # auto | tubing | mask | off
        width_from: Optional[str] = None,       # tail | all  (which ends decide bigger_tube)
        min_ratio: Optional[float] = None,      # width ratio needed to call one BIGGER
        mm_per_px: Optional[float] = None,      # calibration -> width_mm
        draw_scale: Optional[float] = None,
        box_fill: Optional[float] = None,
        draw_masks: Optional[bool] = None,
        draw_unconfirmed: Optional[bool] = None,
        draw_overlay: bool = False,
        draw_roi: bool = False,
        **extra: Any,
    ):
        """Settings left as None use config.yaml or the wheel's DEFAULTS. ``extra`` may carry any
        other wheel setting (iou, merge_iou, merge_contain, max_heads, attach_gap, ...); unknown keys are ignored."""
        if rvf is None:
            raise RuntimeError(
                f"The head-tail-analyzer wheel could not be imported: {_WHEEL_ERROR}. "
                "Install the wheel (pip install head_tail_analyzer_v-*.whl)."
            )

        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        target_device = str(device)
        if target_device != "cpu" and not cuda_is_available():
            logger.warning(f"CUDA requested (device={target_device!r}) but not available; falling back to CPU")
            target_device = "cpu"

        chosen = dict(conf=conf, tile=tile, tile_overlap=tile_overlap, gray=gray,
                      min_frames=min_frames, min_frames_tail=min_frames_tail,
                      max_dist=max_dist, max_dist_tail=max_dist_tail, max_missed=max_missed,
                      scale=scale, conf_head=conf_head, conf_tail=conf_tail,
                      width_method=width_method, width_from=width_from,
                      min_ratio=min_ratio, mm_per_px=mm_per_px,
                      draw_scale=draw_scale, box_fill=box_fill,
                      draw_masks=draw_masks, draw_unconfirmed=draw_unconfirmed)
        cfg_path = config_path or CONFIG_PATH
        self.cfg = _build_settings(cfg_path, path, target_device, draw_roi, chosen, extra)
        self._warn_if_config_has_roi(cfg_path)

        self.draw_overlay = draw_overlay
        self.draw_roi = draw_roi

        self.model_path = str(path)
        self.model_mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0

        self.entry = _load_entry(path, target_device)       # the wheel's LoadedModel
        self.model = self.entry.model
        self.role_ids = self.entry.role_ids
        self.tubing_id: Optional[int] = self.entry.tubing_id
        # The wheel's own decision (width_method auto -> tubing/mask, non-seg model -> off).
        try:
            self._masks_ok = bool(rvf.configure_for_model(self.cfg, self.entry))
        except rvf.AnalyzerError as e:
            raise RuntimeError(str(e)) from None
        self._masks_ok = self._masks_ok and bool(self.cfg.draw_masks)   # same as HeadTailAnalyzer

        # Check if underlying rvf.analyze_frame supports depth parameter
        import inspect
        try:
            self._depth_supported = "depth" in inspect.signature(rvf.analyze_frame).parameters
        except Exception:
            self._depth_supported = False

        # Per-session ROI: start with full frame (no ROI). ROI is only
        # applied when the user explicitly draws one via set_roi().
        self.roi: Optional[np.ndarray] = None
        self._pending_roi: Optional[Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]] = None
        self.state = rvf.StreamState()      # tracks, ids, peak counts, fps, frame_total, video_fps: ALL the wheel's
        self._last_analysis: Optional[FrameAnalysis] = None
        self._warmed_shapes: Set[Tuple[int, int]] = _GLOBAL_WARMED_SHAPES
        if self.cfg.width_method == "mask" and self.cfg.width_from == "all":
            logger.warning("width_from=all compares connector width with tube width; "
                           "use width_from=tail for a fair 'bigger tube'")
        logger.info(f"TubeAnalyzer ready: model={path} classes={self.model.names} "
                    f"using={self.role_ids} device={self.cfg.device} tile={self.cfg.tile or 'off'}")
        logger.info("TubeAnalyzer settings: %s", json.dumps(self.describe_settings(), default=str, sort_keys=True))

        # Pre-warm directly on standard 1080p camera/video frame shape so the first live frame has zero delay.
        # (Only CUDA start-up; it has no effect on results.)
        self.warm_up((1080, 1920))

    # ---------------------------------------------------------- introspection
    def describe_settings(self) -> Dict[str, Any]:
        """What this analyzer is really running with + which pipeline it uses. Compare with
        ``vars(HeadTailAnalyzer(config).cfg)`` from the CLI side when two runs disagree."""
        d = {k: getattr(self.cfg, k, None) for k in _SETTINGS_TO_REPORT}
        d["pipeline"] = dict(WHEEL_INFO)
        d["roi_active"] = self.roi is not None
        return d

    def _warn_if_config_has_roi(self, cfg_path: str) -> None:
        """The CLI applies ``roi_json`` from config.yaml; this backend applies only the ROI the
        user draws in the UI. Say so, because it changes which ends are counted."""
        try:
            import yaml
            data = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
            if data.get("roi_json"):
                logger.warning("config.yaml sets roi_json=%r but the backend IGNORES it (ROI comes from the UI). "
                               "CLI runs WILL count only inside that polygon; counts can differ.", data["roi_json"])
        except Exception:
            pass

    # ---------------------------------------------- state (the wheel's StreamState)
    @property
    def tracks(self) -> list:
        return self.state.tracks

    @property
    def next_id(self) -> int:
        return self.state.next_id

    @property
    def peak(self) -> Dict[str, int]:
        return self.state.peak

    # ------------------------------------------------------------- lifecycle
    def warm_up(self, frame_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Run the model once on a black frame so CUDA initialisation does not land in the first
        real frame's timing. Runs once per frame size and caches in _warmed_shapes."""
        if self.cfg.device == "cpu":
            return
        if frame_shape is not None and len(frame_shape) >= 2:
            height, width = int(frame_shape[0]), int(frame_shape[1])
        else:
            height, width = 1080, 1920
        if (height, width) in self._warmed_shapes:
            return
        try:
            dummy = np.zeros((height, width, 3), dtype=np.uint8)
            rvf.detect(self.model, dummy, self.role_ids, self.cfg, self.tubing_id, lock=self.entry.infer_lock)
            if cuda_is_available():
                torch.cuda.synchronize()
            self._warmed_shapes.add((height, width))
            logger.info(f"TubeAnalyzer warm-up done for {width}x{height}")
        except Exception as e:
            logger.debug(f"TubeAnalyzer warm-up skipped: {e}")

    def has_weights_changed(self) -> bool:
        """Check if the underlying .pt file has been modified on disk since this analyzer was loaded."""
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        try:
            return os.path.getmtime(self.model_path) != self.model_mtime
        except Exception:
            return False

    def set_calibration(self, mm_per_px: Optional[float]) -> None:
        """Set (or clear with None) millimetres-per-pixel. Takes effect on the next frame."""
        if mm_per_px is not None:
            mm_per_px = float(mm_per_px)
            if not (mm_per_px > 0 and np.isfinite(mm_per_px)):
                raise ValueError("mm_per_px must be a positive number")
        self.cfg.mm_per_px = mm_per_px

    def set_video_info(self, frame_total: Optional[int], video_fps: Optional[float]) -> None:
        """Tell the wheel about the video (same two values HeadTailAnalyzer.process_video sets), so it
        fills ``frame_total`` / ``video_fps`` in every result and draws ``FRAME n / total``."""
        self.state.frame_total = frame_total
        self.state.video_fps = video_fps

    def reset_tracking(self) -> None:
        """Forget all tracks (call before a new video / stream).
        ROI is cleared back to None (full frame). The user must draw a new
        ROI for it to take effect again."""
        self.state.reset()
        self._last_analysis = None
        self.roi = None
        self._pending_roi = None

    reset = reset_tracking   # old name, still works

    def _retry_on_cpu_after_cuda_error(self, error: Exception) -> bool:
        """Reload the detector on CPU after a CUDA-specific model failure."""
        global _CUDA_KERNELS_USABLE
        message = str(error).lower()
        cuda_failure = any(token in message for token in (
            "cuda", "gpu", "no kernel image", "invalid device function",
            "not implemented for", "could not run", "cudnn",
        ))
        if self.cfg.device == "cpu" or not cuda_failure:
            return False

        logger.warning("CUDA inference failed (%s); reloading the detector on CPU and retrying this frame", error)
        _CUDA_KERNELS_USABLE = False
        self.cfg.device = "cpu"
        self.entry = _load_entry(self.cfg.model_path, "cpu")
        self.model = self.entry.model
        self._warmed_shapes.clear()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return True

    # ------------------------------------------------------------ per frame
    def analyze_frame(self, frame: np.ndarray, frame_number: int = 0, annotate: bool = False, depth: Optional[np.ndarray] = None) -> FrameAnalysis:
        """ONE BGR frame (and optional aligned uint16 mm depth frame) -> the wheel's result
        (and, with annotate=True, the wheel's drawn picture).

        The whole pipeline (detect -> track -> ROI -> sizes -> counts -> fps -> depth/3D -> draw)
        runs inside ``rvf.analyze_frame``; this method only applies a queued ROI swap first and trims
        dead flicker tracks afterwards. ``frame_number`` is the real frame index (tracker ``max_missed``
        is measured in it)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return FrameAnalysis(result={}, frame=frame)

        # Apply queued ROI swap at the start of frame processing
        if self._pending_roi is not None:
            new_roi, size_hint = self._pending_roi
            self._pending_roi = None
            if new_roi is not None and getattr(frame, "shape", None) is not None:
                h, w = frame.shape[:2]
                if size_hint and size_hint != (w, h) and size_hint[0] > 0 and size_hint[1] > 0:
                    sx = w / float(size_hint[0])
                    sy = h / float(size_hint[1])
                    new_roi = np.round(new_roi.astype(np.float64) * [sx, sy]).astype(np.int32)
            self.roi = new_roi
            for t in self.state.tracks:
                t.in_roi_ever = False

        def _run():
            if self._depth_supported:
                return rvf.analyze_frame(self.entry, frame, self.state, self.roi, self.cfg, frame_number,
                                         annotate=annotate, masks=self._masks_ok, depth=depth)
            return rvf.analyze_frame(self.entry, frame, self.state, self.roi, self.cfg, frame_number,
                                     annotate=annotate, masks=self._masks_ok)
        try:
            result, vis = _run()
        except Exception as error:
            if not self._retry_on_cpu_after_cuda_error(error):
                raise
            result, vis = _run()

        self._drop_stale_tracks(frame_number)
        analysis = FrameAnalysis(result=result, frame=frame, annotated=vis)
        self._last_analysis = analysis
        return analysis

    def infer(self, frame: np.ndarray, frame_number: int = 0, annotate: bool = False, depth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """ONE call: BGR frame (+ optional depth) in -> the complete dict for the frontend out.

        In order:  warm_up(frame.shape)      once per frame size, not timed
                   analyze_frame()           the wheel's pipeline                (timed)
                   build_frontend_stats()    confidence/class/status/total_count/3D depth ...

        ``fps`` is the wheel's own ``result["fps"]`` (the same number the CLI shows). The model
        speed measured around the wheel call is reported separately as ``frame_time_ms`` /
        ``model_fps``. Whatever other keys the wheel puts in its result pass straight through.

        annotate=True also draws the picture inside the wheel call: read it from
        ``analyzer._last_analysis.annotated`` (or ``draw_annotations``)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return new_stats(status="error", reasons=["Empty frame"], frame=frame_number)

        self.warm_up(frame.shape)
        with inference_context():
            t_start = time.perf_counter()
            analysis = self.analyze_frame(frame, frame_number, annotate=annotate, depth=depth)
            if cuda_is_available():
                torch.cuda.synchronize()
            frame_time_ms = (time.perf_counter() - t_start) * 1000.0

        wheel_fps = analysis.result.get("fps")          # None on the very first frame (nothing to time yet)
        return {
            **build_frontend_stats(analysis.result),
            "status": "processing",
            "frame": frame_number,
            "fps": wheel_fps if wheel_fps else 0.0,     # 0.0 = "not known yet", so the UI never gets null
            "model_fps": round(1000.0 / frame_time_ms, 1) if frame_time_ms > 0 else 0.0,
            "frame_time_ms": round(frame_time_ms, 2),
            "latency_ms": round(frame_time_ms, 2),     # legacy key, same value
            "video_width": frame.shape[1],
            "video_height": frame.shape[0],
        }

    def set_roi(self, points: Any, frame_size: Optional[Tuple[int, int]] = None) -> None:
        """Queue an ROI swap between frames (thread-safe for both camera and video).
        Validates in the caller's thread and applies at the start of the next frame.
        """
        validated_roi = roi_from_points(points, frame_size)
        size_hint = (int(frame_size[0]), int(frame_size[1])) if frame_size else None
        self._pending_roi = (validated_roi, size_hint)

    def draw_annotations(self, frame: np.ndarray, analysis: Optional[FrameAnalysis] = None) -> np.ndarray:
        """The picture the wheel drew for an analysis (needs ``annotate=True`` when it was analysed).
        Without one, the frame is returned untouched. Nothing is re-drawn later from live tracker
        state, because that state keeps moving and the labels would stop matching ``result``."""
        analysis = analysis or self._last_analysis
        if analysis is None or analysis.annotated is None:
            return frame
        return analysis.annotated

    def get_clean_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame for live streaming. No boxes (the frontend draws them from ``detections``).
        Returns the SAME object unless the ROI outline is enabled, so do not modify it."""
        if self.draw_roi and self.roi is not None and frame is not None:
            frame = frame.copy()
            cv2.polylines(frame, [self.roi.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        return frame

    # ------------------------------------------------- old API, same behaviour
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, return_clean: bool = False, depth: Optional[np.ndarray] = None):
        """Old combined call, kept so existing callers keep working.

        return_clean=True -> (result, annotated_frame, clean_frame)
        otherwise         -> (result, annotated_frame if draw_overlay else clean_frame)
        New code should use infer() / analyze_frame()."""
        want_drawn = return_clean or self.draw_overlay
        analysis = self.analyze_frame(frame, frame_idx, annotate=want_drawn, depth=depth)
        clean_frame = self.get_clean_frame(frame)
        annotated = analysis.annotated if want_drawn else None
        if return_clean:
            return analysis.result, annotated, clean_frame
        return (analysis.result, annotated) if self.draw_overlay else (analysis.result, clean_frame)

    # -------------------------------------------------------------- internal
    def _drop_stale_tracks(self, frame_number: int) -> None:
        """Drop flicker tracks that are past re-matching. They can never be matched again and
        never counted (unconfirmed), so no result changes; it only stops the list growing forever
        on a long camera session. Confirmed tracks stay: they feed heads_total / tails_total."""
        oldest_live = frame_number - self.cfg.max_missed
        self.state.tracks = [t for t in self.state.tracks if t.confirmed or t.last_seen >= oldest_live]
