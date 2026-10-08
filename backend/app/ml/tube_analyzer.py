"""
tube_analyzer.py
================
Thin adapter between the backend services and the ``head_tail_analyzer`` wheel.

The wheel (``head_tail_analyzer.run_video_frames``) already contains all the ML
functions. Its own ``run()`` only works on a video FILE that it opens itself, so
a backend that receives frames from a camera / upload loop calls the same
per-frame functions one by one instead. That is all this file does:

    analyze_frame(frame, frame_number)
        1. rvf.detect()              YOLO on 640px tiles, duplicates merged
        2. rvf.update_tracks()       give every end a stable id across frames
        3. _drop_stale_tracks()      memory cleanup only (does not change results)
        4. rvf.in_roi()              ROI limits what is REPORTED, not what is tracked
        5. rvf.build_frame_result()  counts + detections dict
       -> FrameAnalysis(result, frame, detections)

    draw_annotations(frame, analysis)   boxes / ids / header  (only when you need it)
    get_clean_frame(frame)              the frame for streaming (ROI outline if enabled)

Typical use::

    analyzer = TubeAnalyzer(model_path=..., config_path=..., roi_path=...)
    analyzer.warm_up(frame.shape)                       # once, NOT part of the timing
    t0 = time.perf_counter()
    analysis = analyzer.analyze_frame(frame, n)         # <- this is what you time
    frame_time_ms = (time.perf_counter() - t0) * 1000
    stats = build_frontend_stats(analysis.result)       # the dict the frontend receives

``analysis.result`` is exactly what the wheel produces::

    {"frame", "heads_count", "tails_count", "heads_visible", "tails_visible",
     "heads_total", "tails_total",
     "detections": [{"id", "role", "conf", "bbox": [x, y, w, h], "center": [x, y]}]}

One TubeAnalyzer = one video / stream of tracker state. It is NOT thread-safe:
call it from one thread at a time (the services do this with a lock / a single
worker thread).
"""

from __future__ import annotations
from anyio import to_thread

import gc
import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    import torch
    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = True
except Exception:
    torch = None

_CUDA_KERNELS_USABLE: Optional[bool] = None

try:
    from head_tail_analyzer import run_video_frames as rvf
    _WHEEL_ERROR = None
except ImportError as e:
    rvf = None
    _WHEEL_ERROR = e

logger = logging.getLogger("tube-analyzer")

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
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            _CUDA_KERNELS_USABLE = False
            return False
        probe = torch.ones((1,), device="cuda:0")
        _ = probe + 1
        torch.cuda.synchronize(0)
        del probe
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


def release_gpu_memory() -> None:
    """Call AFTER the last reference to a model is dropped, to give VRAM back."""
    gc.collect()
    if cuda_is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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
_GLOBAL_WARMED_SHAPES: Set[Tuple[int, int]] = set()

def _load_yolo_model(path: str):
    """Load the YOLO model (cached) for fast inference."""
    if path in _GLOBAL_YOLO_CACHE:
        return _GLOBAL_YOLO_CACHE[path]
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics / PyTorch failed to load. Check the PyTorch installation.") from e
    model = YOLO(path)
    _GLOBAL_YOLO_CACHE[path] = model
    return model

#################### 3. Data Formatting ####################
# ------------------------------------------------------ what the frontend gets
def build_frontend_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    """Wheel result dict -> the dict the frontend / API returns (one place, used by both services).
    These functions prepare the raw ML data so the React frontend can understand it.
    Adds: per-detection ``confidence`` / ``class`` / ``status`` aliases, ``total_count``
    (heads + tails visible in THIS frame; use heads_total + tails_total for the running
    total), and the always-None ``bigger_tube`` / ``smaller_tube`` the frontend expects.
    An empty result ({} for an empty frame) gives all-zero counts."""
    heads = result.get("heads_count", 0)
    tails = result.get("tails_count", 0)
    detections = []
    for d in result.get("detections", []):
        det = dict(d)
        det["confidence"] = det.get("conf", 1.0)
        det["class"] = det.get("role", "HEAD")
        det["status"] = "OK"
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
        "bigger_tube": None,
        "smaller_tube": None,
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

    result      the wheel's result dict ({} for an empty frame)
    frame       the frame that was analysed (the same object, NOT a copy)
    detections  the wheel's internal detections inside the ROI, kept so the frame can
                be drawn later without running the model again
    """
    result: Dict[str, Any]
    frame: Optional[np.ndarray]
    detections: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------- the analyzer
def _build_settings(config_path: Optional[str], model_path: str, device: str,
                    draw_roi: bool, chosen: Dict[str, Any], extra: Dict[str, Any]) -> SimpleNamespace:
    """wheel DEFAULTS <- config.yaml <- explicit arguments (later wins)."""
    cfg = dict(rvf.DEFAULTS)

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            for k, v in data.items():
                if k in cfg and v is not None:
                    cfg[k] = v
        except Exception as e:
            logger.warning(f"Could not read config file {config_path}: {e}")

    cfg.update({k: v for k, v in chosen.items() if v is not None})

    unknown = sorted(k for k in extra if k not in cfg)
    if unknown:
        logger.warning(f"TubeAnalyzer ignoring settings the wheel does not have: {unknown}")
    cfg.update({k: v for k, v in extra.items() if k in cfg and v is not None})

    cfg.update(model_path=model_path, device=device, tile=int(cfg["tile"] or 0),
               draw_roi=draw_roi, roi_json=None)   # ROI is loaded by the analyzer, not the wheel
    return SimpleNamespace(**cfg)


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
        max_dist: Optional[float] = None,
        max_missed: Optional[int] = None,
        draw_overlay: bool = False,
        draw_roi: bool = False,
        **extra: Any,
    ):
        """Settings left as None use config.yaml or the wheel's DEFAULTS. ``extra`` may carry any
        other wheel setting (iou, merge_iou, merge_contain); unknown keys are ignored."""
        if rvf is None:
            raise RuntimeError(
                f"The head-tail-analyzer wheel could not be imported: {_WHEEL_ERROR}. "
                "Install the wheel: pip install backend/app/ml/whl/head_tail_analyzer-0.1.3-py3-none-any.whl"
            )

        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        target_device = str(device)
        if target_device != "cpu" and not cuda_is_available():
            logger.warning(f"CUDA requested (device={target_device!r}) but not available; falling back to CPU")
            target_device = "cpu"

        chosen = dict(conf=conf, tile=tile, tile_overlap=tile_overlap, gray=gray,
                      min_frames=min_frames, max_dist=max_dist, max_missed=max_missed)
        self.cfg = _build_settings(config_path or CONFIG_PATH, path, target_device, draw_roi, chosen, extra)

        self.draw_overlay = draw_overlay
        self.draw_roi = draw_roi

        self.model = _load_yolo_model(path)
        try:
            self.role_ids = rvf.role_ids_from_names(self.model.names)
        except SystemExit as e:        # the wheel exits instead of raising; SystemExit is not an Exception
            raise RuntimeError(f"Model {path} is not usable for head/tail counting: {e}") from None
        logger.info(f"TubeAnalyzer ready: model={path} classes={self.model.names} "
                    f"using={self.role_ids} device={self.cfg.device} tile={self.cfg.tile or 'off'}")

        # Per-session ROI: start with full frame (no ROI). ROI is only
        # applied when the user explicitly draws one via set_roi().
        self.roi: Optional[np.ndarray] = None
        self._pending_roi: Optional[Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]] = None
        self.tracks: list = []
        self.next_id = 1
        self._last_analysis: Optional[FrameAnalysis] = None
        self._warmed_shapes: Set[Tuple[int, int]] = _GLOBAL_WARMED_SHAPES

        self.warm_up()   # one tile; call warm_up(frame.shape) again once the real frame size is known

    # ------------------------------------------------------------- lifecycle
    def warm_up(self, frame_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Run the model once on a black frame so CUDA / cuDNN initialisation and the
        autotuner do not land in the first real frame's timing.

        Pass ``frame.shape`` of the real frames: a 1920x1080 frame is split into several
        640px tiles that are sent to the model as ONE batch, and cuDNN autotunes per batch
        shape, so warming up with a single tile does not cover it. Runs once per frame size."""
        if self.cfg.device == "cpu":
            return
        side = self.cfg.tile or 640
        height, width = (int(frame_shape[0]), int(frame_shape[1])) if frame_shape is not None else (side, side)
        if (height, width) in self._warmed_shapes:
            return
        try:
            dummy = np.zeros((height, width, 3), dtype=np.uint8)
            rvf.detect(self.model, dummy, self.role_ids, self.cfg)
            if cuda_is_available():
                torch.cuda.synchronize()
            self._warmed_shapes.add((height, width))
            logger.debug(f"TubeAnalyzer warm-up done for {width}x{height}")
        except Exception as e:
            logger.debug(f"TubeAnalyzer warm-up skipped: {e}")

    def reset_tracking(self) -> None:
        """Forget all tracks (call before a new video / stream).
        ROI is cleared back to None (full frame). The user must draw a new
        ROI for it to take effect again."""
        self.tracks = []
        self.next_id = 1
        self._last_analysis = None
        self.roi = None
        self._pending_roi = None
        self.peak = {}

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
        self.model = _load_yolo_model(self.cfg.model_path)
        self._warmed_shapes.clear()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return True

    # ------------------------------------------------------------ per frame
    """ analyze_frame() This is the core ML loop that runs 30 times a second. It executes exactly in this order:
        rvf.detect(): Scans the image to find Tubes (Heads/Tails).
        rvf.update_tracks(): Compares the Tubes found in this frame to the Tubes found in the last frame, giving them a persistent ID so they aren't double-counted.
        _drop_stale_tracks(): Clears out memory for tubes that walked off-screen.
        rvf.in_roi(): Throws away any Tube that isn't standing inside your drawn Polygon zone.
        rvf.build_frame_result(): Tallies up the final counts and builds the result dictionary."""
    def analyze_frame(self, frame: np.ndarray, frame_number: int = 0) -> FrameAnalysis:
        """Detect -> track -> ROI filter -> result for ONE BGR frame.

        This is the part worth timing. It does no drawing and no frame copy.
        ``frame_number`` is the real frame index (tracker ``max_missed`` is measured in it)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return FrameAnalysis(result={}, frame=frame, detections=[])

        # Apply queued ROI swap at the start of frame processing
        if self._pending_roi is not None:
            new_roi, size_hint = self._pending_roi
            self._pending_roi = None
            if new_roi is not None and frame is not None and getattr(frame, "shape", None) is not None:
                h, w = frame.shape[:2]
                if size_hint and size_hint != (w, h) and size_hint[0] > 0 and size_hint[1] > 0:
                    sx = w / float(size_hint[0])
                    sy = h / float(size_hint[1])
                    new_roi = np.round(new_roi.astype(np.float64) * [sx, sy]).astype(np.int32)
            self.roi = new_roi
            for t in self.tracks:
                t.in_roi_ever = False
        #rvf.detect(): Scans the image to find Tubes (Heads/Tails).
        try:
            detections = rvf.detect(self.model, frame, self.role_ids, self.cfg)
        except Exception as error:
            if not self._retry_on_cpu_after_cuda_error(error):
                raise
            detections = rvf.detect(self.model, frame, self.role_ids, self.cfg)
        #rvf.update_tracks(): Compares the Tubes found in this frame to the Tubes found in the last frame, giving them a persistent ID so they aren't double-counted.
        self.next_id = rvf.update_tracks(self.tracks, detections, frame_number, self.next_id, self.cfg)
        self._drop_stale_tracks(frame_number)
        #rvf.in_roi(): Throws away any Tube that isn't standing inside your drawn Polygon zone.
        inside_roi = [d for d in detections if rvf.in_roi(d, self.roi)]   # ROI limits output only
        inside_roi = rvf.cap_per_role(inside_roi, self.cfg)
        for d in inside_roi:
            d["track"].in_roi_ever = True
        #rvf.build_frame_result(): Tallies up the final counts and builds the result dictionary.
        result = rvf.build_frame_result(frame_number, inside_roi, self.tracks)

        if not hasattr(self, "peak"):
            self.peak = {}
        for role in ["HEAD", "TAIL"]:
            key = role.lower() + "s"
            self.peak[key] = max(self.peak.get(key, 0), result.get(f"{key}_count", 0))
            result[f"{key}_peak"] = self.peak[key]

        analysis = FrameAnalysis(result=result, frame=frame, detections=inside_roi)
        self._last_analysis = analysis
        return analysis

    def infer(self, frame: np.ndarray, frame_number: int = 0) -> Dict[str, Any]:
        """ONE call: BGR frame in -> the complete dict for the frontend out.

        In order:  warm_up(frame.shape)      once per frame size, not timed
                   analyze_frame()           detect -> track -> ROI -> result   (timed)
                   build_frontend_stats()    confidence/class/status/total_count ...

        Whatever keys the wheel puts in its result pass straight through
        (build_frontend_stats spreads ``**result``), so a new wheel field reaches the
        frontend without touching any service code.

        Drawing is NOT done here (an image is not JSON). Afterwards call
        ``draw_annotations(frame)`` -- it reuses the detections of THIS call -- or
        ``get_clean_frame(frame)``."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return new_stats(status="error", reasons=["Empty frame"], frame=frame_number)

        self.warm_up(frame.shape)
        with inference_context():
            t_start = time.perf_counter()
            analysis = self.analyze_frame(frame, frame_number)
            if cuda_is_available():
                torch.cuda.synchronize()
            frame_time_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            **build_frontend_stats(analysis.result),
            "status": "processing",
            "frame": frame_number,
            "fps": round(1000.0 / frame_time_ms, 1) if frame_time_ms > 0 else 0.0,
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
        """Frame with boxes, ids and the counts header drawn on (a new image).

        Not part of analyze_frame() so it is not in the timing and costs nothing when unused.
        Pass ``analysis=None`` to reuse the most recent detections on a NEW frame (for frames
        that were skipped by a stride)."""
        analysis = analysis or self._last_analysis
        if analysis is None or not analysis.result:
            return frame
        return rvf.draw(frame, analysis.detections, analysis.result, self.roi, self.cfg)

    def get_clean_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame for live streaming. No boxes (the frontend draws them from ``detections``).
        Returns the SAME object unless the ROI outline is enabled, so do not modify it."""
        if self.draw_roi and self.roi is not None and frame is not None:
            frame = frame.copy()
            cv2.polylines(frame, [self.roi.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        return frame

    # ------------------------------------------------- old API, same behaviour
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, return_clean: bool = False):
        """Old combined call, kept so existing callers keep working.

        return_clean=True -> (result, annotated_frame, clean_frame)
        otherwise         -> (result, annotated_frame if draw_overlay else clean_frame)
        New code should use analyze_frame() and draw only when needed."""
        analysis = self.analyze_frame(frame, frame_idx)
        clean_frame = self.get_clean_frame(frame)
        annotated = self.draw_annotations(frame, analysis) if (return_clean or self.draw_overlay) else None
        if return_clean:
            return analysis.result, annotated, clean_frame
        return (analysis.result, annotated) if self.draw_overlay else (analysis.result, clean_frame)

    # -------------------------------------------------------------- internal
    def _drop_stale_tracks(self, frame_number: int) -> None:
        """Drop flicker tracks that are past re-matching. They can never be matched again and
        never counted (unconfirmed), so no result changes; it only stops the list growing forever
        on a long camera session. Confirmed tracks stay: they feed heads_total / tails_total."""
        oldest_live = frame_number - self.cfg.max_missed
        self.tracks = [t for t in self.tracks if t.confirmed or t.last_seen >= oldest_live]
