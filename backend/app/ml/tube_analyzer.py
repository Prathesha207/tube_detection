"""
tube_analyzer.py
================
Thin adapter between the backend services and the ``head_tail_analyzer`` wheel.

THE WHEEL DOES THE ML. This file only adds what a backend needs around it:

    * model cache + hot-reload (weights file changed on disk)
    * CUDA probe / CPU fallback
    * per-session ROI (drawn by the user) applied between frames
    * per-frame calls and backend timing metadata

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

The installed wheel is REQUIRED. If it is missing or lacks the required API, startup fails.

``analysis.result`` is exactly what the wheel produces::

    {"frame", "heads_count", "tails_count", "heads_visible", "tails_visible",
     "heads_total", "tails_total", "heads_peak", "tails_peak", "frame_total", "fps",
     "video_fps", "depth", "bigger_tube",
     "detections": [{"id", "role", "conf", "bbox", "center", "width_px", "width_mm",
                     "width_now_px", "width_spread_px", "width_source", "size",
                     "xyz_m", "xyz_now_m", "z_m", "depth_source", "width_mm_depth"}]}

One TubeAnalyzer = one video / stream of tracker state. It is NOT thread-safe:
call it from one thread at a time (the services do this with a lock / a single
worker thread).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from app.utils.roi_utils import roi_from_points

try:
    import torch
except Exception:
    torch = None

_CUDA_KERNELS_USABLE: Optional[bool] = None

_REQUIRED_WHEEL_FUNCS = (
    "analyze_frame", "StreamState", "configure_for_model", "load_stream_config",
    "ModelManager", "load_model", "unload_model", "open_video", "depth3d", "DEFAULTS",
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
    if "analyze_frame" in dir(_wheel_rvf):
        import inspect
        if "depth" not in inspect.signature(_wheel_rvf.analyze_frame).parameters:
            _missing.append("analyze_frame(depth=...)")
    if _missing:
        raise ImportError(f"installed head_tail_analyzer is too old, missing: {_missing}")
    rvf = _wheel_rvf
    WHEEL_INFO = {"source": "wheel", "version": getattr(_hta, "__version__", "?"), "file": rvf.__file__}
    try:
        wheel_file = Path(rvf.__file__).resolve()
        WHEEL_INFO["module_mtime"] = wheel_file.stat().st_mtime
        WHEEL_INFO["module_sha256"] = hashlib.sha256(wheel_file.read_bytes()).hexdigest()
    except OSError:
        WHEEL_INFO["module_mtime"] = None
        WHEEL_INFO["module_sha256"] = None
except Exception as e:
    _WHEEL_ERROR = e

if rvf is not None:
    logger.info("head_tail_analyzer pipeline: %s v%s  (%s)", WHEEL_INFO.get("source"),
                WHEEL_INFO.get("version"), WHEEL_INFO.get("file"))


_ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(_ML_DIR / "model" / "best.pt")
CONFIG_PATH = str(_ML_DIR / "config" / "config.yaml")
_MODEL_MANAGER = rvf.ModelManager(verbose=False) if rvf is not None else None
_MODEL_MTIMES: Dict[str, float] = {}

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
        if rvf is not None and _MODEL_MANAGER is not None:
            rvf.unload_model(manager=_MODEL_MANAGER)
        _MODEL_MTIMES.clear()
    gc.collect()
    if cuda_is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def clear_yolo_cache() -> None:
    """Explicitly release all cached YOLO models to completely free GPU VRAM."""
    release_gpu_memory(clear_cache=True)

#################### 4. Model lifecycle delegated to the wheel ####################
def _load_entry(path: str, device: str = "cpu"):
    """Get the wheel-owned model entry and reload it when the weights file changes."""
    if rvf is None or _MODEL_MANAGER is None:
        raise RuntimeError(f"head_tail_analyzer wheel is unavailable: {_WHEEL_ERROR}")
    path = str(Path(path).resolve())
    mtime = os.path.getmtime(path)
    if path in _MODEL_MTIMES and _MODEL_MTIMES[path] != mtime:
        rvf.unload_model(path, manager=_MODEL_MANAGER)
    entry = rvf.load_model(path, device=device, manager=_MODEL_MANAGER)
    _MODEL_MTIMES[path] = mtime
    return entry


#################### 3. Session Lifecycle Placeholder ####################
def new_stats(**fields: Any) -> Dict[str, Any]:
    """Lifecycle placeholder for a session before the wheel returns its first frame."""
    stats = {
        "status": "queued",
        "frame": 0,
        "frames_processed": 0,
        "fps": 0.0,
        "frame_time_ms": None,
        "latency_ms": None,           # legacy key, same value as frame_time_ms
        "video_width": 0,
        "video_height": 0,
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

        # Per-session ROI: start with full frame (no ROI). ROI is only
        # applied when the user explicitly draws one via set_roi().
        self.roi: Optional[np.ndarray] = None
        self._pending_roi: Optional[Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]] = None
        self.state = rvf.StreamState()      # tracks, ids, peak counts, fps, frame_total, video_fps: ALL the wheel's
        self._last_analysis: Optional[FrameAnalysis] = None
        if self.cfg.width_method == "mask" and self.cfg.width_from == "all":
            logger.warning("width_from=all compares connector width with tube width; "
                           "use width_from=tail for a fair 'bigger tube'")
        logger.info(f"TubeAnalyzer ready: model={path} classes={self.model.names} "
                    f"using={self.role_ids} device={self.cfg.device} tile={self.cfg.tile or 'off'}")
        logger.info("TubeAnalyzer settings: %s", json.dumps(self.describe_settings(), default=str, sort_keys=True))

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

    def set_camera_matrix(self, matrix: Any, calib_width: int, calib_height: int, dist_coeffs: Optional[Any] = None) -> None:
        """Dynamically update camera intrinsics (e.g. from OAK EEPROM) for the active resolution."""
        if matrix is not None:
            self.cfg.camera_matrix = matrix
            self.cfg.calib_width = int(calib_width)
            self.cfg.calib_height = int(calib_height)
            if dist_coeffs is not None:
                self.cfg.dist_coeffs = dist_coeffs

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
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return True

    # ------------------------------------------------------------ per frame
    def analyze_frame(self, frame: np.ndarray, frame_number: Optional[int] = None, annotate: bool = False, depth: Optional[np.ndarray] = None) -> FrameAnalysis:
        """ONE BGR frame (and optional aligned uint16 mm depth frame) -> the wheel's result
        (and, with annotate=True, the wheel's drawn picture).

        The whole pipeline (detect -> track -> ROI -> sizes -> counts -> fps -> depth/3D -> draw)
        runs inside ``rvf.analyze_frame``; this method applies a queued UI ROI before the call.
        ``frame_number`` is the real frame index (tracker ``max_missed``
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
            return rvf.analyze_frame(self.entry, frame, self.state, self.roi, self.cfg, frame_number,
                                     annotate=annotate, masks=self._masks_ok, depth=depth)
        try:
            result, vis = _run()
        except Exception as error:
            if not self._retry_on_cpu_after_cuda_error(error):
                raise
            result, vis = _run()

        self._prune_expired_unconfirmed_tracks()
        analysis = FrameAnalysis(result=result, frame=frame, annotated=vis)
        self._last_analysis = analysis
        return analysis

    def infer(self, frame: np.ndarray, frame_number: Optional[int] = None, annotate: bool = False, depth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """ONE call: BGR frame (+ optional depth) in -> the complete dict for the frontend out.

        In order:  analyze_frame()           the wheel's pipeline                (timed)
                   wheel result pass-through plus backend execution metadata

        ``fps`` is the wheel's own ``result["fps"]`` (the same number the CLI shows). The model
        speed measured around the wheel call is reported separately as ``frame_time_ms`` /
        ``model_fps``. Whatever other keys the wheel puts in its result pass straight through.

        annotate=True also draws the picture inside the wheel call: read it from
        ``analyzer._last_analysis.annotated`` (or ``draw_annotations``)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return new_stats(status="error", reasons=["Empty frame"], frame=frame_number)

        with inference_context():
            t_start = time.perf_counter()
            analysis = self.analyze_frame(frame, frame_number, annotate=annotate, depth=depth)
            if cuda_is_available():
                torch.cuda.synchronize()
            frame_time_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            **analysis.result,
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

    def _prune_expired_unconfirmed_tracks(self) -> None:
        """Bound tracker memory using the wheel's own max_missed setting.

        The wheel excludes expired tracks from matching, but retains them in StreamState.
        Pruning only expired, unconfirmed entries does not change the wheel result or totals.
        """
        oldest_live_frame = self.state.frame_index - self.cfg.max_missed
        self.state.tracks = [
            track for track in self.state.tracks
            if track.confirmed or track.last_seen >= oldest_live_frame
        ]
