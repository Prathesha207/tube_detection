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

import gc
import json
import logging
import os
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


# --------------------------------------------------------------- small helpers
def cuda_is_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


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


def load_roi(path: Optional[str] = None) -> Optional[np.ndarray]:
    """Polygon from roi.json as an (N, 2) int32 array, or None (missing / invalid / < 3 points)."""
    target = Path(path) if path else Path(ROI_JSON)
    if not target.exists():
        logger.info(f"No ROI file at {target} -- the whole frame is counted")
        return None
    try:
        points = json.loads(target.read_text(encoding="utf-8")).get("points")
        roi = np.array(points, dtype=np.int32) if points else None
    except Exception as e:
        logger.warning(f"Could not read ROI file {target}: {e} -- running without ROI")
        return None
    return roi if roi is not None and roi.ndim == 2 and len(roi) >= 3 else None


def _load_yolo_model(path: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics / PyTorch failed to load. Check the PyTorch installation.") from e
    return YOLO(path)


# ------------------------------------------------------ what the frontend gets
def build_frontend_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    """Wheel result dict -> the dict the frontend / API returns (one place, used by both services).

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
        roi_path: Optional[str] = None,
        draw_overlay: bool = False,
        draw_roi: bool = False,
        **extra: Any,
    ):
        """Settings left as None use config.yaml or the wheel's DEFAULTS. ``extra`` may carry any
        other wheel setting (iou, merge_iou, merge_contain); unknown keys are ignored."""
        if rvf is None:
            raise RuntimeError(
                f"The head-tail-analyzer wheel could not be imported: {_WHEEL_ERROR}. "
                "Install the wheel: pip install head_tail_analyzer-0.1.2-py3-none-any.whl"
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
        self.roi_path = roi_path or ROI_JSON

        self.model = _load_yolo_model(path)
        try:
            self.role_ids = rvf.role_ids_from_names(self.model.names)
        except SystemExit as e:        # the wheel exits instead of raising; SystemExit is not an Exception
            raise RuntimeError(f"Model {path} is not usable for head/tail counting: {e}") from None
        logger.info(f"TubeAnalyzer ready: model={path} classes={self.model.names} "
                    f"using={self.role_ids} device={self.cfg.device} tile={self.cfg.tile or 'off'}")

        self.roi = load_roi(self.roi_path)
        self.tracks: list = []
        self.next_id = 1
        self._last_analysis: Optional[FrameAnalysis] = None
        self._warmed_shapes: Set[Tuple[int, int]] = set()

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
        """Forget all tracks (call before a new video / stream) and re-read the ROI file."""
        self.tracks = []
        self.next_id = 1
        self._last_analysis = None
        self.roi = load_roi(self.roi_path)

    reset = reset_tracking   # old name, still works

    # ------------------------------------------------------------ per frame
    def analyze_frame(self, frame: np.ndarray, frame_number: int = 0) -> FrameAnalysis:
        """Detect -> track -> ROI filter -> result for ONE BGR frame.

        This is the part worth timing. It does no drawing and no frame copy.
        ``frame_number`` is the real frame index (tracker ``max_missed`` is measured in it)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return FrameAnalysis(result={}, frame=frame, detections=[])

        detections = rvf.detect(self.model, frame, self.role_ids, self.cfg)
        self.next_id = rvf.update_tracks(self.tracks, detections, frame_number, self.next_id, self.cfg)
        self._drop_stale_tracks(frame_number)

        inside_roi = [d for d in detections if rvf.in_roi(d, self.roi)]   # ROI limits output only
        for d in inside_roi:
            d["track"].in_roi_ever = True
        result = rvf.build_frame_result(frame_number, inside_roi, self.tracks)

        analysis = FrameAnalysis(result=result, frame=frame, detections=inside_roi)
        self._last_analysis = analysis
        return analysis

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
