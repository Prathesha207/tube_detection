"""
tube_analyzer.py
================
Thin adapter between the backend services and the ``head_tail_analyzer`` wheel.

All ML work happens inside the wheel (``head_tail_analyzer.run_video_frames``).
This file only loads the model, keeps the tracker state for ONE video / stream,
and calls the wheel's functions in order for each frame. The result dict it
returns is exactly what the wheel produces::

    {"frame", "heads_count", "tails_count", "heads_visible", "tails_visible",
     "heads_total", "tails_total",
     "detections": [{"id", "role", "conf", "bbox": [x, y, w, h], "center": [x, y]}]}

Nothing is added to or removed from it. (``frame_time_ms`` is added by the
services, which are the ones that time the ``process_frame()`` call.)
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

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


def load_roi(path: Optional[str] = None) -> Optional[np.ndarray]:
    """Polygon from roi.json as an (N, 2) int32 array, or None (missing / invalid / < 3 points)."""
    target = Path(path) if path else Path(ROI_JSON)
    if not target.exists():
        return None
    try:
        points = json.loads(target.read_text(encoding="utf-8")).get("points")
        roi = np.array(points, dtype=np.int32) if points else None
    except Exception as e:
        logger.warning(f"Could not read ROI file {target}: {e} -- running without ROI")
        return None
    return roi if roi is not None and roi.ndim == 2 and len(roi) >= 3 else None


def _load_yolo(path: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics / PyTorch failed to load. Check the PyTorch installation.") from e
    return YOLO(path)


class TubeAnalyzer:
    """One instance = one video / stream worth of tracker state."""

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
        """Settings left as None use config.yaml or the wheel's DEFAULTS. ``extra`` may carry any other
        wheel setting (iou, merge_iou, merge_contain); unknown keys are ignored."""
        if rvf is None:
            raise RuntimeError(
                f"The head-tail-analyzer wheel could not be imported: {_WHEEL_ERROR}. "
                "Install the wheel: pip install head_tail_analyzer-0.1.2-py3-none-any.whl"
            )

        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        cfg = dict(rvf.DEFAULTS)

        # Load settings from config.yaml if present
        cfg_file = config_path or CONFIG_PATH
        if cfg_file and os.path.exists(cfg_file):
            try:
                import yaml
                data = yaml.safe_load(Path(cfg_file).read_text(encoding="utf-8")) or {}
                for k, v in data.items():
                    if k in cfg and v is not None:
                        cfg[k] = v
            except Exception as e:
                logger.warning(f"Could not read config file {cfg_file}: {e}")

        chosen = dict(conf=conf, tile=tile, tile_overlap=tile_overlap, gray=gray,
                      min_frames=min_frames, max_dist=max_dist, max_missed=max_missed)
        cfg.update({k: v for k, v in chosen.items() if v is not None})
        unknown = sorted(k for k in extra if k not in cfg)
        if unknown:
            logger.warning(f"TubeAnalyzer ignoring settings the wheel does not have: {unknown}")
        cfg.update({k: v for k, v in extra.items() if k in cfg and v is not None})

        # Graceful CPU fallback if CUDA is requested but unavailable
        target_device = str(device)
        if target_device == "0" and not (torch and torch.cuda.is_available()):
            logger.warning("CUDA requested (device='0') but not available; falling back to CPU")
            target_device = "cpu"

        cfg.update(model_path=path, device=target_device, tile=int(cfg["tile"] or 0),
                   draw_roi=draw_roi, roi_json=None)
        self.cfg = SimpleNamespace(**cfg)

        self.draw_overlay = draw_overlay
        self.draw_roi = draw_roi
        self.roi_path = roi_path or ROI_JSON

        self.model = _load_yolo(path)
        try:
            self.role_ids = rvf.role_ids_from_names(self.model.names)
        except SystemExit as e:        # the wheel exits instead of raising; SystemExit is not an Exception
            raise RuntimeError(f"Model {path} is not usable for head/tail counting: {e}") from None
        logger.info(f"TubeAnalyzer ready: model={path} classes={self.model.names} "
                    f"using={self.role_ids} device={self.cfg.device} tile={self.cfg.tile or 'off'}")

        self.roi = load_roi(self.roi_path)
        self.tracks = []
        self.next_id = 1

        # Warm up model if running on GPU so first live frame isn't penalized by CUDA/cuDNN init
        if self.cfg.device != "cpu":
            try:
                dummy_dim = self.cfg.tile or 640
                dummy = np.zeros((dummy_dim, dummy_dim, 3), dtype=np.uint8)
                _ = rvf.detect(self.model, dummy, self.role_ids, self.cfg)
                if torch and torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.debug("TubeAnalyzer warmup pass completed.")
            except Exception as e:
                logger.debug(f"TubeAnalyzer warmup pass skipped: {e}")

    def reset(self) -> None:
        """Forget all tracks (call before a new video / stream) and re-read the ROI file."""
        self.tracks = []
        self.next_id = 1
        self.roi = load_roi(self.roi_path)

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, return_clean: bool = False):
        """Detect -> track -> ROI filter -> the wheel's result dict, for one BGR frame.

        return_clean=True -> (result, annotated_frame, clean_frame)
        otherwise         -> (result, annotated_frame if draw_overlay else clean_frame)
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return ({}, frame, frame) if return_clean else ({}, frame)

        dets = rvf.detect(self.model, frame, self.role_ids, self.cfg)
        self.next_id = rvf.update_tracks(self.tracks, dets, frame_idx, self.next_id, self.cfg)
        self._prune_tracks(frame_idx)

        shown = [d for d in dets if rvf.in_roi(d, self.roi)]     # ROI limits output only, not tracking
        for d in shown:
            d["track"].in_roi_ever = True
        result = rvf.build_frame_result(frame_idx, shown, self.tracks)

        clean_frame = frame.copy()
        if self.draw_roi and self.roi is not None:
            cv2.polylines(clean_frame, [self.roi.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

        annotated_frame = None
        if return_clean or self.draw_overlay:
            annotated_frame = rvf.draw(frame, shown, result, self.roi, self.cfg)

        if return_clean:
            return result, annotated_frame, clean_frame
        return (result, annotated_frame) if self.draw_overlay else (result, clean_frame)

    def _prune_tracks(self, frame_idx: int) -> None:
        """Drop flicker tracks that are past re-matching. They can never be matched again and
        never counted (unconfirmed), so no result changes; it only stops the list growing forever
        on a long camera session. Confirmed tracks stay: they feed heads_total / tails_total."""
        oldest_live = frame_idx - self.cfg.max_missed
        self.tracks = [t for t in self.tracks if t.confirmed or t.last_seen >= oldest_live]
