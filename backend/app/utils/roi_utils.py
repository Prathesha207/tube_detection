from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("tube-roi")
ROI_JSON = str(Path(__file__).resolve().parent.parent / "ml" / "config" / "roi.json")

#################### UI ROI persistence and validation ####################
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

