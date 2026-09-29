"""
roi_router.py
=============
Router for managing Region of Interest (ROI) configuration.
Supports fetching and updating the inspection mask points with
atomic persistence and runtime hot-swap for both live camera
and video inference pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app.ml.tube_analyzer import ROI_JSON, roi_from_points, save_roi, load_roi
from app.ml.camera_inference_service import set_camera_roi, _sessions as camera_sessions, _sessions_lock as camera_sessions_lock
from app.ml.video_inference_service import set_video_roi, video_inference_service

logger = logging.getLogger("roi-router")
router = APIRouter()


class RoiPayload(BaseModel):
    points: Optional[List[List[float]]] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    session_id: Optional[str] = None
    source_type: Optional[str] = None  # "camera", "video", or None


@router.get("")
@router.get("/")
def get_roi_config(
    session_id: Optional[str] = None,
    stream_width: Optional[int] = Query(None, alias="stream_width"),
    stream_height: Optional[int] = Query(None, alias="stream_height"),
):
    """Return currently saved ROI polygon and dimensions.
    Handles missing file, empty points, and legacy files without frame dimensions
    by falling back to the active stream/session size.
    """
    roi_file = Path(ROI_JSON)
    if not roi_file.exists():
        return {"points": [], "frame_width": None, "frame_height": None}

    try:
        data = json.loads(roi_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read roi.json: {e}")
        return {"points": [], "frame_width": None, "frame_height": None}

    points = data.get("points") or []
    fw = data.get("frame_width")
    fh = data.get("frame_height")

    # Fallback to stream / session size if legacy file has no frame dimensions
    if (fw is None or fh is None) and points:
        fallback_w = stream_width
        fallback_h = stream_height

        # Try session_id lookup
        if (not fallback_w or not fallback_h) and session_id:
            with video_inference_service._sessions_lock:
                vs = video_inference_service.sessions.get(session_id)
                if vs and vs.get("stats"):
                    fallback_w = vs["stats"].get("video_width") or fallback_w
                    fallback_h = vs["stats"].get("video_height") or fallback_h
            if not fallback_w or not fallback_h:
                with camera_sessions_lock:
                    cs = camera_sessions.get(session_id)
                    if cs and cs.get("last_stats"):
                        fallback_w = cs["last_stats"].get("video_width") or fallback_w
                        fallback_h = cs["last_stats"].get("video_height") or fallback_h

        # Fallback to any active session if still unknown
        if not fallback_w or not fallback_h:
            with camera_sessions_lock:
                for s in camera_sessions.values():
                    ls = s.get("last_stats")
                    if ls and ls.get("video_width"):
                        fallback_w = ls.get("video_width")
                        fallback_h = ls.get("video_height")
                        break
        if not fallback_w or not fallback_h:
            with video_inference_service._sessions_lock:
                for s in video_inference_service.sessions.values():
                    st = s.get("stats")
                    if st and st.get("video_width"):
                        fallback_w = st.get("video_width")
                        fallback_h = st.get("video_height")
                        break

        fw = fallback_w
        fh = fallback_h

    return {
        "points": points,
        "frame_width": fw,
        "frame_height": fh,
    }


@router.post("")
@router.post("/")
def save_roi_config(payload: RoiPayload):
    """Validate, save, and hot-swap the ROI for camera and/or video inference."""
    frame_size = None
    if payload.frame_width and payload.frame_height and payload.frame_width > 0 and payload.frame_height > 0:
        frame_size = (int(payload.frame_width), int(payload.frame_height))

    # Validation in caller thread -> returns 422 if invalid
    try:
        validated = roi_from_points(payload.points, frame_size)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    clean_points = validated.tolist() if validated is not None else []

    try:
        # Hot-swap and persist
        if payload.source_type == "camera":
            set_camera_roi(payload.session_id, clean_points, frame_size)
        elif payload.source_type == "video":
            set_video_roi(payload.session_id, clean_points, frame_size)
        else:
            # Shared: update both services and save
            set_camera_roi(payload.session_id, clean_points, frame_size)
            set_video_roi(payload.session_id, clean_points, frame_size)

        logger.info(f"ROI updated: {len(clean_points)} points, frame_size={frame_size}")
        return {
            "status": "ok",
            "message": "ROI updated successfully",
            "points": clean_points,
            "frame_width": payload.frame_width,
            "frame_height": payload.frame_height,
        }
    except Exception as e:
        logger.error(f"Failed to update ROI: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update ROI: {e}")
