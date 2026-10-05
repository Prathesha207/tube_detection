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
def get_roi_config(
    session_id: Optional[str] = None,
    stream_width: Optional[int] = Query(None, alias="stream_width"),
    stream_height: Optional[int] = Query(None, alias="stream_height"),
):
    """Return in-memory ROI polygon and dimensions for the active session.
    Every new session starts with no ROI (full frame, points: []).
    """
    points: List[Any] = []
    fw = stream_width
    fh = stream_height

    # Retrieve in-memory ROI for the target session if drawn by the user
    if session_id:
        with video_inference_service._sessions_lock:
            vs = video_inference_service.sessions.get(session_id)
            if vs:
                points = vs.get("roi_points") or []
                if vs.get("roi_frame_size"):
                    fw, fh = vs["roi_frame_size"]
                elif vs.get("stats"):
                    fw = vs["stats"].get("video_width") or fw
                    fh = vs["stats"].get("video_height") or fh

        if not points:
            with camera_sessions_lock:
                cs = camera_sessions.get(session_id)
                if cs:
                    points = cs.get("roi_points") or []
                    if cs.get("roi_frame_size"):
                        fw, fh = cs["roi_frame_size"]
                    elif cs.get("last_stats"):
                        fw = cs["last_stats"].get("video_width") or fw
                        fh = cs["last_stats"].get("video_height") or fh

    return {
        "points": points,
        "frame_width": fw,
        "frame_height": fh,
    }


@router.post("")
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
