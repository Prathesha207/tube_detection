"""
camera_inference_service.py
===========================
Handles live, real-time per-frame inference (camera feeds, OAK streams, RTSP) for Tube tracing.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import torch
except Exception:
    torch = None
from app.ml import app_state
from app.core.logger import setup_logger
from .tube_analyzer import TubeAnalyzer

logger = setup_logger("tube-camera-inference")

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
_SESSION_IDLE_TIMEOUT_SEC = 300

# Path to the actual trained model resolved dynamically
_TUBE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = str(_TUBE_DIR / "model" / "best.pt")

def _get_or_create_session(session_id: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    existing = _sessions.get(session_id)
    if existing is not None and "analyzer" in existing:
        return existing

    with _sessions_lock:
        if session_id in _sessions and "analyzer" in _sessions[session_id]:
            return _sessions[session_id]

        if not app_state.try_enter_inference("camera"):
            active_kind = app_state.get_active_inference_kind()
            if app_state.get_mode() == "TRAINING":
                raise RuntimeError("Training is in progress -- camera inference cannot start")
            raise RuntimeError(f"GPU is currently in use by {active_kind or 'another process'} -- try again shortly")

        try:
            logger.info(f"Creating fresh TubeAnalyzer for camera session {session_id}.")
            path = model_path or _DEFAULT_MODEL_PATH
            is_cuda = bool(torch and torch.cuda.is_available())
            analyzer = TubeAnalyzer(model_path=path, device="0" if is_cuda else "cpu", draw_overlay=True)
        except Exception as e:
            logger.error(f"Failed to create camera analyzer: {e}", exc_info=True)
            app_state.exit_inference("camera")
            raise

        session = {
            "analyzer": analyzer,
            "frames_processed": 0,
            "last_active": time.time(),
            "inference_claimed": True,
            "inference_lock": threading.Lock(),
            "last_stats": {
                "session_id": session_id,
                "status": "processing",
                "frame": 0,
                "frames_processed": 0,
                "fps": 0.0,
                "heads_count": 0,
                "tails_count": 0,
                "detections": [],
                "bigger_tube": None,
                "smaller_tube": None,
                "video_width": 0,
                "video_height": 0,
            }
        }
        _sessions[session_id] = session
        return session

def cleanup_stale_sessions(max_idle_seconds: int = _SESSION_IDLE_TIMEOUT_SEC) -> None:
    now = time.time()
    with _sessions_lock:
        stale = [sid for sid, s in list(_sessions.items()) if now - s.get("last_active", now) > max_idle_seconds]
    for sid in stale:
        logger.info(f"Evicting stale camera session {sid}")
        clear_session(sid)

def run_inference(frame, session_id: str, model_path: Optional[str] = None, video_name: Optional[str] = None, **kwargs) -> Tuple[Dict[str, Any], Any]:
    if app_state.get_mode() == "TRAINING":
        return {"session_id": session_id, "status": "paused", "reasons": ["Training is in progress -- camera inference paused"], "frames_processed": 0}, frame

    try:
        session = _get_or_create_session(session_id, model_path)
    except Exception as e:
        logger.warning(f"Could not start camera session {session_id}: {e}")
        return {"session_id": session_id, "status": "error", "reasons": [str(e)], "frames_processed": 0}, frame

    session["last_active"] = time.time()
    analyzer = session["analyzer"]
    session["frames_processed"] += 1
    
    t_infer_start = time.perf_counter()
    try:
        if torch is not None:
            with torch.inference_mode():
                result, annotated_frame = analyzer.process_frame(frame, session["frames_processed"])
        else:
            result, annotated_frame = analyzer.process_frame(frame, session["frames_processed"])
    except Exception as e:
        logger.error(f"Error in TubeAnalyzer for session {session_id}: {e}", exc_info=True)
        return {"session_id": session_id, "status": "error", "reasons": [str(e)], "frames_processed": session["frames_processed"]}, frame

    infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000
    infer_fps = round(1000.0 / infer_latency_ms, 1) if infer_latency_ms > 0 else 0.0

    stats = {
        "session_id": session_id,
        "status": "processing",
        "frame": session["frames_processed"],
        "frames_processed": session["frames_processed"],
        "fps": infer_fps,
        "latency_ms": round(infer_latency_ms, 1),
        "video_width": frame.shape[1],
        "video_height": frame.shape[0],
        **result
    }

    session["last_stats"] = stats
    return stats, annotated_frame

def get_session_status(session_id: str) -> Optional[Dict[str, Any]]:
    session = _sessions.get(session_id)
    return session.get("last_stats") if session else None

def clear_session(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
        if session:
            if session.get("inference_claimed"):
                app_state.exit_inference("camera")
            logger.info(f"Released TubeAnalyzer resources for camera session {session_id}")

def reset_session_for_next_video(session_id: str) -> None:
    clear_session(session_id)

def _set_camera_analyzer(analyzer):
    pass

