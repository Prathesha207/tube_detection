"""
camera_inference_service.py
===========================
Handles live, real-time per-frame inference (camera feeds, OAK streams, RTSP) for Tube tracing.
"""

import time
import threading
from contextlib import nullcontext
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

# Path to the actual trained model and config resolved dynamically
_ML_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(_ML_DIR / "model" / "best.pt")
_ROI_PATH = str(_ML_DIR / "config" / "roi.json")
_CONFIG_PATH = str(_ML_DIR / "config" / "config.yaml")

def _get_or_create_session(session_id: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    cleanup_stale_sessions()
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
            analyzer = TubeAnalyzer(
                model_path=path,
                config_path=_CONFIG_PATH,
                roi_path=_ROI_PATH,
                device="0" if is_cuda else "cpu",
                draw_overlay=False,
            )
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
                "frame_time_ms": None,
                "latency_ms": None,
                "heads_count": 0,
                "tails_count": 0,
                "heads_total": 0,
                "tails_total": 0,
                "heads_visible": 0,
                "tails_visible": 0,
                "total_count": 0,
                "bigger_tube": None,
                "smaller_tube": None,
                "detections": [],
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

    try:
        # One frame at a time per session: the analyzer's tracker state is not thread-safe.
        with session["inference_lock"]:
            frame_idx = session["frames_processed"] + 1
            with (torch.inference_mode() if torch is not None else nullcontext()):
                # frame_time_ms covers ONLY the process_frame() call: not frame reading,
                # not waiting for the lock, not encoding.
                t_start = time.perf_counter()
                result, annotated_frame, clean_frame = analyzer.process_frame(frame, frame_idx, return_clean=True)
                frame_time_ms = (time.perf_counter() - t_start) * 1000.0
            session["frames_processed"] = frame_idx
    except Exception as e:
        logger.error(f"Error in TubeAnalyzer for session {session_id}: {e}", exc_info=True)
        return {"session_id": session_id, "status": "error", "reasons": [str(e)], "frames_processed": session["frames_processed"]}, frame

    fps = round(1000.0 / frame_time_ms, 1) if frame_time_ms > 0 else 0.0

    detections = []
    for d in result.get("detections", []):
        det = dict(d)
        det["confidence"] = det.get("conf", 1.0)
        det["class"] = det.get("role", "HEAD")
        det["status"] = "OK"
        detections.append(det)

    stats = {
        **result,
        "session_id": session_id,
        "status": "processing",
        "frame": frame_idx,
        "frames_processed": frame_idx,
        "fps": fps,
        "frame_time_ms": round(frame_time_ms, 2),
        "latency_ms": round(frame_time_ms, 2),   # legacy key, same value as frame_time_ms
        "video_width": frame.shape[1],
        "video_height": frame.shape[0],
        "total_count": result.get("heads_count", 0) + result.get("tails_count", 0),
        "heads_total": result.get("heads_total", 0),
        "tails_total": result.get("tails_total", 0),
        "heads_visible": result.get("heads_visible", result.get("heads_count", 0)),
        "tails_visible": result.get("tails_visible", result.get("tails_count", 0)),
        "bigger_tube": None,
        "smaller_tube": None,
        "detections": detections,
    }

    session["last_stats"] = stats
    session["last_annotated_frame"] = annotated_frame
    return stats, clean_frame

def get_session_status(session_id: str) -> Optional[Dict[str, Any]]:
    session = _sessions.get(session_id)
    return session.get("last_stats") if session else None

def clear_session(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session:
        lock = session.get("inference_lock")
        if lock:
            with lock:
                _free_session_resources(session, session_id)
        else:
            _free_session_resources(session, session_id)

def _free_session_resources(session: Dict[str, Any], session_id: str) -> None:
    try:
        session["analyzer"] = None
        if session.get("inference_claimed"):
            app_state.exit_inference("camera")
            session["inference_claimed"] = False
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Released TubeAnalyzer resources for camera session {session_id}")
    except Exception as e:
        logger.warning(f"Error freeing resources for session {session_id}: {e}")

def reset_session_for_next_video(session_id: str) -> None:
    clear_session(session_id)

def _set_camera_analyzer(analyzer):
    pass

