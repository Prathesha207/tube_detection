"""
camera_inference_service.py
===========================
Live, real-time per-frame inference (camera feeds, OAK streams, RTSP) for Tube tracing.

One session_id = one TubeAnalyzer (its own YOLO model + tracker state) = one GPU claim
in app_state ("camera"). Public entry points:

    analyze_camera_frame(frame, session_id)   -> (stats dict, clean frame)
    get_camera_session_stats(session_id)      -> last stats dict, or None
    close_camera_session(session_id)          -> free the model and release the GPU claim
    evict_idle_camera_sessions()              -> close sessions that stopped sending frames

The old names (run_inference, get_session_status, clear_session, cleanup_stale_sessions,
reset_session_for_next_video) are kept at the bottom as aliases.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except Exception:
    torch = None
from app.ml import app_state
from app.core.logger import setup_logger
from .tube_analyzer import (
    TubeAnalyzer,
    build_frontend_stats,
    cuda_is_available,
    default_device,
    inference_context,
    new_stats,
    release_gpu_memory,
    save_roi,
    load_roi,
)

logger = setup_logger("tube-camera-inference")

_ML_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(_ML_DIR / "model" / "best.pt")
_ROI_PATH = str(_ML_DIR / "config" / "roi.json")
_CONFIG_PATH = str(_ML_DIR / "config" / "config.yaml")

_SESSION_IDLE_TIMEOUT_SEC = 300     # no frame for this long -> session is closed, GPU released
_IDLE_SWEEP_INTERVAL_SEC = 30       # how often the background thread looks for idle sessions

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
_idle_sweeper: Optional[threading.Thread] = None


# ------------------------------------------------------------------ sessions
def _start_idle_sweeper_once() -> None:
    """Background thread that closes idle sessions. Without it, a browser tab that is simply
    closed keeps the 'camera' GPU claim (and the model in VRAM) until some other frame arrives,
    which blocks video uploads and training in the meantime. Call with _sessions_lock held."""
    global _idle_sweeper
    if _idle_sweeper is not None and _idle_sweeper.is_alive():
        return

    def _loop():
        while True:
            time.sleep(_IDLE_SWEEP_INTERVAL_SEC)
            try:
                evict_idle_camera_sessions()
            except Exception:
                logger.exception("Idle camera session sweep failed")

    _idle_sweeper = threading.Thread(target=_loop, name="camera-idle-sweeper", daemon=True)
    _idle_sweeper.start()


def _get_or_create_camera_session(session_id: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    existing = _sessions.get(session_id)
    if existing is not None and existing.get("analyzer") is not None:
        return existing

    with _sessions_lock:
        existing = _sessions.get(session_id)
        if existing is not None and existing.get("analyzer") is not None:
            return existing

        if not app_state.try_enter_inference("camera"):
            if app_state.get_mode() == "TRAINING":
                raise RuntimeError("Training is in progress -- camera inference cannot start")
            active_kind = app_state.get_active_inference_kind()
            raise RuntimeError(f"GPU is currently in use by {active_kind or 'another process'} -- try again shortly")

        try:
            logger.info(f"Creating TubeAnalyzer for camera session {session_id}")
            analyzer = TubeAnalyzer(
                model_path=model_path or _DEFAULT_MODEL_PATH,
                config_path=_CONFIG_PATH,
                roi_path=_ROI_PATH,
                device=default_device(),
                draw_overlay=False,
            )
        except Exception as e:
            logger.error(f"Failed to create camera analyzer: {e}", exc_info=True)
            app_state.exit_inference("camera")      # never leak the claim
            raise

        session = {
            "analyzer": analyzer,
            "frames_processed": 0,
            "last_active": time.time(),
            "inference_claimed": True,
            "inference_lock": threading.Lock(),
            "last_stats": new_stats(session_id=session_id, status="processing"),
            "last_annotated_frame": None,
            "roi_points": [],
            "roi_frame_size": None,
        }
        _sessions[session_id] = session
        _start_idle_sweeper_once()
        return session


def close_camera_session(session_id: str) -> None:
    """Free the model and release the GPU claim. Safe to call twice."""
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return

    # Wait for a frame that is mid-inference; new frames will see analyzer=None and stop.
    with session["inference_lock"]:
        session["analyzer"] = None
        session["last_annotated_frame"] = None
        release_gpu_memory()                        # VRAM back BEFORE anyone else may claim the GPU
        if session.get("inference_claimed"):
            app_state.exit_inference("camera")
            session["inference_claimed"] = False
    logger.info(f"Released TubeAnalyzer resources for camera session {session_id}")


def evict_idle_camera_sessions(max_idle_seconds: int = _SESSION_IDLE_TIMEOUT_SEC) -> None:
    now = time.time()
    with _sessions_lock:
        idle = [sid for sid, s in _sessions.items() if now - s.get("last_active", now) > max_idle_seconds]
    for sid in idle:
        logger.info(f"Evicting idle camera session {sid}")
        close_camera_session(sid)


def get_camera_session_stats(session_id: str) -> Optional[Dict[str, Any]]:
    session = _sessions.get(session_id)
    return session.get("last_stats") if session else None


# ----------------------------------------------------------------- inference
def _error_stats(session_id: str, reason: str, frames_processed: int = 0, status: str = "error") -> Dict[str, Any]:
    return {"session_id": session_id, "status": status, "reasons": [reason], "frames_processed": frames_processed}


def analyze_camera_frame(
    frame,
    session_id: str,
    model_path: Optional[str] = None,
    video_name: Optional[str] = None,      # accepted for old callers, not used
    annotate: bool = False,
    **kwargs,
) -> Tuple[Dict[str, Any], Any]:
    """Run the model on one camera frame. Returns (stats, clean_frame).

    ``frame_time_ms`` covers ONLY TubeAnalyzer.analyze_frame(): detect + track + ROI +
    result. It excludes frame reading, waiting for the session lock, warm-up, drawing
    and encoding. ``fps`` is 1000 / frame_time_ms: how fast the model could run, not how
    fast the camera delivers frames.

    ``annotate=True`` also draws the boxes and stores the image in
    session["last_annotated_frame"]; it is off by default because drawing is wasted work
    when the frontend draws boxes itself from stats["detections"]."""
    if app_state.get_mode() == "TRAINING":
        return _error_stats(session_id, "Training is in progress -- camera inference paused", status="paused"), frame

    if frame is None or getattr(frame, "size", 0) == 0:
        return _error_stats(session_id, "Empty frame"), frame

    try:
        session = _get_or_create_camera_session(session_id, model_path)
    except Exception as e:
        logger.warning(f"Could not start camera session {session_id}: {e}")
        return _error_stats(session_id, str(e)), frame

    session["last_active"] = time.time()

    # One frame at a time per session: the analyzer's tracker state is not thread-safe.
    with session["inference_lock"]:
        analyzer = session["analyzer"]
        if analyzer is None:        # session was closed while this frame waited for the lock
            return _error_stats(session_id, "Camera session was closed", session["frames_processed"]), frame

        frame_number = session["frames_processed"] + 1
        try:
            with inference_context():
                analyzer.warm_up(frame.shape)          # no-op after the first frame of this size; not timed
                t_start = time.perf_counter()
                analysis = analyzer.analyze_frame(frame, frame_number)
                if cuda_is_available():
                    torch.cuda.synchronize()
                frame_time_ms = (time.perf_counter() - t_start) * 1000.0
            session["frames_processed"] = frame_number   # only count frames that succeeded
        except Exception as e:
            logger.error(f"Error in TubeAnalyzer for session {session_id}: {e}", exc_info=True)
            return _error_stats(session_id, str(e), session["frames_processed"]), frame

        clean_frame = analyzer.get_clean_frame(frame)
        if annotate:
            session["last_annotated_frame"] = analyzer.draw_annotations(frame, analysis)

    stats = {
        **build_frontend_stats(analysis.result),
        "session_id": session_id,
        "status": "processing",
        "frame": frame_number,
        "frames_processed": frame_number,
        "fps": round(1000.0 / frame_time_ms, 1) if frame_time_ms > 0 else 0.0,
        "frame_time_ms": round(frame_time_ms, 2),
        "latency_ms": round(frame_time_ms, 2),     # legacy key, same value as frame_time_ms
        "video_width": frame.shape[1],
        "video_height": frame.shape[0],
    }
    session["last_stats"] = stats
    return stats, clean_frame


# ------------------------------------------------- old names (same behaviour)
run_inference = analyze_camera_frame
get_session_status = get_camera_session_stats
clear_session = close_camera_session
cleanup_stale_sessions = evict_idle_camera_sessions
reset_session_for_next_video = close_camera_session


def set_camera_roi(session_id: Optional[str] = None, points: Any = None, frame_size: Optional[Tuple[int, int]] = None) -> None:
    """Hot-swap active camera session analyzer in-memory (per-session ROI, no disk persistence)."""
    with _sessions_lock:
        if session_id and session_id in _sessions:
            targets = [_sessions[session_id]]
        else:
            targets = list(_sessions.values())
        for s in targets:
            s["roi_points"] = points or []
            s["roi_frame_size"] = frame_size
            analyzer = s.get("analyzer")
            if analyzer is not None:
                analyzer.set_roi(points, frame_size)


def _set_camera_analyzer(analyzer):
    pass
