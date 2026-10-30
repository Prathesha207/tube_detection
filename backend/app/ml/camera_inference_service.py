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

from app.ml import app_state
from app.core.logger import setup_logger
from .tube_analyzer import (
    TubeAnalyzer,
    default_device,
    new_stats,
    release_gpu_memory,
)

logger = setup_logger("tube-camera-inference")

_ML_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(_ML_DIR / "model" / "best.pt")
_CONFIG_PATH = str(_ML_DIR / "config" / "config.yaml")

_SESSION_IDLE_TIMEOUT_SEC = 300     # no frame for this long -> session is closed, GPU released
_IDLE_SWEEP_INTERVAL_SEC = 30       # how often the background thread looks for idle sessions

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
_idle_sweeper: Optional[threading.Thread] = None
_shared_camera_analyzer: Optional[TubeAnalyzer] = None
_global_roi_points: list = []
_global_roi_frame_size: Optional[Tuple[int, int]] = None
_global_camera_calib: Optional[Dict[str, Any]] = None

_model_ready_event = threading.Event()

############### Setup & Lifecycle Functions ###############
def _set_camera_analyzer(analyzer: TubeAnalyzer) -> None:
    """Preloaded TubeAnalyzer from backend startup lifespan for instant zero-delay camera inference."""
    global _shared_camera_analyzer, _model_prewarmed, _global_camera_calib
    _shared_camera_analyzer = analyzer
    if _global_camera_calib and hasattr(analyzer, "set_camera_matrix"):
        analyzer.set_camera_matrix(
            _global_camera_calib["matrix"],
            _global_camera_calib["width"],
            _global_camera_calib["height"],
            _global_camera_calib.get("dist_coeffs"),
        )
    _model_prewarmed = True
    _model_ready_event.set()
    logger.info("Preloaded TubeAnalyzer registered into camera_inference_service — ready for instant inference")


_prewarm_lock = threading.Lock()
_model_prewarmed: bool = False


def prewarm_camera_model() -> None:
    """Pre-warm TubeAnalyzer on GPU in the background when camera connects or stream starts."""
    global _shared_camera_analyzer, _model_prewarmed
    if _model_prewarmed and _shared_camera_analyzer is not None:
        return
    with _prewarm_lock:
        if _model_prewarmed and _shared_camera_analyzer is not None:
            return
        try:
            if _shared_camera_analyzer is None:
                try:
                    from app.ml.video_inference_service import video_inference_service
                    if video_inference_service._shared_analyzer is not None:
                        _shared_camera_analyzer = video_inference_service._shared_analyzer
                        _model_prewarmed = True
                        _model_ready_event.set()
                except Exception:
                    pass

            if _shared_camera_analyzer is None:
                if _model_ready_event.wait(timeout=0.2) and _shared_camera_analyzer is not None:
                    _model_prewarmed = True
                    return

            if _shared_camera_analyzer is None:
                logger.info("[PREWARM] Preloading TubeAnalyzer for camera in background...")
                _shared_camera_analyzer = TubeAnalyzer(
                    model_path=_DEFAULT_MODEL_PATH,
                    config_path=_CONFIG_PATH,
                    device=default_device(),
                    draw_overlay=False,
                )
            # Pre-warm directly with full 1080p frame so CUDA kernels and cuDNN graphs compile
            _shared_camera_analyzer.warm_up((1080, 1920))
            _model_prewarmed = True
            _model_ready_event.set()
            logger.info("[PREWARM] TubeAnalyzer fully pre-warmed on GPU — ready for instant 0.0s camera inference.")
        except Exception as e:
            logger.warning(f"[PREWARM] Background prewarm skipped or failed: {e}")


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
    if existing is not None:
        if existing.get("analyzer") is not None or existing.get("load_error") is not None:
            return existing

    with _sessions_lock:
        existing = _sessions.get(session_id)
        if existing is not None:
            if existing.get("analyzer") is not None or existing.get("load_error") is not None:
                return existing

        if not app_state.try_enter_inference("camera"):
            if app_state.get_mode() == "TRAINING":
                raise RuntimeError("Training is in progress -- camera inference cannot start")
            active_kind = app_state.get_active_inference_kind()
            raise RuntimeError(f"GPU is currently in use by {active_kind or 'another process'} -- try again shortly")

        # Fast path: Ensure preloaded shared analyzer is ready
        global _shared_camera_analyzer, _model_prewarmed
        if _shared_camera_analyzer is None:
            try:
                from app.ml.video_inference_service import video_inference_service
                if video_inference_service._shared_analyzer is not None:
                    _shared_camera_analyzer = video_inference_service._shared_analyzer
                    _model_prewarmed = True
                    _model_ready_event.set()
            except Exception:
                pass

        if _shared_camera_analyzer is None:
            # Quick check (max 0.2s) in case startup thread is just about to finish setting the analyzer
            if _model_ready_event.wait(timeout=0.2) and _shared_camera_analyzer is not None:
                _model_prewarmed = True

        if _shared_camera_analyzer is None:
            # Synchronously load the model immediately so the session starts in ~1.5s instead of hanging for 10-12s
            logger.info("Shared analyzer not preloaded — loading TubeAnalyzer synchronously for instant camera start")
            try:
                _shared_camera_analyzer = TubeAnalyzer(
                    model_path=model_path or _DEFAULT_MODEL_PATH,
                    config_path=_CONFIG_PATH,
                    device=default_device(),
                    draw_overlay=False,
                )
                _model_prewarmed = True
                _model_ready_event.set()
                try:
                    from app.ml.video_inference_service import _set_shared_analyzer
                    _set_shared_analyzer(_shared_camera_analyzer)
                except Exception:
                    pass
            except Exception as load_err:
                logger.error(f"Failed to load camera analyzer synchronously: {load_err}", exc_info=True)
                if app_state.get_active_inference_kind() == "camera":
                    app_state.exit_inference("camera")
                raise RuntimeError(f"Failed to initialize AI model for camera: {load_err}")

        if _shared_camera_analyzer is not None:
            if getattr(_shared_camera_analyzer, "has_weights_changed", lambda: False)():
                logger.info(f"[HOT-RELOAD] Detected updated weights at {_shared_camera_analyzer.model_path}. Hot-reloading TubeAnalyzer...")
                try:
                    new_analyzer = TubeAnalyzer(
                        model_path=_shared_camera_analyzer.model_path,
                        config_path=_CONFIG_PATH,
                        device=default_device(),
                        draw_overlay=False,
                    )
                    _shared_camera_analyzer = new_analyzer
                    try:
                        from app.ml.video_inference_service import _set_shared_analyzer
                        _set_shared_analyzer(new_analyzer)
                    except Exception:
                        pass
                    logger.info("[HOT-RELOAD] TubeAnalyzer successfully hot-reloaded with new model weights.")
                except Exception as reload_err:
                    logger.error(f"[HOT-RELOAD] Failed to hot-reload updated model: {reload_err}", exc_info=True)

        _shared_camera_analyzer.reset_tracking()
        if _global_camera_calib and hasattr(_shared_camera_analyzer, "set_camera_matrix"):
            _shared_camera_analyzer.set_camera_matrix(
                _global_camera_calib["matrix"],
                _global_camera_calib["width"],
                _global_camera_calib["height"],
                _global_camera_calib.get("dist_coeffs"),
            )
        if _global_roi_points:
            try:
                _shared_camera_analyzer.set_roi(_global_roi_points, _global_roi_frame_size)
            except Exception as e:
                logger.warning(f"Stored ROI invalid: {e}")

        session = {
            "analyzer": _shared_camera_analyzer,
            "frames_processed": 0,
            "last_active": time.time(),
            "inference_claimed": True,
            "inference_lock": threading.Lock(),
            "last_stats": new_stats(session_id=session_id, status="running"),
            "last_annotated_frame": None,
            "roi_points": list(_global_roi_points),
            "roi_frame_size": _global_roi_frame_size,
            "load_error": None,
        }
        _sessions[session_id] = session
        _start_idle_sweeper_once()
        logger.info(f"Reusing TubeAnalyzer for camera session {session_id} — instantaneous GPU start")
        return session


def close_camera_session(session_id: str) -> None:
    """Free the model and release the GPU claim. Safe to call twice."""
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return

    # Wait for a frame that is mid-inference; new frames will see analyzer=None and stop.
    with session["inference_lock"]:
        analyzer = session.get("analyzer")
        if analyzer is not None and analyzer is not _shared_camera_analyzer:
            release_gpu_memory()
        elif analyzer is _shared_camera_analyzer:
            analyzer.reset_tracking()
        session["analyzer"] = None
        session["last_annotated_frame"] = None
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

############### Inference & Helper Functions ###############
# ----------------------------------------------------------------- inference
def _error_stats(session_id: str, reason: str, frames_processed: int = 0, status: str = "error") -> Dict[str, Any]:
    return {"session_id": session_id, "status": status, "reasons": [reason], "frames_processed": frames_processed}


def analyze_camera_frame(
    frame,
    session_id: str,
    depth_frame: Optional[Any] = None,
    model_path: Optional[str] = None,
    video_name: Optional[str] = None,      # accepted for old callers, not used
    annotate: bool = False,
    wait_for_model: bool = False,
    frame_seq: Optional[int] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Any]:
    """Run the model on one camera frame (with optional aligned depth frame). Returns (stats, clean_frame).

    ``fps`` is the wheel's own processing speed (``result["fps"]``, the number the CLI shows).
    ``frame_time_ms`` / ``model_fps`` are measured around TubeAnalyzer.analyze_frame() only
    (detect + track + ROI + result), excluding frame reading, waiting for the session lock,
    warm-up and encoding.

    ``depth_frame`` provides uint16 depth in mm aligned to CAM_A (RGB), enabling 3D positions
    (xyz_m, z_m, z_mm) and true physical diameter (width_mm_depth) without needing a calibration ruler.

    ``annotate=True`` makes the wheel draw the boxes inside the same call and stores the image in
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

    if wait_for_model:
        while True:
            with _sessions_lock:
                s = _sessions.get(session_id)
            if s is None or s.get("analyzer") != "LOADING":
                break
            time.sleep(0.1)

    # One frame at a time per session: the analyzer's tracker state is not thread-safe.
    with session["inference_lock"]:
        analyzer = session["analyzer"]
        if analyzer == "LOADING":
            stats = new_stats(session_id=session_id, status="LOADING_MODEL")
            session["last_stats"] = stats
            return stats, frame
        if session.get("load_error") or analyzer == "ERROR":
            return _error_stats(session_id, session.get("load_error") or "Model failed to load", session["frames_processed"]), frame
        if analyzer is None:        # session was closed while this frame waited for the lock
            return _error_stats(session_id, "Camera session was closed", session["frames_processed"]), frame

        frame_number = frame_seq if frame_seq is not None else (session["frames_processed"] + 1)
        try:
            stats = analyzer.infer(frame, frame_number, annotate=annotate, depth=depth_frame)      # <- the ONE ML call
            session["frames_processed"] += 1
        except Exception as e:
            logger.error(f"Error in TubeAnalyzer for session {session_id}: {e}", exc_info=True)
            return _error_stats(session_id, str(e), session["frames_processed"]), frame

        clean_frame = analyzer.get_clean_frame(frame)
        if annotate:
            session["last_annotated_frame"] = analyzer.draw_annotations(frame)   # the wheel's picture of THIS frame

    stats.update(session_id=session_id, frames_processed=session["frames_processed"])
    session["last_stats"] = stats
    return stats, clean_frame


def set_camera_roi(session_id: Optional[str] = None, points: Any = None, frame_size: Optional[Tuple[int, int]] = None) -> None:
    """Hot-swap active camera session analyzer in-memory (per-session ROI, no disk persistence)."""
    global _global_roi_points, _global_roi_frame_size
    if points and len(points) >= 3:
        try:
            from app.ml.tube_analyzer import roi_from_points
            _ = roi_from_points(points, frame_size)
        except Exception as val_err:
            logger.warning(f"Invalid ROI geometry in set_camera_roi: {val_err}")
            raise

    with _sessions_lock:
        # Save globally so new sessions inherit it
        _global_roi_points = points or []
        _global_roi_frame_size = frame_size
        
        if session_id and session_id in _sessions:
            targets = [_sessions[session_id]]
        elif session_id in ("camera", "live") and "live" in _sessions:
            targets = [_sessions["live"]]
        else:
            targets = list(_sessions.values())
        for s in targets:
            s["roi_points"] = list(_global_roi_points)
            s["roi_frame_size"] = frame_size
            inference_lock = s.get("inference_lock")
            if inference_lock:
                with inference_lock:
                    analyzer = s.get("analyzer")
                    if isinstance(analyzer, TubeAnalyzer):
                        analyzer.set_roi(points, frame_size)
            else:
                analyzer = s.get("analyzer")
                if isinstance(analyzer, TubeAnalyzer):
                    analyzer.set_roi(points, frame_size)


def open_offline_depth_reader(video_path: str):
    """Optional paired depth video reader (<name>_depth.mkv) for offline videos."""
    if not video_path:
        return None
    try:
        from app.ml.tube_analyzer import rvf
        if rvf is not None and hasattr(rvf, "depth3d"):
            d3d = rvf.depth3d
            depth_pair = d3d.find_depth_pair(video_path)
            if depth_pair and Path(depth_pair).exists():
                return d3d.DepthReader(depth_pair)
    except Exception as e:
        logger.debug(f"Could not open depth pair for {video_path}: {e}")
    return None


def update_camera_calibration(camera_matrix: Any, calib_width: int, calib_height: int, dist_coeffs: Optional[Any] = None) -> None:
    """Update active camera intrinsics from camera hardware calibration."""
    global _shared_camera_analyzer, _global_camera_calib
    _global_camera_calib = {
        "matrix": camera_matrix,
        "width": calib_width,
        "height": calib_height,
        "dist_coeffs": dist_coeffs,
    }
    if _shared_camera_analyzer is not None and hasattr(_shared_camera_analyzer, "set_camera_matrix"):
        _shared_camera_analyzer.set_camera_matrix(camera_matrix, calib_width, calib_height, dist_coeffs)
    with _sessions_lock:
        for session in _sessions.values():
            analyzer = session.get("analyzer")
            if analyzer and hasattr(analyzer, "set_camera_matrix"):
                analyzer.set_camera_matrix(camera_matrix, calib_width, calib_height, dist_coeffs)


# Backward compatibility aliases
clear_session = close_camera_session
get_session_status = get_camera_session_stats
cleanup_stale_sessions = evict_idle_camera_sessions
run_inference = analyze_camera_frame


