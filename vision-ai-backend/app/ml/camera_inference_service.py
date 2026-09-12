"""
camera_inference_service.py
===========================
Handles live, real-time per-frame inference (camera feeds, OAK streams, RTSP).

When to use this file:
- A caller (e.g. `oak_camera_service.py`) captures one frame at a time from a camera
  and needs real-time duck detection, counting, and anomaly classification returned immediately.
- Manages fast in-memory sessions, per-frame tracking states, anchor counts, and idle session eviction.

When NOT to use this file:
- If you are processing a full uploaded/recorded video file (MP4/AVI) in the background,
  use `video_inference_service.py` instead!

Main functions:
- `run_inference(frame, session_id, expected_duck_count, video_name)`
- `get_session_status(session_id)`
- `update_expected_ducks(session_id, new_count)`
- `clear_session(session_id)`
- `reset_session_for_next_video(session_id)`
"""

import os
import sys
import time
import logging
import threading
import yaml
from typing import Dict, Any, Optional, Tuple

try:
    from duck_analyzer import DuckAnalyzer
except ImportError:
    try:
        from app.ml.duck_analyzer.analyzer import DuckAnalyzer
    except ImportError:
        DuckAnalyzer = None

from app.ml import app_state  # shared GPU-mode flag used by video_inference_service.py too

logger = logging.getLogger("camera-inference")

_sessions: Dict[str, Dict[str, Any]] = {}

_sessions_lock = threading.Lock()

# Config path: look in app/ml/config/config.yaml first (canonical),
# then fall back to app/ml/config.yaml (legacy) and PyInstaller bundle.
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_CANDIDATES = [
    os.path.join(_ML_DIR, "config", "config.yaml"),  # canonical
    os.path.join(_ML_DIR, "config.yaml"),             # legacy flat layout
]
if getattr(sys, "_MEIPASS", None):
    _CONFIG_CANDIDATES.insert(0, os.path.join(sys._MEIPASS, "app", "ml", "config", "config.yaml"))
    _CONFIG_CANDIDATES.insert(1, os.path.join(sys._MEIPASS, "app", "ml", "config.yaml"))
_CONFIG_PATH = next((c for c in _CONFIG_CANDIDATES if os.path.exists(c)), _CONFIG_CANDIDATES[0])

# Idle sessions get evicted after this many seconds without a frame --
_SESSION_IDLE_TIMEOUT_SEC = 900


def _get_or_create_session(session_id: str, expected_duck_count: int,
                            original_filename: Optional[str]) -> Dict[str, Any]:
    # Fast path: session already exists, no need to touch the lock at all.
    existing = _sessions.get(session_id)
    if existing is not None and "analyzer" in existing:
        return existing

    with _sessions_lock:
        if session_id in _sessions and "analyzer" in _sessions[session_id]:
            return _sessions[session_id]
        
        # If there is a shell session holding the expected count, grab it
        shell = _sessions.get(session_id)
        if shell and "expected_duck_count" in shell:
            expected_duck_count = shell["expected_duck_count"]

        if not os.path.isfile(_CONFIG_PATH):
            raise RuntimeError(
                f"config.yaml not found at {_CONFIG_PATH} -- camera inference needs "
                "the same real config the video pipeline uses, not a stub.")

        if not app_state.try_enter_inference("camera"):
            active_kind = app_state.get_active_inference_kind()
            if app_state.get_mode() == "TRAINING":
                raise RuntimeError(
                    "Training is in progress -- camera inference cannot start")
            raise RuntimeError(
                f"GPU is currently in use by {active_kind or 'another process'} "
                "-- try again shortly")

        try:
            with open(_CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}

            # Resolve model_path portably: app/ml/model/best.pt first (canonical),
            # then app/ml/models/best.pt (legacy), then the value from config.yaml.
            ml_dir = _ML_DIR
            candidates = [
                os.path.join(ml_dir, "model", "best.pt"),    # canonical
                os.path.join(ml_dir, "models", "best.pt"),   # legacy
            ]
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                candidates.insert(0, os.path.join(sys._MEIPASS, "app", "ml", "model", "best.pt"))
                candidates.insert(1, os.path.join(sys._MEIPASS, "app", "ml", "models", "best.pt"))

            configured_path = cfg.get("model_path")
            if configured_path:
                rel = os.path.join(ml_dir, configured_path)
                if os.path.exists(rel):
                    candidates.append(rel)
                if os.path.isabs(configured_path) and os.path.exists(configured_path):
                    candidates.append(configured_path)

            resolved_model = next((c for c in candidates if c and os.path.exists(c)), None)

            if not resolved_model:
                checked = "\n  ".join(c for c in candidates if c)
                raise FileNotFoundError(
                    "Could not find best.pt. Checked:\n  " + checked +
                    f"\n\nPlace your YOLO weights at: {os.path.join(ml_dir, 'model', 'best.pt')}"
                )

            logger.info(f"[MODEL] Using weights: {resolved_model}")
            cfg["model_path"] = resolved_model

            # Resolve roi_path relative to the config file's directory
            cfg_dir = os.path.dirname(os.path.abspath(_CONFIG_PATH))
            roi_raw = cfg.get("roi_path", "hand_roi.json")
            if roi_raw and not os.path.isabs(roi_raw):
                cfg["roi_path"] = os.path.join(cfg_dir, roi_raw)

            # Production mode: no disk writes from DuckAnalyzer
            cfg["save_local"] = False
            cfg["annotated_dir"] = None

            # Dynamic device selection
            try:
                import torch
                cfg["device"] = 0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
            except Exception:
                cfg["device"] = "cpu"

            # Temporary session config path with resolved model_path in a guaranteed WRITABLE directory
            try:
                from app.core.app_paths import get_ml_session_config_dir
                session_cfg_dir = str(get_ml_session_config_dir())
            except Exception:
                import tempfile
                session_cfg_dir = os.path.join(tempfile.gettempdir(), "vision_monitor_sessions")
            os.makedirs(session_cfg_dir, exist_ok=True)
            session_cfg_path = os.path.join(session_cfg_dir, f"camera_config_{session_id}.yaml")
            with open(session_cfg_path, "w") as f:
                yaml.dump(cfg, f)

            analyzer = DuckAnalyzer(session_cfg_path, expected_duck_count=expected_duck_count)
        except Exception:
            app_state.exit_inference("camera")
            raise

        session = {
            "analyzer": analyzer,
            "frames_processed": 0,
            "expected_duck_count": expected_duck_count,
            "original_filename": original_filename,
            "last_active": time.time(),
            "inference_claimed": True,  # so clear_session knows to release it
        }
        _sessions[session_id] = session
        return session


def cleanup_stale_sessions(max_idle_seconds: int = _SESSION_IDLE_TIMEOUT_SEC) -> None:
    """Call this periodically (e.g. from a FastAPI startup background task
    running every few minutes) -- without it, _sessions grows forever for
    any camera client that disconnects without calling clear_session()."""
    now = time.time()
    stale = [sid for sid, s in _sessions.items()
             if now - s.get("last_active", now) > max_idle_seconds]
    for sid in stale:
        logger.info(f"Evicting stale camera session {sid}")
        clear_session(sid)


def run_inference(frame, session_id: str, expected_duck_count: int = 18,
                   video_name: Optional[str] = None) -> Tuple[Dict[str, Any], Any]:
    """
    Returns (stats, raw_frame) as a TUPLE, not one dict -- stats is JSON-safe
    on its own (no numpy arrays inside it), raw_frame is yours to encode/send
    however your camera transport needs (matches what VideoInferenceService
    does: stats go one way, pixels go a separate way).
    """
    if DuckAnalyzer is None:
        logger.error("DuckAnalyzer not installed. Run 'pip install ml/duck_analyzer-1.0.0-py3-none-any.whl'")
        return {"session_id": session_id, "status": "error",
                "reasons": ["DuckAnalyzer not installed"], "done": True}, None

    # Same shared flag video_inference_service.py checks -- refuse to burn GPU on
    # camera frames while a training job owns it, instead of silently
    # fighting it for VRAM. Cheap per-frame check, no lock contention with
    # session creation since get_mode() only reads.
    if app_state.get_mode() == "TRAINING":
        return {"session_id": session_id, "status": "paused",
                "reasons": ["Training is in progress -- camera inference paused"],
                "frames_processed": 0}, frame

    try:

        session = _get_or_create_session(session_id, expected_duck_count, video_name)
    except RuntimeError as e:
        # Covers both "config.yaml missing" and "GPU claimed by video/training"
        logger.warning(f"Could not start camera session {session_id}: {e}")
        return {"session_id": session_id, "status": "error",
                "reasons": [str(e)], "frames_processed": 0}, frame

    session["last_active"] = time.time()
    analyzer = session["analyzer"]
    session["frames_processed"] += 1

    annotated_frame = frame.copy()
    try:
        result = analyzer.process_frame(annotated_frame)
    except Exception as e:
        logger.error(f"Error in DuckAnalyzer for session {session_id}: {e}", exc_info=True)
        return {"session_id": session_id, "status": "error",
                "reasons": [str(e)], "frames_processed": session["frames_processed"]}, frame

    if isinstance(result, dict) and result.get("annotated_frame") is not None:
        annotated_frame = result["annotated_frame"]

    # ── Pass through EVERYTHING DuckAnalyzer returns — no re-interpretation ──
    # DuckAnalyzer._finish() is the single source of truth for all these fields.
    # camera_inference_service is purely an orchestrator: feed one frame in,
    # collect the result dict, relay it to the caller unchanged.
    anchor_locked = bool(result.get("anchor_locked", getattr(analyzer, "anchor_locked", False)))
    missing_ids   = result.get("missing_ids", [])
    added_ids     = result.get("added_ids", [])
    other_ids     = result.get("other_ids", [])
    detected_others = result.get("other_count", len(other_ids))
    hand_detected = bool(result.get("hand_detected", False))
    reasons       = list(result.get("reasons", []))
    is_anomaly    = (result.get("status") == "ANOMALY")

    new_thumbnails = result.get("thumbnails", [])
    if "thumbnails" not in session:
        session["thumbnails"] = []
    if new_thumbnails:
        existing = {
            (str(t.get("id")), str(t.get("event")))
            for t in session["thumbnails"]
        }
        session["thumbnails"].extend(
            t for t in new_thumbnails
            if (str(t.get("id")), str(t.get("event"))) not in existing
        )

    stats = {
        "session_id":              session_id,
        "original_filename":       session.get("original_filename"),
        "status":                  result.get("status", "processing"),
        "frame":                   result.get("frame", session["frames_processed"]),
        "frames_processed":        session["frames_processed"],
        "fps":                     round(result.get("fps", 0), 1),
        "detected_duck_count":     result.get("detected_duck_count", 0),
        "expected_duck_count":     session["expected_duck_count"],
        "other_count":             detected_others,
        "detected_other_toy_count": detected_others,   # alias for older frontend code
        "anchor_locked":           anchor_locked,
        "hand_detected":           hand_detected,
        "missing_ids":             missing_ids,
        "missing_count":           result.get("missing_count", len(missing_ids)),
        "added_ids":               added_ids,
        "added_count":             result.get("added_count", len(added_ids)),
        "other_ids":               other_ids,
        "reasons":                 reasons,
        "detections":              result.get("detections", []),
        "thumbnails":              session["thumbnails"],
        "is_anomaly_frame":        is_anomaly,
        "video_width":             frame.shape[1],
        "video_height":            frame.shape[0],
    }

    session["last_stats"] = stats
    return stats, annotated_frame


def get_session_status(session_id: str) -> Optional[Dict[str, Any]]:
    session = _sessions.get(session_id)
    if not session:
        return None
    stats = session.get("last_stats")
    if stats and isinstance(stats, dict):
        import numpy as np
        return {
            k: v for k, v in stats.items()
            if not k.startswith("_") and not isinstance(v, np.ndarray)
        }
    return stats


def update_expected_ducks(session_id: str, count: int) -> None:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if not session:
            _sessions[session_id] = {
                "expected_duck_count": count,
                "last_active": time.time()
            }
            return
        session["expected_duck_count"] = count
        session["last_active"] = time.time()
        analyzer = session.get("analyzer")
        if analyzer:
            analyzer.set_expected_duck_count(count)


def clear_session(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
        
        if session:
            analyzer = session.get("analyzer")
            if analyzer and hasattr(analyzer, "close"):
                analyzer.close()
            # NEW: release the cross-kind GPU claim this session took at
            # creation time, so video-upload inference (or training) can start
            # once the last camera session is gone.
            if session.get("inference_claimed"):
                app_state.exit_inference("camera")
            
            # Preserve expected_duck_count for the next run (e.g. start_inference)
            if "expected_duck_count" in session:
                if session_id not in _sessions:
                    _sessions[session_id] = {
                        "expected_duck_count": session["expected_duck_count"],
                        "last_active": time.time()
                    }


def reset_session_for_next_video(session_id: str) -> None:
    clear_session(session_id)