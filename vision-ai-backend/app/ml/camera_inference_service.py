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
    from app.ml.debug.duck_analyzer import DuckAnalyzer
except ImportError:
    DuckAnalyzer = None

from app.ml import app_state  # shared GPU mutual-exclusion flag (video vs camera vs training)
try:
    import torch
except ImportError:
    torch = None

from app.core.logger import setup_logger

logger = setup_logger("camera-inference")


import time

_cuda_operational_cached = None

def is_cuda_operational() -> bool:
    """Verifies that PyTorch can actually execute CUDA kernels on GPU 0,
    rather than just checking if the driver library is queryable."""
    global _cuda_operational_cached
    if _cuda_operational_cached is not None:
        return _cuda_operational_cached
    if torch is None:
        return False

    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        return False

    for attempt in range(1, 4):
        try:
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            torch.cuda.set_device(0)
            
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

            t = torch.zeros((1, 1), device="cuda:0")
            _ = t + 1.0
            del t
            torch.cuda.synchronize(0)
            _cuda_operational_cached = True
            return True
        except Exception as e:
            logger.warning(f"CUDA initialization attempt {attempt}/3 encountered: {e}. Retrying...")
            time.sleep(0.3)
            
    logger.error("CUDA is present but kernel execution failed after 3 attempts; falling back to CPU.")
    _cuda_operational_cached = False
    return False

_sessions: Dict[str, Dict[str, Any]] = {}

_sessions_lock = threading.Lock()

# Shared DuckAnalyzer instance: loaded once at app startup (by main.py lifespan)
# and reused for every camera session. Each session resets the analyzer state
# (anchor, trackers, warmup) but keeps the YOLO weights in GPU memory.
# If startup preload hasn't run yet (dev mode, or preload failed), falls back
# to per-session construction (original behavior).
_shared_camera_analyzer = None
_shared_analyzer_lock = threading.Lock()


def _set_camera_analyzer(analyzer) -> None:
    """Called by main.py lifespan after startup preload: injects the already-loaded
    DuckAnalyzer so camera sessions reuse it instead of loading weights from scratch."""
    global _shared_camera_analyzer
    with _shared_analyzer_lock:
        _shared_camera_analyzer = analyzer
    logger.info("[STARTUP] Pre-loaded DuckAnalyzer injected into camera_inference_service.")


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

        global _shared_camera_analyzer
        try:
            # ── Try to reuse the shared preloaded analyzer (zero model reload) ──
            with _shared_analyzer_lock:
                preloaded = _shared_camera_analyzer

            if preloaded is None:
                try:
                    from app.ml.video_inference_service import video_inference_service
                    if video_inference_service._shared_analyzer is not None:
                        preloaded = video_inference_service._shared_analyzer
                        with _shared_analyzer_lock:
                            _shared_camera_analyzer = preloaded
                        logger.info("Reusing DuckAnalyzer loaded by video service for camera session (instant start, no reload)...")
                except Exception:
                    pass

            if preloaded is not None:
                logger.info(
                    "Reusing pre-loaded DuckAnalyzer for camera session "
                    f"{session_id} (no model reload).")
                analyzer = preloaded
                
                # GPU re-assertion: If device was downgraded to CPU but GPU is working, restore it!
                if str(getattr(analyzer, "_device_str", "")).lower() == "cpu" and is_cuda_operational():
                    logger.info("GPU is operational — restoring shared analyzer to CUDA from CPU.")
                    analyzer._device_str = "cuda:0"
                    analyzer.use_half = bool(analyzer.cfg.get("use_half", False))
                    if hasattr(analyzer, "model") and analyzer.model is not None:
                        analyzer.model.to("cuda:0")
                    if hasattr(analyzer, "embedder") and analyzer.embedder is not None:
                        try:
                            analyzer.embedder.device = "cuda:0"
                            if hasattr(analyzer.embedder, "model") and analyzer.embedder.model is not None:
                                analyzer.embedder.model.to("cuda:0")
                            if hasattr(analyzer.embedder, "_mean"):
                                analyzer.embedder._mean = analyzer.embedder._mean.to("cuda:0")
                            if hasattr(analyzer.embedder, "_std"):
                                analyzer.embedder._std = analyzer.embedder._std.to("cuda:0")
                        except Exception:
                            pass
                if hasattr(analyzer, "_gpu_fail_count"):
                    analyzer._gpu_fail_count = 0

                # Reset all per-session state so this session starts clean
                analyzer.expected = int(expected_duck_count)
                analyzer.frame_idx = 0
                analyzer.anchor_locked = False
                analyzer.warmup_best = None
                analyzer.warmup_count = 0
                analyzer._last_time = None
                analyzer._hand_hold = 0
                analyzer._excess_sent = False
                for attr in ("tid_to_display", "display_info", "otid_to_display",
                              "other_info", "prov_new", "reclaim_candidates",
                              "_prov_other"):
                    obj = getattr(analyzer, attr, None)
                    if isinstance(obj, dict):
                        obj.clear()
                for attr in ("confirmed_sent", "missing_active", "other_sent"):
                    obj = getattr(analyzer, attr, None)
                    if isinstance(obj, set):
                        obj.clear()
                for attr in ("count_history",):
                    obj = getattr(analyzer, attr, None)
                    if hasattr(obj, "clear"):
                        obj.clear()
                analyzer.next_display_id = 1
                analyzer.num_anchor = 0
                analyzer.next_other_display = 1
                analyzer.num_other_anchor = 0
                try:
                    if hasattr(analyzer.model, "predictor"):
                        analyzer.model.predictor = None
                except Exception:
                    pass
                # Re-init MediaPipe if it was closed or graph became inactive
                if hasattr(analyzer, "hand_backend") and analyzer.hand_backend == "mediapipe":
                    need_reinit = False
                    if getattr(analyzer, "_mp_hands", None) is None:
                        need_reinit = True
                    elif getattr(analyzer._mp_hands, "_graph", None) is None:
                        need_reinit = True
                    if need_reinit:
                        try:
                            import mediapipe as mp
                            analyzer._mp_hands = mp.solutions.hands.Hands(
                                static_image_mode=False,
                                max_num_hands=2,
                                min_detection_confidence=analyzer.hand_conf,
                                min_tracking_confidence=analyzer.hand_track_conf,
                            )
                            logger.info("Re-initialized MediaPipe hands graph successfully.")
                        except Exception as mp_err:
                            logger.warning(f"Could not re-init MediaPipe: {mp_err}")
            else:
                # ── Fallback: build a fresh analyzer (dev mode or preload failed) ──
                logger.info(
                    f"No pre-loaded analyzer available — creating DuckAnalyzer for "
                    f"camera session {session_id} (first-time model load).")

                with open(_CONFIG_PATH, "r") as f:
                    cfg = yaml.safe_load(f) or {}

                # Resolve model_path portably
                ml_dir = _ML_DIR
                candidates = [
                    os.path.join(ml_dir, "model", "best.pt"),
                    os.path.join(ml_dir, "debug", "best.pt"),
                    os.path.join(ml_dir, "models", "best.pt"),
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

                # Resolve roi_path portably without triggering uvicorn WatchFiles reload
                cfg_dir = os.path.dirname(os.path.abspath(_CONFIG_PATH))
                roi_raw = cfg.get("roi_path", "model/hand_roi.json")
                roi_candidates = [
                    os.path.join(ml_dir, roi_raw),
                    os.path.join(ml_dir, "model", "hand_roi.json"),
                    os.path.join(cfg_dir, roi_raw),
                    os.path.join(cfg_dir, "hand_roi.json"),
                    os.path.join(ml_dir, "hand_roi.json"),
                ]
                resolved_roi = next((r for r in roi_candidates if os.path.exists(r)), None)
                import tempfile
                runtime_roi_path = os.path.join(tempfile.gettempdir(), "vision_hand_roi.json")
                if resolved_roi and not os.path.exists(runtime_roi_path):
                    try:
                        import shutil
                        shutil.copy2(resolved_roi, runtime_roi_path)
                    except Exception:
                        pass
                cfg["roi_path"] = runtime_roi_path if os.path.exists(runtime_roi_path) else (resolved_roi or runtime_roi_path)

                cfg["save_local"] = False
                cfg["annotated_dir"] = None
                cfg["device"] = 0 if is_cuda_operational() else "cpu"

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
                with _shared_analyzer_lock:
                    if _shared_camera_analyzer is None:
                        _shared_camera_analyzer = analyzer
                try:
                    from app.ml.video_inference_service import video_inference_service
                    if video_inference_service._shared_analyzer is None:
                        video_inference_service._shared_analyzer = analyzer
                except Exception:
                    pass
                try:
                    if os.path.exists(session_cfg_path):
                        os.remove(session_cfg_path)
                except Exception:
                    pass

        except Exception:
            app_state.exit_inference("camera")
            raise

        session = {
            "analyzer": analyzer,
            "frames_processed": 0,
            "expected_duck_count": expected_duck_count,
            "original_filename": original_filename,
            "last_active": time.time(),
            "inference_claimed": True,
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
    t_infer_start = time.perf_counter()
    try:
        if torch is not None:
            with torch.inference_mode():
                result = analyzer.process_frame(annotated_frame)
        else:
            result = analyzer.process_frame(annotated_frame)
        if hasattr(analyzer, "_gpu_fail_count"):
            analyzer._gpu_fail_count = 0
    except Exception as e:
        recovered = False
        if not hasattr(analyzer, "_gpu_fail_count"):
            analyzer._gpu_fail_count = 0
            
        if hasattr(analyzer, "_device_str") and str(analyzer._device_str).lower() != "cpu":
            analyzer._gpu_fail_count += 1
            logger.warning(
                f"Camera session {session_id}: GPU frame error #{analyzer._gpu_fail_count} ({e}); "
                "retrying this frame on CPU."
            )
            
            # If we hit 5 consecutive GPU failures, downgrade permanently
            if analyzer._gpu_fail_count >= 5:
                logger.error(f"Camera session {session_id}: GPU failed 5 times — downgrading session to CPU permanently.")
                analyzer._device_str = "cpu"
                analyzer.use_half = False
                try:
                    if hasattr(analyzer, "model") and analyzer.model is not None:
                        analyzer.model.to("cpu")
                    if hasattr(analyzer, "embedder") and analyzer.embedder is not None:
                        try:
                            analyzer.embedder.device = "cpu"
                            if hasattr(analyzer.embedder, "model") and analyzer.embedder.model is not None:
                                analyzer.embedder.model.to("cpu")
                            if hasattr(analyzer.embedder, "_mean"):
                                analyzer.embedder._mean = analyzer.embedder._mean.to("cpu")
                            if hasattr(analyzer.embedder, "_std"):
                                analyzer.embedder._std = analyzer.embedder._std.to("cpu")
                        except Exception:
                            pass
                with torch.inference_mode():
                    result = analyzer.process_frame(annotated_frame)
                recovered = True
                logger.info(f"Camera session {session_id}: successfully recovered on CPU.")
            except Exception as cpu_err:
                logger.error(f"Camera session {session_id}: CPU fallback retry also failed: {cpu_err}")

        if not recovered:
            logger.error(f"Error in DuckAnalyzer for session {session_id}: {e}", exc_info=True)
            return {"session_id": session_id, "status": "error",
                    "reasons": [str(e)], "frames_processed": session["frames_processed"]}, frame
    infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000
    infer_fps = round(1000.0 / infer_latency_ms, 1) if infer_latency_ms > 0 else 0.0

    if isinstance(result, dict) and result.get("annotated_frame") is not None:
        annotated_frame = result["annotated_frame"]

    # ── Pass through EVERYTHING DuckAnalyzer returns — no re-interpretation ──
    # DuckAnalyzer._finish() is the single source of truth for all these fields.
    # camera_inference_service is purely an orchestrator: feed one frame in,
    # collect the result dict, relay it to the caller unchanged.
    anchor_locked = bool(result.get("anchor_locked", getattr(analyzer, "anchor_locked", False)))
    missing_ids   = result.get("missing_ids", [])
    added_ids     = result.get("added_ids", [])
    excess_ids    = result.get("excess_ids", [])
    excess_count  = result.get("excess_count", len(excess_ids))
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

    raw_detected = result.get("detected_duck_count", 0)
    if hand_detected or result.get("status") == "HAND":
        detected_ducks = session.get("last_detected_count") or session["expected_duck_count"]
    else:
        detected_ducks = raw_detected
        session["last_detected_count"] = raw_detected

    stats = {
        "session_id":              session_id,
        "original_filename":       session.get("original_filename"),
        "status":                  result.get("status", "processing"),
        "frame":                   result.get("frame", session["frames_processed"]),
        "frames_processed":        session["frames_processed"],
        "fps":                     infer_fps if infer_fps > 0 else round(result.get("fps", 0), 1),
        "latency_ms":              round(infer_latency_ms, 1),
        "detected_duck_count":     detected_ducks,
        "expected_duck_count":     session["expected_duck_count"],
        "other_count":             detected_others,
        "detected_other_toy_count": detected_others,   # alias for older frontend code
        "anchor_locked":           anchor_locked,
        "hand_detected":           hand_detected,
        "missing_ids":             missing_ids,
        "missing_count":           result.get("missing_count", len(missing_ids)),
        "added_ids":               added_ids,
        "added_count":             result.get("added_count", len(added_ids)),
        "excess_ids":              excess_ids,
        "excess_count":            excess_count,
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
            if analyzer is not None:
                with _shared_analyzer_lock:
                    global _shared_camera_analyzer
                    if _shared_camera_analyzer is None:
                        _shared_camera_analyzer = analyzer
            # Release cross-kind GPU claim so video or other sessions can start
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