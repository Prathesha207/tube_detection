"""
debug_routes.py - add this router to your FastAPI app to answer, from a
running server, "which tube_analyzer is actually loaded right now?" --
without needing shell access to the box.

    from debug_routes import router as debug_router
    app.include_router(debug_router)

Then: GET /debug/ml-module-info
"""

import os
import sys
import hashlib
import importlib.metadata as md

from fastapi import APIRouter

router = APIRouter(prefix="/debug", tags=["debug"])


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "unreadable"


@router.get("/ml-module-info")
def ml_module_info():
    """
    Reports info about the TubeAnalyzer module: file path, content hash,
    and whether process_frame is ready. Also reports the installed wheel version.
    """
    info = {"import_succeeded": False}
    try:
        from app.ml import tube_analyzer
        file_path = getattr(tube_analyzer, "__file__", None)
        
        wheel_error = getattr(tube_analyzer, "_WHEEL_ERROR", None)
        wheel_version = None
        
        try:
            wheel_version = md.version("head_tail_analyzer")
        except md.PackageNotFoundError:
            wheel_version = "Not installed"

        if wheel_error:
            msg = f"ML code imported, but the core engine failed to load: {wheel_error}"
            ready = False
        else:
            msg = f"ML Module is fully loaded, healthy, and ready for inference. Using wheel version: {wheel_version}"
            ready = hasattr(getattr(tube_analyzer, "TubeAnalyzer", object), "analyze_frame")

        info.update({
            "message": msg,
            "wheel_version": wheel_version,
            "import_succeeded": True,
            "file_path": file_path,
            "last_modified": os.path.getmtime(file_path) if file_path and os.path.exists(file_path) else None,
            "sha256_short": _file_hash(file_path) if file_path else None,
            "is_tube_analyzer_ready": ready,
        })
    except ImportError as e:
        info["error"] = str(e)
        info["message"] = f"CRITICAL: Failed to import the ML module entirely: {e}"

    return info


@router.get("/model-status")
def model_status():
    """
    Returns the current status of the model on disk vs in-memory:
    shows whether a new .pt model file has been placed on disk and if reload is pending.
    """
    from app.ml.tube_analyzer import MODEL_PATH, _GLOBAL_YOLO_MTIMES
    from app.ml.camera_inference_service import _shared_camera_analyzer
    from app.ml.video_inference_service import video_inference_service

    model_path = MODEL_PATH
    exists = os.path.exists(model_path)
    disk_mtime = os.path.getmtime(model_path) if exists else None
    cached_mtime = _GLOBAL_YOLO_MTIMES.get(model_path)
    file_size_bytes = os.path.getsize(model_path) if exists else 0

    shared_analyzer = _shared_camera_analyzer or getattr(video_inference_service, "_shared_analyzer", None)
    in_memory_mtime = getattr(shared_analyzer, "model_mtime", None)

    needs_reload = False
    if exists and in_memory_mtime is not None:
        needs_reload = (disk_mtime != in_memory_mtime)

    return {
        "model_path": model_path,
        "exists": exists,
        "size_mb": round(file_size_bytes / (1024 * 1024), 2) if exists else 0,
        "disk_mtime": disk_mtime,
        "in_memory_mtime": in_memory_mtime,
        "cached_yolo_mtime": cached_mtime,
        "needs_reload": needs_reload,
        "active_classes": getattr(shared_analyzer.model, "names", None) if shared_analyzer and hasattr(shared_analyzer, "model") else None,
        "device": getattr(getattr(shared_analyzer, "cfg", None), "device", None) if shared_analyzer else None,
    }


@router.post("/reload-model")
def reload_model():
    """
    Force an immediate hot-reload of the model weights from disk without restarting the server.
    """
    from app.ml.tube_analyzer import TubeAnalyzer, MODEL_PATH, CONFIG_PATH, default_device, release_gpu_memory
    from app.ml.camera_inference_service import _set_camera_analyzer
    from app.ml.video_inference_service import _set_shared_analyzer

    if not os.path.exists(MODEL_PATH):
        return {"status": "error", "message": f"Model file not found at {MODEL_PATH}"}

    try:
        # Purge existing GPU memory cache
        release_gpu_memory(clear_cache=True)

        new_analyzer = TubeAnalyzer(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            device=default_device(),
            draw_overlay=False,
        )

        # Warm up
        try:
            import numpy as np
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            new_analyzer.process_frame(dummy)
        except Exception:
            pass

        _set_camera_analyzer(new_analyzer)
        _set_shared_analyzer(new_analyzer)

        return {
            "status": "success",
            "message": f"Successfully hot-reloaded model from {MODEL_PATH}",
            "classes": getattr(new_analyzer.model, "names", None),
            "model_mtime": new_analyzer.model_mtime,
            "device": new_analyzer.cfg.device,
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to reload model: {e}"}