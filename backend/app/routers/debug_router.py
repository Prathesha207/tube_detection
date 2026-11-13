"""
debug_routes.py - add this router to your FastAPI app to answer, from a
running server, "which tube_analyzer is actually loaded right now?" --
without needing shell access to the box.

    from debug_routes import router as debug_router
    app.include_router(debug_router)

Then: GET /debug/ml-module-info
"""

import os
import hashlib
import importlib.metadata as md
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/debug", tags=["debug"])


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "unreadable"


def _wheel_distributions_for_module(module_file: str | None) -> tuple[dict | None, list[dict]]:
    accepted_names = {"head-tail-analyzer-v", "head-tail-analyzer"}
    loaded_path = Path(module_file).resolve() if module_file else None
    matches = []
    installed = []

    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name not in accepted_names:
            continue
        root = Path(dist.locate_file("")).resolve()
        item = {"name": dist.metadata.get("Name"), "version": dist.version, "location": str(root)}
        installed.append(item)
        if loaded_path is not None:
            try:
                loaded_path.relative_to(root)
                matches.append(item)
            except ValueError:
                pass

    # Prefer the installed distribution whose site-packages root contains the
    # module Python actually imported; don't report an unrelated duplicate.
    return (matches[0] if matches else None), installed


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
        wheel_module = getattr(tube_analyzer, "rvf", None)
        wheel_module_path = getattr(wheel_module, "__file__", None)
        loaded_info = getattr(tube_analyzer, "WHEEL_INFO", {})

        try:
            import head_tail_analyzer as _hta
            runtime_package_version = getattr(_hta, "__version__", None)
        except Exception:
            runtime_package_version = None

        loaded_distribution, installed_distributions = _wheel_distributions_for_module(wheel_module_path)
        distribution_version = loaded_distribution.get("version") if loaded_distribution else None
        wheel_version = runtime_package_version or distribution_version or "Not installed"
        current_module_hash = _file_hash(wheel_module_path) if wheel_module_path else None
        loaded_module_hash = loaded_info.get("module_sha256")
        disk_changed_since_import = bool(
            loaded_module_hash and current_module_hash
            and current_module_hash != "unreadable"
            and loaded_module_hash != current_module_hash
        )
        version_mismatch = bool(
            runtime_package_version and distribution_version
            and runtime_package_version != distribution_version
        )
        stale_loaded_module = disk_changed_since_import or version_mismatch
        required_wheel_functions = getattr(tube_analyzer, "_REQUIRED_WHEEL_FUNCS", ())
        missing_wheel_functions = [
            name for name in required_wheel_functions
            if wheel_module is None or not hasattr(wheel_module, name)
        ]

        if wheel_error:
            msg = f"ML code imported, but the core engine failed to load: {wheel_error}"
            ready = False
        elif wheel_module is None or missing_wheel_functions:
            msg = f"The loaded analyzer wheel is missing required API: {missing_wheel_functions}"
            ready = False
        elif stale_loaded_module:
            msg = "The wheel files or metadata changed after this process imported them; restart the backend to load the installed wheel."
            ready = False
        else:
            msg = f"ML module is loaded and ready. Runtime wheel version: {wheel_version}"
            ready = True

        info.update({
            "message": msg,
            "wheel_version": wheel_version,
            "runtime_package_version": runtime_package_version,
            "installed_distribution_version": distribution_version,
            "installed_distribution_location": loaded_distribution.get("location") if loaded_distribution else None,
            "installed_wheel_distributions": installed_distributions,
            "wheel_module_path": wheel_module_path,
            "wheel_module_sha256_loaded": loaded_module_hash,
            "wheel_module_sha256_on_disk": current_module_hash,
            "loaded_module_is_stale": stale_loaded_module,
            "restart_required": stale_loaded_module,
            "missing_wheel_functions": missing_wheel_functions,
            "import_succeeded": True,
            "file_path": file_path,
            "last_modified": os.path.getmtime(file_path) if file_path and os.path.exists(file_path) else None,
            "sha256_short": _file_hash(file_path)[:12] if file_path else None,
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
    from app.ml.tube_analyzer import MODEL_PATH, _MODEL_MTIMES
    from app.ml.camera_inference_service import _shared_camera_analyzer
    from app.ml.video_inference_service import video_inference_service

    model_path = MODEL_PATH
    exists = os.path.exists(model_path)
    disk_mtime = os.path.getmtime(model_path) if exists else None
    cached_mtime = _MODEL_MTIMES.get(str(Path(model_path).resolve()))
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
        "cached_model_mtime": cached_mtime,
        "needs_reload": needs_reload,
        "has_shared_camera_analyzer": bool(_shared_camera_analyzer is not None),
        "has_video_shared_analyzer": bool(getattr(video_inference_service, "_shared_analyzer", None) is not None),
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
