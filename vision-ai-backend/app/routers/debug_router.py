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
    and whether process_frame is ready.
    """
    info = {"import_succeeded": False}
    try:
        from app.ml.tube.inference import tube_analyzer
        file_path = getattr(tube_analyzer, "__file__", None)
        info.update({
            "import_succeeded": True,
            "file_path": file_path,
            "last_modified": os.path.getmtime(file_path) if file_path and os.path.exists(file_path) else None,
            "sha256_short": _file_hash(file_path) if file_path else None,
            "is_tube_analyzer_ready": hasattr(getattr(tube_analyzer, "TubeAnalyzer", object), "process_frame"),
        })
    except ImportError as e:
        info["error"] = str(e)

    return info