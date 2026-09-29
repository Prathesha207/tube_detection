"""
app/app_paths.py

Central place to resolve file paths that work both:
  - during development  (running: uvicorn app.main:app)
  - inside the .exe     (running: dist/backend.exe)

Usage anywhere in your app:
    from app.app_paths import BASE_DIR, get_path

    DATABASE_URL = f"sqlite:///{get_path('storage/database.db')}"
    MODEL_PATH   = get_path('models/my_model.pt')
"""

import os
import sys


def get_base_dir() -> str:
    """
    Returns the directory that should be treated as the project root.

    - When frozen (inside .exe): folder containing the .exe file
    - When running normally:     folder containing this file's package root
    """
    if getattr(sys, "frozen", False):
        # PyInstaller sets sys.executable to the .exe path
        return os.path.dirname(sys.executable)
    else:
        # Go up from app/app_paths.py → project root
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = get_base_dir()


def get_path(*relative_parts: str) -> str:
    """
    Build an absolute path relative to BASE_DIR.

    Examples:
        get_path("storage", "database.db")   → /project/storage/database.db
        get_path("models", "yolo.pt")        → /project/models/yolo.pt
    """
    path = os.path.join(BASE_DIR, *relative_parts)
    # Auto-create parent directories if they don't exist
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    return path


# ─── Pre-built common paths ───────────────────────────────────────────────────

STORAGE_DIR  = get_path("storage")
MODELS_DIR   = get_path("models")
DATABASE_URL = f"sqlite:///{get_path('storage', 'database.db')}"

# Uncomment and adjust if you use recordings or uploads:
# RECORDINGS_DIR = get_path("recordings_ffv1")
# LOGS_DIR       = get_path("logs")
