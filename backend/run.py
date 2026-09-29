
import os
import shutil
import sys
import uvicorn
from pathlib import Path


def user_data_dir() -> Path:
    if sys.platform == "win32":
        root = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        base = Path(root or Path.home())
        new_dir = base / "Vision-Monitor"
        old_dir = base / "Vision-AI"
        if not new_dir.exists() and old_dir.exists():
            try:
                import shutil
                shutil.copytree(old_dir, new_dir, dirs_exist_ok=True)
            except Exception:
                pass
        return new_dir

    xdg = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    new_dir = xdg / "vision-monitor"
    old_dir = xdg / "vision-ai"
    if not new_dir.exists() and old_dir.exists():
        try:
            import shutil
            shutil.copytree(old_dir, new_dir, dirs_exist_ok=True)
        except Exception:
            pass
    return new_dir


DATA_DIR = user_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATA_DIR / "vision_ai.db"

# Set this before importing app.main: app.core.database reads the setting
# during import. This keeps the database writable on Windows and Linux.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")

def resource_path(relative_path):
    """
    Get absolute path for PyInstaller
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Set TORCH_HOME to local bundled cache so offline client PCs don't attempt network downloads
_bundled_torch_hub = resource_path(os.path.join("app", "ml", "torch_hub"))
if os.path.exists(_bundled_torch_hub):
    os.environ["TORCH_HOME"] = _bundled_torch_hub

# -------------------------------------------------------------------------
# CRITICAL FIX FOR PYINSTALLER + PYTORCH DEADLOCKS
# -------------------------------------------------------------------------
# PyTorch/OpenMP often deadlocks when run inside a ThreadPoolExecutor on Windows 
# from a PyInstaller executable. We must force single-threading for the backend BLAS/OMP.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from app.main import app

def setup_db():
    dst_db = str(DATABASE_PATH)

    # If DB already exists, preserve it for existing user
    if os.path.exists(dst_db):
        print(f"[BOOTSTRAP] Existing user database loaded at {dst_db}")
        return

    # Fresh install on user's PC: create new clean database file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    open(dst_db, "a").close()
    print(f"[BOOTSTRAP] Fresh user database initialized at {dst_db}")

if __name__ == "__main__":

    setup_db()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )
