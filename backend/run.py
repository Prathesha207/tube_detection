import os
import shutil
import sys
import multiprocessing

multiprocessing.freeze_support()

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

if getattr(sys, "frozen", False):
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    torch_lib_dirs = (
        bundle_dir / "torch" / "lib",
        bundle_dir / "_internal" / "torch" / "lib",
        bundle_dir / "torch",
    )
    existing = [str(p) for p in torch_lib_dirs if p.is_dir()]
    if existing:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(existing + [os.environ.get("LD_LIBRARY_PATH", "")])

try:
    import torch
    if getattr(sys, "frozen", False):
        print(f"[BOOTSTRAP] PyTorch loaded: {torch.__version__}; CUDA={torch.cuda.is_available()}")
    torch.set_num_threads(1)
except Exception as exc:
    if getattr(sys, "frozen", False):
        import traceback
        print("[BOOTSTRAP] PyTorch import failed:", repr(exc), flush=True)
        traceback.print_exc()

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
    multiprocessing.freeze_support()
    setup_db()

    target_port = int(os.getenv("BACKEND_PORT", os.getenv("PORT", "8000")))

    # Pre-flight check: if Vision Monitor backend is already running and healthy, reuse it!
    import urllib.request
    import json

    def is_our_backend_already_running(p: int) -> bool:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{p}/health", headers={"User-Agent": "vision-monitor-launcher"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status in (200, 503):
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("app") == "vision-monitor" or data.get("status") in ("ready", "starting"):
                        return True
        except Exception:
            pass
        return False

    if is_our_backend_already_running(target_port):
        print(f"[STARTUP] Vision Monitor backend is already running on port {target_port}. Reusing active instance.")
        sys.exit(0)

    import socket
    def is_port_available(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', p))
                return True
            except OSError:
                return False

    port = target_port
    while not is_port_available(port) and port < target_port + 20:
        port += 1

    if port != target_port:
        print(f"[STARTUP] Target port {target_port} was occupied. Auto-shifted backend to port {port}.")
    else:
        print(f"[STARTUP] Starting backend on port {port}.")

    # Save active port marker for frontend/electron auto-detection
    try:
        port_file = DATA_DIR / "backend.port"
        port_file.write_text(str(port), encoding="utf-8")
    except Exception:
        pass
    try:
        (Path.cwd() / "backend.port").write_text(str(port), encoding="utf-8")
    except Exception:
        pass
    try:
        (Path(__file__).parent / "backend.port").write_text(str(port), encoding="utf-8")
    except Exception:
        pass

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )

