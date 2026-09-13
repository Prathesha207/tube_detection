from pathlib import Path
import os
import sys


def _user_data_dir() -> Path:
    """Return a writable per-user data directory on every supported OS."""
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

    # XDG is standard on Ubuntu and other Linux desktop distributions.
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


def get_desktop_dir() -> Path:
    """Resolve Desktop robustly on ANY PC, handling Windows Registry, OneDrive redirection, Linux, and custom user profiles."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
            )
            val, _ = winreg.QueryValueEx(key, "Desktop")
            winreg.CloseKey(key)
            expanded = os.path.expandvars(val)
            p = Path(expanded)
            if p.exists() and p.is_dir():
                return p
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass

    home = Path.home()
    # Check OneDrive Desktop or Standard Desktop
    for candidate in [home / "OneDrive" / "Desktop", home / "Desktop"]:
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Linux standard XDG desktop check
    xdg_desktop = os.getenv("XDG_DESKTOP_DIR")
    if xdg_desktop:
        try:
            p = Path(xdg_desktop)
            if p.exists() and p.is_dir():
                return p
        except Exception:
            pass

    # Ensure fallback Desktop exists on any PC
    try:
        fallback = home / "Desktop"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception:
        fallback = APP_DIR / "Desktop"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


APP_DIR = _user_data_dir()
APP_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = APP_DIR / "vision_ai.db"


from datetime import datetime


def get_ml_output_dir() -> Path:
    """Return the output directory for ML inference sessions (videos, thumbnails, results.json) directly on Desktop."""
    desktop = get_desktop_dir()
    today_str = datetime.now().strftime("%Y-%m-%d")
    out = desktop / "inference_results" / today_str
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_ml_session_config_dir() -> Path:
    """Return a guaranteed WRITABLE directory for temporary camera and video YAML configs."""
    cfg_dir = APP_DIR / "sessions"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


def get_logs_dir() -> Path:
    """Return logs directory: project root 'logs' during development, APP_DIR 'logs' in production/frozen."""
    if getattr(sys, "frozen", False):
        log_dir = APP_DIR / "logs"
    else:
        dev_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        try:
            dev_dir.mkdir(parents=True, exist_ok=True)
            return dev_dir
        except Exception:
            log_dir = APP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

