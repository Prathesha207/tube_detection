import sys
from pathlib import Path

def resource_path(relative_path: str):
    """
    Get absolute path to resource for dev and PyInstaller
    """

    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(".")

    return str(base_path / relative_path)