"""Shared paths for the backend's active ML config and model."""

from pathlib import Path

import yaml


ML_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ML_DIR / "config" / "config.yaml"


def _model_path_from_config() -> str:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"ML config not found: {CONFIG_PATH}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    configured_path = config.get("model_path")
    if not configured_path:
        raise ValueError(f"model_path is not set in {CONFIG_PATH}")

    model_path = Path(str(configured_path)).expanduser()
    if not model_path.is_absolute():
        model_path = CONFIG_PATH.parent / model_path
    return str(model_path.resolve())


# The YAML is the single source of truth. Restart the backend after changing it.
MODEL_PATH = _model_path_from_config()
