"""
app.ml package
Exports TubeAnalyzer, inference services, and ML state coordination.
"""

from .tube_analyzer import TubeAnalyzer, MODEL_PATH, ROI_JSON, CONFIG_PATH, load_roi
from .camera_inference_service import (
    analyze_camera_frame as run_camera_inference,
    close_camera_session as clear_camera_session,
    get_camera_session_stats as get_camera_session_status,
)
from .video_inference_service import (
    VideoInferenceService,
    video_inference_service,
    ml_inference_service,
)
from . import app_state

__all__ = [
    "TubeAnalyzer",
    "MODEL_PATH",
    "ROI_JSON",
    "CONFIG_PATH",
    "load_roi",
    "run_camera_inference",
    "clear_camera_session",
    "get_camera_session_status",
    "VideoInferenceService",
    "video_inference_service",
    "ml_inference_service",
    "app_state",
]
