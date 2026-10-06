# app/schemas/camera_live_control_schema.py

from pydantic import BaseModel
from typing import Optional

class CameraLiveControl(BaseModel):
    control_mode: Optional[str] = None
    exposure: Optional[int] = None
    gain: Optional[int] = None
    focus: Optional[int] = None
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    # Accepted for legacy clients, but camera AE/AF are always disabled.
    auto_focus: Optional[bool] = None
    autoFocus: Optional[bool] = None
    auto_exposure: Optional[bool] = None
    autoExposure: Optional[bool] = None
