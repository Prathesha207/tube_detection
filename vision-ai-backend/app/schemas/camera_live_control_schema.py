# app/schemas/camera_live_control_schema.py

from pydantic import BaseModel
from typing import Optional

class CameraLiveControl(BaseModel):
    exposure: Optional[int] = None
    gain: Optional[int] = None
    focus: Optional[int] = None
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    auto_focus: Optional[bool] = None
    autoFocus: Optional[bool] = None