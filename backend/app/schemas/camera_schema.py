from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal


class CameraBase(BaseModel):
    name: str
    ip_address: Optional[str] = None

    resolution: Optional[str] = None
    fps: int = Field(default=30, ge=5, le=30)
    rotation_angle: int = Field(default=0, ge=0, le=360)

    # Mode
    control_mode: Literal["auto", "manual"] = "auto"

    # Manual Controls
    exposure: Optional[float] = Field(default=None, ge=0)
    gain: Optional[float] = Field(default=None, ge=0)
    focus: Optional[int] = Field(default=None, ge=0)

    brightness: Optional[int] = Field(default=None, ge=0, le=100)
    contrast: Optional[int] = Field(default=None, ge=0, le=100)

    # Auto Toggles
    auto_exposure: bool = True
    auto_focus: bool = True

    # Streaming / AI
    stream_url: Optional[str] = None
    is_inference_enabled: bool = False
    roi_enabled: bool = False
    recording_video_path: Optional[str] = None
    recording_video_testing_path: Optional[str] = None
    recording_format: Optional[str] = "MP4"


class CameraCreate(CameraBase):
    is_enabled: bool = True


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    ip_address: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = Field(default=None, ge=5, le=30)
    rotation_angle: Optional[int] = Field(default=None, ge=0, le=360)

    control_mode: Optional[Literal["auto", "manual"]] = None

    exposure: Optional[float] = Field(default=None, ge=0)
    gain: Optional[float] = Field(default=None, ge=0)
    focus: Optional[int] = Field(default=None, ge=0)

    brightness: Optional[int] = Field(default=None, ge=0, le=100)
    contrast: Optional[int] = Field(default=None, ge=0, le=100)

    auto_exposure: Optional[bool] = None
    auto_focus: Optional[bool] = None

    stream_url: Optional[str] = None
    is_inference_enabled: Optional[bool] = None
    roi_enabled: Optional[bool] = None

    is_enabled: Optional[bool] = None
    recording_video_path: Optional[str] = None
    recording_video_testing_path: Optional[str] = None
    recording_format: Optional[str] = None


class CameraResponse(CameraBase):
    id: int
    is_enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True