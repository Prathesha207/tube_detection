from typing import Optional

from pydantic import BaseModel

from app.schemas.camera_schema import CameraUpdate


class InferenceConfigUpdate(BaseModel):
    mode: Optional[str] = None


class BasicConfigUpdate(BaseModel):
    camera: Optional[CameraUpdate]
    inference: Optional[InferenceConfigUpdate]


class CameraBasicResponse(BaseModel):
    name: Optional[str] = None
    ip_address: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    rotation_angle: Optional[int] = None
    recording_video_path: Optional[str] = None
    control_mode: Optional[str] = None
    exposure: Optional[int] = None
    gain: Optional[int] = None
    focus: Optional[int] = None
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    auto_exposure: Optional[bool] = None
    auto_focus: Optional[bool] = None
    stream_url: Optional[str] = None
    is_inference_enabled: Optional[bool] = None
    roi_enabled: Optional[bool] = None
    is_enabled: Optional[bool] = None

    class Config:
        from_attributes = True


class InferenceBasicResponse(BaseModel):
    mode: Optional[str] = None


class BasicConfigResponse(BaseModel):
    camera: Optional[CameraBasicResponse] = None
    inference: InferenceBasicResponse
