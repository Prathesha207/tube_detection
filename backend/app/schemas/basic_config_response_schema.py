# app/schemas/basic_config_response_schema.py

from pydantic import BaseModel
from typing import Optional


class CameraBasicResponse(BaseModel):
    name: Optional[str] = None
    ip_address: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    rotation_angle: Optional[int] = None
    recording_video_path: Optional[str] = None
    recording_video_testing_path: Optional[str] = None

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


class TrainingBasicResponse(BaseModel):
    record_duration: Optional[int] = None
    feature_window: Optional[int] = None
    anomaly_scorer_nn_count: Optional[int] = None


class InferenceBasicResponse(BaseModel):
    trained_model: Optional[str] = None
    min_detection_area: Optional[int] = None
    use_dynamic_threshold: Optional[bool] = None
    processed_video: Optional[bool] = None
    raw_video: Optional[bool] = None
    mode: Optional[str] = None
    inference_video_path: Optional[str] = None
    inference_video_testing_path: Optional[str] = None
    model1_frame_count: Optional[int] = None
    model2_frame_count: Optional[int] = None
    model3_frame_count: Optional[int] = None
    model1_pass_frames: Optional[int] = None
    model2_pass_frames: Optional[int] = None
    model3_pass_frames: Optional[int] = None
    model4_absent_count: Optional[int] = None
    model2_start_skip_frame: Optional[int] = None
    model3_start_skip_frame: Optional[int] = None
    recording_format: Optional[str] = None
    inference_format: Optional[str] = None


class BasicConfigResponse(BaseModel):
    camera: Optional[CameraBasicResponse] = None
    training: TrainingBasicResponse
    inference: InferenceBasicResponse
