# app/schemas/basic_config_update_schema.py

from pydantic import BaseModel
from app.schemas.camera_schema import CameraUpdate
from typing import Optional


class TrainingConfigUpdate(BaseModel):
    record_duration: Optional[int] = None
    feature_window: Optional[int] = None
    anomaly_scorer_nn_count: Optional[int] = None


class InferenceConfigUpdate(BaseModel):
    trained_model: Optional[str] = None
    min_detection_area: Optional[int] = None
    use_dynamic_threshold: Optional[bool] = None
    processed_video: Optional[bool] = None
    raw_video: Optional[bool] = None
    save_failed_frame: Optional[bool] = None
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

class BasicConfigUpdate(BaseModel):
    camera: Optional[CameraUpdate]
    training: Optional[TrainingConfigUpdate]
    inference: Optional[InferenceConfigUpdate]


    