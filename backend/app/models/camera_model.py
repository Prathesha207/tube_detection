from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float
from app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(100), nullable=False)

    ip_address = Column(String(255), nullable=True)

    # New Fields
    resolution = Column(String(50), nullable=True)  
    # Example: "1920x1080", "1280x720"

    fps = Column(Integer, default=30)  
    # Frames per second (allowed: 5–30, enforce in validation layer)

    rotation_angle = Column(Integer, default=0)  
    # Example: 0, 90, 180, 270
    
    #  Mode
    control_mode = Column(String(20), default="auto")  
    # "auto" | "manual"
    
     # Manual Controls (ONLY used when manual mode)
    exposure = Column(Float, nullable=True)
    gain = Column(Float, nullable=True)
    focus = Column(Integer, nullable=True)

    brightness = Column(Integer, nullable=True)
    contrast = Column(Integer, nullable=True)

    # Auto Toggles
    auto_exposure = Column(Boolean, default=True)
    auto_focus = Column(Boolean, default=True)
    
     # Streaming / AI
    stream_url = Column(String(500), nullable=True)
    is_inference_enabled = Column(Boolean, default=False)
    roi_enabled = Column(Boolean, default=False)
    recording_video_path = Column(String, nullable=True)
    recording_video_testing_path = Column(String, nullable=True)
    recording_format = Column(String(20), default="MP4")

    is_enabled = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)