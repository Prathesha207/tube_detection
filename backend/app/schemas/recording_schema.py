from typing import Optional
from pydantic import BaseModel


# ================= START =================
class StartRecordingRequest(BaseModel):
    session_id: str
    recording_format: Optional[str] = None


# ================= FRAME (optional / keep if needed) =================
class FrameRequest(BaseModel):
    session_id: str
    frame: str


# ================= STOP =================
class StopRecordingRequest(BaseModel):
    session_id: str