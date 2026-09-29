from pydantic import BaseModel
from typing import List

class VideoUploadResponse(BaseModel):
    frame_count: int
    frames: List[str]