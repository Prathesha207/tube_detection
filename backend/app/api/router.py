from fastapi import APIRouter

from app.routers.video_router import router as new_video_router
from app.api import (
    camera_api,
    recording_api
)
from app.routers import oak_camera_router
from app.routers.realtime_logs import router as realtime_logs_router
from app.routers.roi_router import router as roi_router


router = APIRouter()

router.include_router(camera_api.router, prefix="/camera", tags=["Camera"])
router.include_router(recording_api.router, prefix="/recording", tags=["Recording"])
router.include_router(new_video_router, prefix="/video", tags=["Video"])
router.include_router(oak_camera_router.router, prefix="/oak", tags=["OAK Camera"])
router.include_router(realtime_logs_router)
router.include_router(roi_router, prefix="/roi", tags=["ROI"])
router.include_router(roi_router, prefix="/video/roi", tags=["ROI"])
