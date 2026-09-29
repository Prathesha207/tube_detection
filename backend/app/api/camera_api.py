import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.schemas.camera_schema import CameraCreate, CameraResponse, CameraUpdate
from app.schemas.basic_config_update_schema import BasicConfigUpdate
from app.schemas.basic_config_response_schema import BasicConfigResponse

from app.services import camera_service
from app.schemas.camera_live_control_schema import CameraLiveControl
from app.services.oak_camera_service import oak_camera_service
from app.services.realtime_log_service import realtime_log_service


class InferenceModeUpdate(BaseModel):
    mode: Literal["testing", "production"]

logger = logging.getLogger("camera-api")

router = APIRouter()


@router.post("/create", response_model=CameraResponse)
def create_camera(data: CameraCreate, db: Session = Depends(get_db)):
    try:
        return camera_service.create_camera(db, data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /camera/create: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Create camera failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error creating camera: {e}")


@router.put("/update/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: int, data: CameraUpdate, db: Session = Depends(get_db)):
    try:
        return camera_service.update_camera_partial(db, camera_id, data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] PUT /camera/update/{camera_id}: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Update camera {camera_id} failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error updating camera: {e}")


@router.post("/enable/{camera_id}", response_model=CameraResponse)
def enable_camera(camera_id: int, db: Session = Depends(get_db)):
    try:
        return camera_service.enable_camera(db, camera_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /camera/enable/{camera_id}: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Enable camera {camera_id} failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error enabling camera: {e}")


@router.post("/disable/{camera_id}", response_model=CameraResponse)
def disable_camera(camera_id: int, db: Session = Depends(get_db)):
    try:
        return camera_service.disable_camera(db, camera_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /camera/disable/{camera_id}: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Disable camera {camera_id} failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error disabling camera: {e}")


@router.delete("/{camera_id}")
@router.delete("/delete/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    try:
        return camera_service.delete_camera(db, camera_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] DELETE /camera/{camera_id}: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Delete camera {camera_id} failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error deleting camera: {e}")


@router.get("/", response_model=List[CameraResponse])
def get_cameras(db: Session = Depends(get_db)):
    try:
        return camera_service.get_cameras(db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /camera/: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Get cameras failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error fetching cameras: {e}")


@router.put("/basic-config-update")
def update_basic_config(
    data: BasicConfigUpdate,
    db: Session = Depends(get_db)
):
    try:
        return camera_service.update_basic_config(db, data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] PUT /camera/basic-config-update: {e}", exc_info=True)
        realtime_log_service.add_log("system", "CRASH", f"Update basic config failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error updating config: {e}")


@router.get("/config", response_model=BasicConfigResponse)
def get_basic_config(db: Session = Depends(get_db)):
    try:
        return camera_service.get_basic_config(db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /camera/config: {e}", exc_info=True)
        realtime_log_service.add_log("system", "CRASH", f"Get basic config failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error fetching config: {e}")


@router.patch("/inference-mode")
def update_inference_mode(data: InferenceModeUpdate, db: Session = Depends(get_db)):
    return camera_service.update_inference_mode(db, data.mode)


@router.patch("/live-controls/{camera_id}")
def update_live_controls(camera_id: int, data: CameraLiveControl):
    logger.info(
        f"[LIVE CONTROLS] camera_id={camera_id} | "
        f"exposure={data.exposure}, gain={data.gain}, focus={data.focus}, "
        f"brightness={data.brightness}, contrast={data.contrast}"
    )

    if not oak_camera_service._is_running:
        logger.warning("[LIVE CONTROLS] Rejected — pipeline not running")
        return {"status": "error", "message": "Camera is not running"}

    try:
        af = data.auto_focus if data.auto_focus is not None else data.autoFocus
        ae = data.auto_exposure if data.auto_exposure is not None else data.autoExposure
        oak_camera_service.update_controls(
            exposure=data.exposure,
            gain=data.gain,
            focus=data.focus,
            brightness=data.brightness,
            contrast=data.contrast,
            auto_focus=af,
            auto_exposure=ae,
        )
        logger.info("[LIVE CONTROLS] Applied successfully")
        return {"status": "success", "message": "Live controls applied"}

    except Exception as e:
        logger.error(f"[LIVE CONTROLS] Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}