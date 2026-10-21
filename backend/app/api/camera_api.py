import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
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
            control_mode=data.control_mode,
        )
        logger.info("[LIVE CONTROLS] Applied successfully")
        return {"status": "success", "message": "Live controls applied"}

    except Exception as e:
        logger.error(f"[LIVE CONTROLS] Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


class StoragePathUpdate(BaseModel):
    path: Optional[str] = None


class StoragePathResponse(BaseModel):
    storage_path: str
    default_path: str
    status: str = "success"


class BrowseFolderRequest(BaseModel):
    initial_dir: Optional[str] = None


@router.get("/storage-path", response_model=StoragePathResponse)
def get_storage_path():
    try:
        from app.core.app_paths import get_configured_storage_dir, get_desktop_dir
        return StoragePathResponse(
            storage_path=str(get_configured_storage_dir()),
            default_path=str(get_desktop_dir()),
            status="success",
        )
    except Exception as e:
        logger.error(f"[API ERROR] GET /camera/storage-path: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/storage-path", response_model=StoragePathResponse)
def set_storage_path(data: StoragePathUpdate):
    try:
        from app.core.app_paths import set_configured_storage_dir, get_desktop_dir
        final_path = set_configured_storage_dir(data.path)
        realtime_log_service.add_log(
            "camera",
            "CONFIG",
            f"Storage path updated: {final_path}",
            "success",
        )
        return StoragePathResponse(
            storage_path=final_path,
            default_path=str(get_desktop_dir()),
            status="success",
        )
    except Exception as e:
        logger.error(f"[API ERROR] POST /camera/storage-path: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/browse-folder")
def browse_folder(data: Optional[BrowseFolderRequest] = None):
    try:
        import concurrent.futures
        import os
        from app.core.app_paths import get_configured_storage_dir, get_desktop_dir

        req_dir = data.initial_dir if data and data.initial_dir else None
        start_dir = req_dir or str(get_configured_storage_dir()) or str(get_desktop_dir())

        def _open_dialog():
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", 1)
            selected = filedialog.askdirectory(
                initialdir=start_dir,
                title="Select Directory for Recordings & Inference Results",
            )
            root.destroy()
            return selected

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_open_dialog)
            selected_path = future.result(timeout=120.0)

        if selected_path:
            norm_path = os.path.normpath(selected_path)
            return {"path": norm_path, "canceled": False}
        return {"path": None, "canceled": True}
    except Exception as e:
        logger.error(f"[API ERROR] POST /camera/browse-folder: {e}", exc_info=True)
        return {"path": None, "canceled": True, "error": str(e)}

