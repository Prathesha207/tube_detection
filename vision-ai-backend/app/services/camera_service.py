# app/services/camera_service.py

from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.camera_model import Camera


from app.schemas.camera_schema import CameraCreate, CameraUpdate
from app.schemas.basic_config_update_schema import BasicConfigUpdate

from app.core.logger import setup_logger
from app.services.realtime_log_service import realtime_log_service

logger = setup_logger("camera-service")


# =========================================================
# CAMERA CRUD
# =========================================================

def create_camera(db: Session, data: CameraCreate):
    try:
        logger.info("[CREATE_CAMERA] Called")

        camera = Camera(**data.model_dump())

        # Apply control mode rules
        if camera.control_mode == "auto":
            camera.exposure = None
            camera.gain = None
            camera.focus = None
            camera.auto_exposure = True
            camera.auto_focus = True
        else:
            camera.auto_exposure = False
            camera.auto_focus = False

        db.add(camera)
        db.commit()
        db.refresh(camera)

        logger.info(f"[CREATE_CAMERA] Created camera ID: {camera.id}")
        realtime_log_service.add_log(
            "camera",
            "CAMERA",
            f"Camera created: {camera.name}",
            "success"
        )
        return camera
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"[CREATE_CAMERA] Failed error: {e}", exc_info=True)
        realtime_log_service.add_log(
            "camera",
            "ERROR",
            f"Failed to create camera: {e}",
            "error"
        )
        raise HTTPException(status_code=500, detail=f"Failed to create camera: {e}")


def update_camera_partial(db: Session, camera_id: int, data: CameraUpdate):
    try:
        logger.info(f"[UPDATE_CAMERA] Called with ID: {camera_id}")

        camera = db.query(Camera).filter(Camera.id == camera_id).first()

        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        update_data = data.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            setattr(camera, key, value)

        # Apply control mode AFTER update
        if camera.exposure is not None or camera.gain is not None or camera.focus is not None:
            camera.control_mode = "manual"
            if camera.auto_exposure is None:
                camera.auto_exposure = False
            if camera.auto_focus is None:
                camera.auto_focus = False
        elif camera.control_mode == "auto":
            camera.exposure = None
            camera.gain = None
            camera.focus = None
            camera.auto_exposure = True
            camera.auto_focus = True
        elif camera.control_mode == "manual":
            camera.auto_exposure = False
            camera.auto_focus = False

        db.commit()
        db.refresh(camera)

        # Sync runtime controls to physical OAK camera if pipeline is currently active
        try:
            from app.services.oak_camera_service import oak_camera_service
            if oak_camera_service.is_running:
                oak_camera_service.update_controls(
                    exposure=camera.exposure,
                    gain=camera.gain,
                    focus=camera.focus,
                    brightness=camera.brightness,
                    contrast=camera.contrast,
                    auto_focus=camera.auto_focus,
                    auto_exposure=camera.auto_exposure,
                )
        except Exception as sync_err:
            logger.debug(f"[UPDATE_CAMERA] Dynamic control sync skipped: {sync_err}")

        logger.info("[UPDATE_CAMERA] Success")
        realtime_log_service.add_log(
            "camera",
            "CAMERA",
            f"Camera updated: ID {camera_id}",
            "success"
        )
        return camera

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"[UPDATE_CAMERA] Failed error: {e}", exc_info=True)
        realtime_log_service.add_log(
            "camera",
            "ERROR",
            f"Update camera failed: {e}",
            "error"
        )
        raise HTTPException(status_code=500, detail=f"Failed to update camera: {e}")

def enable_camera(db: Session, camera_id: int):
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        camera.is_enabled = True
        db.commit()
        db.refresh(camera)
        realtime_log_service.add_log(
            "camera",
            "CAMERA",
            f"Camera enabled: ID {camera_id}",
            "success"
        )
        return camera

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"[ENABLE_CAMERA] Failed error: {e}", exc_info=True)
        realtime_log_service.add_log(
            "camera",
            "ERROR",
            f"Enable failed - ID {camera_id}: {e}",
            "error"
        )
        raise HTTPException(status_code=500, detail=f"Failed to enable camera: {e}")


def disable_camera(db: Session, camera_id: int):
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        camera.is_enabled = False
        db.commit()
        db.refresh(camera)
        realtime_log_service.add_log(
            "camera",
            "CAMERA",
            f"Camera disabled: ID {camera_id}",
            "success"
        )
        return camera

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"[DISABLE_CAMERA] Failed error: {e}", exc_info=True)
        realtime_log_service.add_log(
            "camera",
            "ERROR",
            f"Disable failed - ID {camera_id}: {e}",
            "error"
        )
        raise HTTPException(status_code=500, detail=f"Failed to disable camera: {e}")


def delete_camera(db: Session, camera_id: int):
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        db.delete(camera)
        db.commit()
        realtime_log_service.add_log(
            "camera",
            "CAMERA",
            f"Camera deleted: ID {camera_id}",
            "success"
        )
        return {"status": "success", "message": f"Camera {camera_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"[DELETE_CAMERA] Failed error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_cameras(db: Session):
    return db.query(Camera).all()


def get_camera_config(db: Session):
    # 1. Prefer real enabled hardware cameras (exclude mock 127.0.0.1)
    real_cam = (
        db.query(Camera)
        .filter(
            Camera.ip_address.isnot(None),
            Camera.ip_address != "127.0.0.1",
            Camera.is_enabled == True,
        )
        .order_by(Camera.id.desc())
        .first()
    )
    if real_cam:
        return real_cam

    # 2. Fallback to any enabled camera
    enabled_cam = (
        db.query(Camera)
        .filter(Camera.is_enabled == True)
        .order_by(Camera.id.desc())
        .first()
    )
    if enabled_cam:
        return enabled_cam

    # 3. Fallback to newest camera
    return db.query(Camera).order_by(Camera.id.desc()).first()


def get_inference_config(db: Session):
    return {"mode": "production"}


def update_inference_mode(db: Session, mode: str):
    return {"message": "Inference mode updated", "mode": mode}

def update_basic_config(db: Session, data: BasicConfigUpdate):
    logger.info("[UPDATE_BASIC_CONFIG] Started")
    realtime_log_service.add_log(
        "system",
        "CONFIG",
        "Updating system configuration",
        "info"
    )

    try:
        # =========================================================
        # CAMERA
        # =========================================================
        camera = db.query(Camera).first()

        if not camera:
            logger.info("[CONFIG] Creating Camera")

            if not data.camera or not data.camera.name:
                raise HTTPException(status_code=400, detail="Camera name is required")

            camera_data = data.camera.model_dump(exclude_unset=True)
            camera = Camera(**camera_data)
            db.add(camera)

        else:
            if data.camera:
                logger.info("[CONFIG] Updating Camera")

                cam_data = data.camera.model_dump(exclude_unset=True)
                for key, value in cam_data.items():
                    setattr(camera, key, value)

        # Apply control mode AFTER update/create
        if camera.control_mode == "auto":
            camera.exposure = None
            camera.gain = None
            camera.focus = None
            camera.auto_exposure = True
            camera.auto_focus = True
        elif camera.control_mode == "manual":
            camera.auto_exposure = False
            camera.auto_focus = False

        # =========================================================
        # TRAINING
        # =========================================================
        

        # =========================================================
        # INFERENCE
        # =========================================================
        

        db.commit()

        logger.info("[UPDATE_BASIC_CONFIG] Success")
        realtime_log_service.add_log(
            "system",
            "CONFIG",
            "Configuration updated successfully",
            "success"
        )

        return {
            "message": "Config updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"[UPDATE_BASIC_CONFIG ERROR] {e}", exc_info=True)
        realtime_log_service.add_log(
            "system",
            "ERROR",
            "Configuration update failed",
            "error"
        )
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# GET CONFIG
# =========================================================

def get_basic_config(db: Session):
    logger.info("[GET_BASIC_CONFIG] Fetching configs")
    realtime_log_service.add_log(
        "system",
        "CONFIG",
        "Fetching configuration",
        "info"
    )

    camera = db.query(Camera).first()
    

    return {
        "camera": camera,
        
        "inference": {"mode": "production"}
    }