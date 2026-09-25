import os
import logging
import cv2

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.recording_schema import StartRecordingRequest, StopRecordingRequest
from app.services import camera_service
from app.services.oak_camera_service import oak_camera_service
from app.ml.tube.inference.video_inference_service import video_inference_service, ml_inference_service
from app.services.realtime_log_service import realtime_log_service

logger = logging.getLogger("recording-api")

router = APIRouter()


@router.post("/start")
def start_recording(data: StartRecordingRequest, db: Session = Depends(get_db)):
    try:
        camera_config = camera_service.get_camera_config(db)
        if not camera_config:
            raise HTTPException(status_code=404, detail="Camera config not found")

        resolution = camera_config.resolution or "1920x1080"
        try:
            sep = "x" if "x" in resolution else "*"
            width, height = map(int, resolution.split(sep))
        except Exception:
            logger.warning(f"[RECORD] Invalid resolution '{resolution}' — fallback 1920x1080")
            width, height = 1920, 1080

        path = oak_camera_service.start_recording(
            session_id=data.session_id,
            width=width,
            height=height,
            fps=float(camera_config.fps or 30),
            recording_format=data.recording_format,
        )
        return {
            "status": "recording",
            "session_id": data.session_id,
            "recording_path": path,
        }
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.warning(f"[API ERROR] POST /recording/start failed: {e}")
        realtime_log_service.add_log("record", "ERROR", str(e), "error")
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"[API ERROR] POST /recording/start failed: {e}", exc_info=True)
        realtime_log_service.add_log("record", "CRASH", f"Start recording failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error starting recording: {e}")


@router.post("/stop")
def stop_recording(data: StopRecordingRequest):
    try:
        result = oak_camera_service.stop_recording(data.session_id)
        video_path = result.get("recording_path")
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"[RECORD] File not found or empty on stop: {video_path}")
            raise HTTPException(status_code=404, detail="Recording session ended but file was not found")

        filename = result.get("filename", os.path.basename(video_path))
        logger.info(f"[RECORD] Successfully finalized {filename} at {video_path}")

        # Register into ML inference service so frontend can review/run immediately
        ml_session_id = ml_inference_service.create_session(
            original_filename=filename,
        )

        frame0_bytes = None
        w, h = result.get("width", 1920), result.get("height", 1080)
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame0 = cap.read()
            if ret and frame0 is not None:
                h, w = frame0.shape[:2]
                _, buf = cv2.imencode(".jpg", frame0, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame0_bytes = buf.tobytes()
            cap.release()
        except Exception as e:
            logger.warning(f"[RECORD] Could not read preview frame: {e}")

        session = ml_inference_service.sessions.get(ml_session_id)
        if session:
            session["inference_video_path"] = video_path
            session["browser_video_path"] = video_path
            session["status"] = "ready"
            session["stats"]["status"] = "ready"
            session["stats"]["video_width"] = w
            session["stats"]["video_height"] = h
            if frame0_bytes:
                session["last_frame_bytes"] = frame0_bytes

        return {
            "status": "done",
            "session_id": ml_session_id,
            "recording_session_id": data.session_id,
            "recording_path": video_path,
            "filename": filename,
            "duration": result.get("duration", 0),
            "frames": result.get("frames_written", 0),
            "stream_url": f"/video/stream/{ml_session_id}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /recording/stop failed: {e}", exc_info=True)
        realtime_log_service.add_log("record", "CRASH", f"Stop recording failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error stopping recording: {e}")


@router.get("/status")
def recording_status():
    try:
        from app.services.recording_service import active_recordings
        active = oak_camera_service._active_recording is not None
        return {
            "is_recording": active,
            "active_count": len(active_recordings),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /recording/status failed: {e}", exc_info=True)
        realtime_log_service.add_log("record", "CRASH", f"Recording status check failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Internal error checking recording status: {e}")

