import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np
import depthai as dai
from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models.camera_model import Camera
from app.services import camera_service
from app.services.oak_camera_service import oak_camera_service
from app.services.realtime_log_service import realtime_log_service

logger = logging.getLogger("oak-camera")

router = APIRouter()



class StartCameraPayload(BaseModel):
    camera_id: Optional[int] = None
    ip_address: Optional[str] = None

# ==================== App Lifecycle ====================

@router.post("/start")
async def start_camera(payload: Optional[StartCameraPayload] = None, db: Session = Depends(get_db)):
    """App startup — connect device and build pipeline."""
    camera = None
    if payload and payload.camera_id:
        camera = db.query(Camera).filter(Camera.id == payload.camera_id).first()
    elif payload and payload.ip_address:
        camera = db.query(Camera).filter(Camera.ip_address == payload.ip_address).first()

    if not camera:
        camera = camera_service.get_camera_config(db)

    if not camera or not camera.ip_address:
        try:
            device_info = oak_camera_service._resolve_device_info("usb")
            mxid = str(getattr(device_info, "mxid", "") or "")
            if not mxid:
                mxid = "usb"
            
            if not camera:
                from app.schemas.camera_schema import CameraCreate
                new_cam = CameraCreate(name="Auto USB Camera", ip_address=mxid, control_mode="auto")
                camera = camera_service.create_camera(db, new_cam)
            else:
                from app.schemas.camera_schema import CameraUpdate
                camera = camera_service.update_camera_partial(db, camera.id, CameraUpdate(ip_address=mxid))
        except Exception as e:
            logger.warning(f"[OAK DEVICE ENUM] {e}")
            realtime_log_service.add_log("camera", "WARN", f"No camera connected: {e}", "warning")
            return {"status": "error", "message": f"No camera connected: {str(e)}"}

    try:
        result = await oak_camera_service.start(camera)
        return result
    except Exception as e:
        logger.error(f"[API ERROR] POST /oak/start: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Start camera failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {e}")


@router.post("/stop")
async def stop_camera():
    """App shutdown — stop everything and disconnect."""
    try:
        result = await oak_camera_service.stop()
        return result
    except Exception as e:
        logger.error(f"[API ERROR] POST /oak/stop: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Stop camera failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to stop camera: {e}")


# ==================== Stream Lifecycle ====================

@router.post("/stream/start")
async def start_stream():
    """User clicks Start Streaming — starts the 3 capture threads."""
    try:
        result = await oak_camera_service.start_streaming()
        return result
    except Exception as e:
        logger.error(f"[API ERROR] POST /oak/stream/start: {e}", exc_info=True)
        realtime_log_service.add_log("stream", "CRASH", f"Start stream failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")


@router.post("/stream/stop")
async def stop_stream():
    """User clicks Stop Streaming — stops capture threads and clears queues."""
    try:
        result = await oak_camera_service.stop_streaming()
        return result
    except Exception as e:
        logger.error(f"[API ERROR] POST /oak/stream/stop: {e}", exc_info=True)
        realtime_log_service.add_log("stream", "CRASH", f"Stop stream failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to stop stream: {e}")


# ==================== MJPEG Stream ====================

@router.get("/inference/stream/{session_id}")
@router.get("/inference/stream/live")
@router.get("/stream/live")
@router.get("/stream")
async def stream(request: Request, session_id: Optional[str] = None):
    """MJPEG HTTP stream — connect to this after stream/start or on canvas load.
    Subscribes a client frame queue, yields frames until client disconnects,
    then unsubscribes cleanly without interrupting other clients.
    """
    if not oak_camera_service._is_running:
        # Auto-start camera from database if configured
        try:
            from app.core.database import SessionLocal
            db = SessionLocal()
            try:
                cam = camera_service.get_camera_config(db)
                if cam:
                    logger.info(f"[STREAM] Auto-starting camera ID {cam.id} ({cam.ip_address}) for stream")
                    await oak_camera_service.start(cam)
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"[STREAM] Auto-start camera failed: {e}")

    if not oak_camera_service._is_running:
        return Response(
            content=b"Camera not started",
            status_code=503,
            media_type="text/plain",
        )

    # Ensure capture threads are active so frames are actively produced
    if not (oak_camera_service._mjpeg_thread and oak_camera_service._mjpeg_thread.is_alive()):
        logger.info("[STREAM] Capture threads not running — starting them now")
        oak_camera_service._start_capture_threads()

    oak_camera_service._is_streaming = True
    client_queue = oak_camera_service.subscribe_stream()

    async def generate():
        frames_sent = 0
        start_time = time.time()

        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"[STREAM] Client disconnected — frames sent: {frames_sent}")
                    break

                if not oak_camera_service._is_streaming:
                    logger.info(f"[STREAM] Stream stopped — ending client stream cleanly (frames sent: {frames_sent})")
                    break

                try:
                    jpeg = await asyncio.wait_for(client_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    if not oak_camera_service._is_running or not oak_camera_service._is_streaming:
                        break
                    # If capture threads died while streaming is active, attempt restart
                    if oak_camera_service._is_streaming and not (oak_camera_service._mjpeg_thread and oak_camera_service._mjpeg_thread.is_alive()):
                        oak_camera_service._start_capture_threads()
                    continue

                if jpeg is None:
                    if not oak_camera_service._is_streaming:
                        break
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg +
                    b"\r\n"
                )

                frames_sent += 1

                if frames_sent % 150 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[STREAM] Frames sent: {frames_sent} | "
                        f"avg FPS: {frames_sent / max(elapsed, 1):.1f}"
                    )

        except Exception as e:
            logger.error(f"[STREAM] Generator error: {e}")
        finally:
            elapsed = time.time() - start_time
            logger.info(
                f"[STREAM] Ended — frames sent: {frames_sent} | "
                f"duration: {elapsed:.1f}s | "
                f"avg FPS: {frames_sent / max(elapsed, 1):.1f}"
            )
            oak_camera_service.unsubscribe_stream(client_queue)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


# ==================== Snapshot ====================

@router.get("/snapshot")
async def snapshot():
    """Single JPEG frame — returns latest live frame or fallback to prevent UI errors."""
    # 1. Try from stream queue / buffer
    frame = await oak_camera_service.get_stream_frame(timeout=0.5)
    if frame is not None:
        return Response(content=frame, media_type="image/jpeg")

    # 2. Try latest BGR frame from converter
    bgr = oak_camera_service.get_bgr_frame()
    if bgr is not None:
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            return Response(content=buf.tobytes(), media_type="image/jpeg")

    # 3. If capture threads are alive, wait briefly for frame
    if oak_camera_service._mjpeg_thread and oak_camera_service._mjpeg_thread.is_alive():
        frame = await oak_camera_service.get_stream_frame(timeout=1.0)
        if frame is not None:
            return Response(content=frame, media_type="image/jpeg")

    # 4. Fallback: 1280x720 dark frame to avoid 503 error triggering frontend disconnect
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ==================== Health & Devices ====================

@router.get("/health")
def health():
    return oak_camera_service.health()


@router.get("/devices")
async def get_available_devices():
    """Discover connected Luxonis OAK-D devices over USB and Network."""
    devices = []
    loop = asyncio.get_running_loop()
    try:
        raw_devices = await loop.run_in_executor(None, dai.Device.getAllAvailableDevices)
        for d in raw_devices:
            mxid = str(getattr(d, "mxid", "") or "")
            name = str(getattr(d, "name", "") or "")
            protocol = str(getattr(d, "protocol", "") or "")
            state = str(getattr(d, "state", "") or "")
            desc_name = ""
            try:
                desc = d.getXLinkDeviceDesc()
                desc_name = str(getattr(desc, "name", "") or "")
            except Exception:
                pass
            is_usb = "usb" in protocol.lower() or "usb" in name.lower() or "usb" in state.lower()
            devices.append({
                "mxid": mxid,
                "name": desc_name or name or (f"OAK USB ({mxid[:8]})" if is_usb else f"OAK Network ({mxid[:8]})"),
                "ip_or_id": mxid if is_usb else (desc_name or name or mxid),
                "protocol": "USB" if is_usb else "TCP/IP",
                "state": state,
                "is_usb": is_usb,
            })
    except Exception as e:
        logger.warning(f"[DEVICES] Failed to enumerate DepthAI devices: {e}")
    return devices


# ==================== Camera Controls ====================

class CameraControlsRequest(BaseModel):
    exposure: Optional[int] = None
    gain: Optional[int] = None
    focus: Optional[int] = None
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    auto_focus: Optional[bool] = None
    autoFocus: Optional[bool] = None
    auto_exposure: Optional[bool] = None
    autoExposure: Optional[bool] = None

@router.post("/controls")
def update_controls(
    body: Optional[CameraControlsRequest] = Body(default=None),
    exposure: int | None = None,
    gain: int | None = None,
    focus: int | None = None,
    brightness: int | None = None,
    contrast: int | None = None,
    auto_focus: bool | None = None,
    auto_exposure: bool | None = None,
):
    try:
        exp = body.exposure if body and body.exposure is not None else exposure
        g = body.gain if body and body.gain is not None else gain
        f = body.focus if body and body.focus is not None else focus
        b = body.brightness if body and body.brightness is not None else brightness
        c = body.contrast if body and body.contrast is not None else contrast
        af = (body.auto_focus if body and body.auto_focus is not None else (body.autoFocus if body and body.autoFocus is not None else auto_focus))
        ae = (body.auto_exposure if body and body.auto_exposure is not None else (body.autoExposure if body and body.autoExposure is not None else auto_exposure))

        oak_camera_service.update_controls(
            exposure=exp,
            gain=g,
            focus=f,
            brightness=b,
            contrast=c,
            auto_focus=af,
            auto_exposure=ae,
        )
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /oak/controls: {e}", exc_info=True)
        realtime_log_service.add_log("camera", "CRASH", f"Camera controls update failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to update controls: {e}")


# ==================== Inference ====================

class InferenceStartBody(BaseModel):
    offline: bool = False
    videoPath: Optional[str] = None   # absolute path to a single video file
    folderPath: Optional[str] = None  # absolute path to a folder of videos
    videoId: Optional[str] = None     # legacy field — treated as videoPath if videoPath is absent


@router.post("/inference/start/{session_id}")
async def start_inference(session_id: str, body: InferenceStartBody = InferenceStartBody()):
    try:
        loop = asyncio.get_running_loop()
        # Resolve legacy videoId field as videoPath so old frontend builds still work
        resolved_video_path = body.videoPath or body.videoId or None
        result = oak_camera_service.start_inference(
            session_id, loop,
            offline=body.offline,
            video_path=resolved_video_path,
            folder_path=body.folderPath,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /oak/inference/start/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("inference", "CRASH", f"Inference start failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to start inference: {e}")


@router.post("/inference/stop")
def stop_inference():
    try:
        result = oak_camera_service.stop_inference()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /oak/inference/stop: {e}", exc_info=True)
        realtime_log_service.add_log("inference", "CRASH", f"Inference stop failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to stop inference: {e}")

@router.get("/inference/status/{session_id}")
async def get_camera_inference_status(session_id: str):
    try:
        from app.ml.tube.inference.camera_inference_service import get_session_status
        status = get_session_status(session_id)
        if not status:
            return {"status": "idle", "session_id": session_id}
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /oak/inference/status/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("inference", "CRASH", f"Get inference status failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to get inference status: {e}")


@router.websocket("/inference/ws/{session_id}")
async def inference_ws(websocket: WebSocket, session_id: str):
    """WebSocket that streams inference results to the UI.
    UI handles the MJPEG stream separately — this only sends JSON results.
    """
    await websocket.accept()
    logger.info(f"[INFERENCE WS] ✅ Client connected — session: {session_id}")
    logger.info(f"[INFERENCE WS] Waiting for inference results — inference_running: {oak_camera_service._inference_thread is not None and oak_camera_service._inference_thread.is_alive()}")

    results_sent = 0

    try:
        while True:
            result = await oak_camera_service.get_inference_result(timeout=1.0)

            if result is None:
                # Inference has stopped — queue was cleared; exit so the event loop is not starved
                if oak_camera_service._inference_result_queue is None:
                    logger.info(f"[INFERENCE WS] Inference stopped — closing")
                    break
                continue

            results_sent += 1

            # Terminal result — send it, then close cleanly
            if result.get("done"):
                logger.info(f"[INFERENCE WS] Terminal result received (status={result.get('status')}) — closing")
                await websocket.send_json(result)
                break

            logger.info(
                f"[INFERENCE WS] #{results_sent} → status={result.get('status')} | "
                f"record={result.get('record')} | "
                f"detections={len(result.get('detections', []))} | "
                f"fps={result.get('metrics', {}).get('fps', '?')} | "
                f"gpu={result.get('metrics', {}).get('gpu_pct', '?')}% | "
                f"latency={result.get('metrics', {}).get('latency_ms', '?')}ms"
            )

            ws_result = dict(result)
            if "_raw_frame" in ws_result:
                import cv2, base64, asyncio
                frame = ws_result.pop("_raw_frame")
                loop = asyncio.get_running_loop()
                def _encode():
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
                ws_result["frame"] = await loop.run_in_executor(None, _encode)

            await websocket.send_json(ws_result)

    except WebSocketDisconnect:
        logger.info(f"[INFERENCE WS] Client disconnected — session: {session_id} | total sent: {results_sent}")
    except Exception as e:
        logger.error(f"[INFERENCE WS] Error: {e}", exc_info=True)
    finally:
        logger.info(f"[INFERENCE WS] Closed — session: {session_id}")
        try:
            await websocket.close()
        except RuntimeError:
            pass