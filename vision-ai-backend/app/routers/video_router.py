from typing import Optional
import logging
import tempfile
import asyncio
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response

from app.ml.video_inference_service import video_inference_service, ml_inference_service
from app.services.realtime_log_service import realtime_log_service

logger = logging.getLogger("video-router")
router = APIRouter()

import os
import subprocess
import imageio_ffmpeg

@router.post("/upload")
async def upload_video(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None),
    expected_ducks: int = Form(18),
    is_camera_recording: bool = Form(False),
    fps: Optional[int] = Form(None),
):
    try:
        video_name = file.filename if file else (os.path.basename(file_path) if file_path else "video.mp4")
        is_camera_rec = is_camera_recording or bool(video_name and video_name.startswith("recorded_camera"))
        if is_camera_rec:
            from app.ml import app_state
            if app_state.get_active_inference_kind() == "camera":
                try:
                    from app.services.oak_camera_service import oak_camera_service
                    await asyncio.to_thread(oak_camera_service.stop_inference)
                except Exception as e:
                    logger.warning(f"Failed to auto-stop camera inference on recording upload: {e}")

        # Determine target framerate from parameter or filename
        target_fps = fps
        if not target_fps and video_name:
            import re
            m = re.search(r"_(\d+)fps", video_name)
            if m:
                try:
                    target_fps = int(m.group(1))
                except Exception:
                    pass
        if not target_fps or target_fps <= 0:
            target_fps = 30

        # NEW: create_session() now raises RuntimeError while a training job
        # owns the GPU -- surface that as 409 instead of letting it 500.
        try:
            session_id = ml_inference_service.create_session(expected_ducks, original_filename=video_name)
        except RuntimeError as e:
            return JSONResponse(status_code=409, content={"message": str(e)})

        # Save uploaded video directly into its dedicated session folder under guaranteed writable output directory
        try:
            from app.core.app_paths import get_ml_output_dir
            base_output_dir = str(get_ml_output_dir())
        except Exception:
            base_output_dir = os.path.join(tempfile.gettempdir(), "vision_monitor_output")
        session_dir = os.path.join(base_output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        ext = os.path.splitext(video_name)[1] or ".mp4"
        is_direct_local = bool(file_path and os.path.exists(file_path) and os.path.isfile(file_path) and not is_camera_rec)

        if is_camera_rec:
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            from app.core.app_paths import get_desktop_dir
            desktop = get_desktop_dir()
            desktop_rec_dir = desktop / "recordings" / today
            desktop_rec_dir.mkdir(parents=True, exist_ok=True)
            raw_save_path = str(desktop_rec_dir / video_name)
            browser_video_path = raw_save_path
        elif is_direct_local:
            # DIRECT READING: When running in Desktop EXE / Electron, read directly from the user's hard drive!
            # Zero HTTP upload transfer and zero temporary file creation on disk.
            raw_save_path = os.path.abspath(file_path)
            browser_video_path = raw_save_path
            logger.info(f"[DESKTOP EXE] Direct reading video from disk with zero temp files: {raw_save_path}")
        else:
            # Web browser fallback: save uploaded video in system tempdir, auto-cleaned after inference
            raw_save_path = os.path.join(tempfile.gettempdir(), f"vision_upload_{session_id}{ext}")
            browser_video_path = os.path.join(tempfile.gettempdir(), f"vision_transcode_{session_id}.mp4")
        
        if not is_direct_local:
            if not file:
                return JSONResponse(status_code=400, content={"message": "No file uploaded or file path not found on disk."})
            try:
                content = await file.read()
                if len(content) < 5000:
                    logger.warning(f"[UPLOAD] Video file {video_name} is too small ({len(content)} bytes)")
                    return JSONResponse(
                        status_code=400,
                        content={"message": "Recorded video is empty or too short. Please record for at least 2-3 seconds."}
                    )
                with open(raw_save_path, "wb") as f:
                    f.write(content)
                if is_camera_rec:
                    logger.info(f"[RECORD] Saved recorded camera video directly to Desktop: {raw_save_path}")
            except Exception as e:
                logger.error(f"Error saving uploaded file: {e}")
                return JSONResponse(status_code=500, content={"message": "Failed to save uploaded video."})

        # 1. First probe if OpenCV can directly read the raw uploaded video (instantaneous, <0.02s)
        # Camera recordings (WebM/MP4 from MediaRecorder) lack container duration/frame-count headers in OpenCV,
        # so they MUST be transcoded with constant framerate to ensure exact frame counts and progress calculation.
        can_read_directly = False
        frame0_bytes = None
        frame_width = None
        frame_height = None
        is_webm = raw_save_path.lower().endswith(".webm")

        if not is_camera_rec and not is_webm:
            try:
                import cv2
                cap = cv2.VideoCapture(raw_save_path)
                if cap.isOpened():
                    total_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    ret, frame0 = cap.read()
                    if ret and frame0 is not None and total_cnt > 0:
                        can_read_directly = True
                        frame_width = int(frame0.shape[1])
                        frame_height = int(frame0.shape[0])
                        _, buf = cv2.imencode(".jpg", frame0, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame0_bytes = buf.tobytes()
                        logger.info(f"[UPLOAD] OpenCV directly read video ({frame_width}x{frame_height}, {total_cnt} frames), skipping heavy transcode.")
                cap.release()
            except Exception as e:
                logger.warning(f"Direct OpenCV probe failed: {e}")

        effective_inference_path = raw_save_path
        if can_read_directly:
            browser_video_path = raw_save_path

        # 2. Only if OpenCV CANNOT open the raw file directly or for camera recordings, fall back to ffmpeg transcode
        if not can_read_directly:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                if ffmpeg_exe and not os.path.exists(ffmpeg_exe):
                    logger.warning(f"ffmpeg_exe path {ffmpeg_exe} does not exist. Falling back to raw video.")
                    ffmpeg_exe = None
            except Exception as e:
                logger.warning(f"Could not locate ffmpeg: {e}")
                ffmpeg_exe = None

            if ffmpeg_exe:
                def run_transcode():
                    return subprocess.run(
                        [
                            ffmpeg_exe, "-y", "-i", raw_save_path,
                            "-vf", f"fps={target_fps}",
                            "-r", str(target_fps),
                            "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                            "-preset", "ultrafast",
                            "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-movflags", "+faststart",
                            browser_video_path,
                        ],
                        capture_output=True, text=True, timeout=180,
                    )

                try:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, run_transcode)
                    if result.returncode == 0 and os.path.exists(browser_video_path) and os.path.getsize(browser_video_path) > 1000:
                        effective_inference_path = browser_video_path
                        try:
                            import cv2
                            cap = cv2.VideoCapture(browser_video_path)
                            ret, frame0 = cap.read()
                            if ret and frame0 is not None:
                                frame_width = int(frame0.shape[1])
                                frame_height = int(frame0.shape[0])
                                _, buf = cv2.imencode(".jpg", frame0, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                frame0_bytes = buf.tobytes()
                            cap.release()
                        except Exception:
                            pass
                        logger.info(f"[UPLOAD] Transcoded fallback cleanly to H.264 MP4: {browser_video_path}")
                    else:
                        logger.warning(f"ffmpeg transcode failed; using raw upload: {result.stderr}")
                except Exception as e:
                    logger.warning(f"ffmpeg execution failed; using raw upload: {e}")

        # Save path in session state so it's ready to start when user commands
        session = ml_inference_service.sessions.get(session_id)
        if session:
            session["inference_video_path"] = effective_inference_path
            session["browser_video_path"] = browser_video_path
            if not is_camera_rec and not is_direct_local:
                temp_files = [raw_save_path]
                if effective_inference_path != raw_save_path:
                    temp_files.append(effective_inference_path)
                session["temp_files_to_cleanup"] = temp_files
            session["status"] = "ready"
            session["stats"]["status"] = "ready"
            if frame0_bytes:
                session["last_frame_bytes"] = frame0_bytes
                try:
                    with open(os.path.join(session_dir, "last_frame.jpg"), "wb") as f:
                        f.write(frame0_bytes)
                except Exception:
                    pass
            if frame_width and frame_height:
                session["stats"]["video_width"] = frame_width
                session["stats"]["video_height"] = frame_height
        
        return {
            "session_id": session_id,
            "status": "ready",
            "is_camera_recording": is_camera_rec,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /video/upload: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Video upload failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to process video upload: {e}")

from fastapi.responses import FileResponse

@router.get("/raw/{session_id}")
async def get_raw_video(session_id: str):
    try:
        session = ml_inference_service.sessions.get(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"message": "Session not found."})
        video_save_path = session.get("browser_video_path") or session.get("inference_video_path")
        if not video_save_path or not os.path.exists(video_save_path):
            return JSONResponse(status_code=404, content={"message": "Raw video file not found."})
        return FileResponse(video_save_path, media_type="video/mp4")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /video/raw/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Get raw video failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to get raw video: {e}")

def _ensure_session_exists(session_id: str):
    session = ml_inference_service.sessions.get(session_id)
    if session:
        return session

    try:
        from app.core.app_paths import get_ml_output_dir, get_desktop_dir
        base_output_dir = str(get_ml_output_dir())
    except Exception:
        base_output_dir = os.path.join(tempfile.gettempdir(), "vision_monitor_output")
    session_dir = os.path.join(base_output_dir, session_id)
    if not os.path.isdir(session_dir):
        try:
            desktop = get_desktop_dir()
            archive_base = desktop / "inference_results"
            if archive_base.exists():
                for date_folder in sorted(archive_base.iterdir(), reverse=True):
                    cand = date_folder / session_id
                    if cand.is_dir():
                        session_dir = str(cand)
                        break
        except Exception:
            pass
    if os.path.isdir(session_dir):
        candidates = []
        for f in os.listdir(session_dir):
            if f.lower().startswith("annotated_"):
                continue
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
                candidates.append(os.path.join(session_dir, f))
        
        candidates.sort(key=lambda p: (0 if "raw_upload" in os.path.basename(p) else 1 if "source" in os.path.basename(p) else 2))
        
        if candidates and os.path.exists(candidates[0]):
            video_file = candidates[0]
            orig_fn = os.path.basename(video_file)
            expected_ducks = 18
            results_file = os.path.join(session_dir, "results.json")
            if os.path.exists(results_file):
                try:
                    import json
                    with open(results_file, "r") as rf:
                        old_stats = json.load(rf)
                        orig_fn = old_stats.get("original_filename") or orig_fn
                        expected_ducks = int(old_stats.get("expected_duck_count", 18))
                except Exception:
                    pass

            ml_inference_service.create_session(expected_ducks=expected_ducks, original_filename=orig_fn, session_id=session_id)
            session = ml_inference_service.sessions.get(session_id)
            if session:
                session["inference_video_path"] = video_file
                session["browser_video_path"] = video_file
                session["temp_file"] = video_file
                session["status"] = "ready"
                session["stats"]["status"] = "ready"
                last_frame_path = os.path.join(session_dir, "last_frame.jpg")
                if os.path.exists(last_frame_path):
                    try:
                        with open(last_frame_path, "rb") as lf:
                            session["last_frame_bytes"] = lf.read()
                    except Exception:
                        pass
                logger.info(f"Restored session {session_id} from disk: {video_file}")
                return session

    return None

@router.post("/start/{session_id}")
async def start_video_inference(session_id: str, background_tasks: BackgroundTasks):
    try:
        session = _ensure_session_exists(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"message": "Session not found."})
        
        video_save_path = session.get("inference_video_path")
        if not video_save_path or not os.path.exists(video_save_path):
            return JSONResponse(status_code=400, content={"message": "Video file not found for session."})
        
        # Always create a new run. A Stop request is asynchronous, so checking
        # only `status != processing` used to let a quick Stop -> Start silently
        # keep the old run alive. start_run invalidates that old task and clears
        # its queued MJPEG frames before the new task begins.
        run_seq = ml_inference_service.start_run(session_id)
        background_tasks.add_task(
            ml_inference_service.process_video_task,
            session_id,
            video_save_path,
            session.get("original_filename"),
            run_seq,
        )
            
        return {"status": "started", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /video/start/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Start video inference failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to start video inference: {e}")

@router.get("/stream/{session_id}")
async def stream_video(session_id: str, request: Request):
    try:
        _ensure_session_exists(session_id)
        status = ml_inference_service.get_status(session_id)
        if not status:
            return JSONResponse(status_code=404, content={"message": "Session not found."})

        return StreamingResponse(
            ml_inference_service.get_stream_generator(session_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "close",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /video/stream/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Video streaming failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {e}")

@router.get("/status/{session_id}")
async def get_status(session_id: str):
    try:
        _ensure_session_exists(session_id)
        status = ml_inference_service.get_status(session_id)
        if not status:
            return JSONResponse(status_code=404, content={"message": "Session not found."})
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /video/status/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Get video status failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to get video status: {e}")

@router.post("/stop/{session_id}")
async def stop_video(session_id: str):
    try:
        status = ml_inference_service.get_status(session_id)
        if not status:
            return JSONResponse(status_code=404, content={"message": "Session not found."})
        
        ml_inference_service.stop_session(session_id)
        return {"message": "Stop signal sent."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /video/stop/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Stop video inference failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to stop video inference: {e}")

@router.get("/last_frame/{session_id}")
async def get_last_frame(session_id: str):
    try:
        session = _ensure_session_exists(session_id)
        frame_bytes = session.get("last_frame_bytes") if session else None
        
        if not frame_bytes:
            try:
                from app.core.app_paths import get_ml_output_dir
                base_output_dir = str(get_ml_output_dir())
            except Exception:
                base_output_dir = os.path.join(tempfile.gettempdir(), "vision_monitor_output")
            session_dir = os.path.join(base_output_dir, session_id)
            
            # 1. Check last_frame.jpg
            last_frame_file = os.path.join(session_dir, "last_frame.jpg")
            if os.path.exists(last_frame_file) and os.path.getsize(last_frame_file) > 0:
                try:
                    with open(last_frame_file, "rb") as f:
                        frame_bytes = f.read()
                except Exception:
                    pass

            # 2. Check anomaly_frames
            if not frame_bytes:
                anomaly_dir = os.path.join(session_dir, "anomaly_frames")
                if os.path.exists(anomaly_dir):
                    afiles = sorted(os.listdir(anomaly_dir))
                    if afiles:
                        try:
                            with open(os.path.join(anomaly_dir, afiles[-1]), "rb") as f:
                                frame_bytes = f.read()
                        except Exception:
                            pass

            # 3. Check raw_frames
            if not frame_bytes:
                raw_frames_dir = os.path.join(session_dir, "raw_frames")
                if os.path.exists(raw_frames_dir):
                    frames = sorted(os.listdir(raw_frames_dir))
                    if frames:
                        try:
                            with open(os.path.join(raw_frames_dir, frames[-1]), "rb") as f:
                                frame_bytes = f.read()
                        except Exception:
                            pass

            # 4. Check video files on disk or in Desktop archive
            if not frame_bytes:
                potential_videos = []
                if session:
                    for k in ("inference_video_path", "browser_video_path"):
                        p = session.get(k)
                        if p and os.path.exists(p):
                            potential_videos.append(p)
                if os.path.exists(session_dir):
                    for f in os.listdir(session_dir):
                        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                            potential_videos.append(os.path.join(session_dir, f))
                try:
                    from app.core.app_paths import get_desktop_dir
                    desktop = get_desktop_dir()
                    archive_base = os.path.join(str(desktop), "inference_results")
                    if os.path.exists(archive_base):
                        for today in os.listdir(archive_base):
                            target = os.path.join(archive_base, today, session_id)
                            if os.path.exists(target):
                                for fn in os.listdir(target):
                                    if fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                                        potential_videos.append(os.path.join(target, fn))
                except Exception:
                    pass

                for video_path in potential_videos:
                    try:
                        import cv2
                        cap = cv2.VideoCapture(video_path)
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        if total > 1:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 2))
                        ret, frame0 = cap.read()
                        if not ret or frame0 is None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame0 = cap.read()
                        cap.release()
                        if ret and frame0 is not None:
                            _, buf = cv2.imencode(".jpg", frame0, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            frame_bytes = buf.tobytes()
                            if session:
                                session["last_frame_bytes"] = frame_bytes
                            break
                    except Exception as e:
                        logger.warning(f"Failed to extract frame from {video_path}: {e}")

        if not frame_bytes:
            return JSONResponse(status_code=404, content={"message": "No frame available."})
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] GET /video/last_frame/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Get last frame failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to get last frame: {e}")

from pydantic import BaseModel

class ExpectedCountUpdate(BaseModel):
    count: int

@router.post("/update_expected/{session_id}")
async def update_expected(session_id: str, payload: ExpectedCountUpdate):
    try:
        status = ml_inference_service.get_status(session_id)
        if not status:
            return JSONResponse(status_code=404, content={"message": "Session not found."})
        
        ml_inference_service.update_expected_ducks(session_id, payload.count)
        return {"message": "Expected duck count updated."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /video/update_expected/{session_id}: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Update expected count failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to update expected count: {e}")


class StartPathInferenceRequest(BaseModel):
    video_path: str
    expected_ducks: int = 18

@router.post("/inference/path")
async def start_path_inference(data: StartPathInferenceRequest, background_tasks: BackgroundTasks):
    try:
        video_path = os.path.abspath(data.video_path.strip('"').strip("'"))
        if not os.path.exists(video_path) or not os.path.isfile(video_path):
            return JSONResponse(status_code=400, content={"message": f"File does not exist: {video_path}"})
        
        ext = os.path.splitext(video_path)[1].lower()
        valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        if ext not in valid_exts:
            return JSONResponse(status_code=400, content={"message": f"Unsupported video extension: {ext}"})

        filename = os.path.basename(video_path)
        try:
            session_id = ml_inference_service.create_session(data.expected_ducks, original_filename=filename)
        except RuntimeError as e:
            return JSONResponse(status_code=409, content={"message": str(e)})

        session = ml_inference_service.sessions.get(session_id)
        if session:
            session["inference_video_path"] = video_path
            session["status"] = "ready"
            session["stats"]["status"] = "ready"
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                ret, frame0 = cap.read()
                if ret and frame0 is not None:
                    _, buf = cv2.imencode(".jpg", frame0, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    session["last_frame_bytes"] = buf.tobytes()
                    session["stats"]["video_width"] = int(frame0.shape[1])
                    session["stats"]["video_height"] = int(frame0.shape[0])
                cap.release()
            except Exception as e:
                logger.warning(f"Could not pre-extract first frame from {video_path}: {e}")

        return {"session_id": session_id, "status": "ready", "video_name": filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] POST /video/inference/path: {e}", exc_info=True)
        realtime_log_service.add_log("video", "CRASH", f"Start path inference failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to start path inference: {e}")


