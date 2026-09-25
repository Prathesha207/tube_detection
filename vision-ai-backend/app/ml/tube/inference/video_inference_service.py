"""
video_inference_service.py
==========================
Handles batch / offline video file inference for Tube tracing.
"""

import tempfile
import asyncio
import os
import time
import uuid
import logging
from typing import Dict, Any, Optional
import cv2
import concurrent.futures

try:
    import torch
except Exception:
    torch = None
from app.ml import app_state
from app.core.logger import setup_logger
from .tube_analyzer import TubeAnalyzer

logger = setup_logger("tube-video-inference")

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "model", "best.pt"
)

import threading

class VideoInferenceService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()
        self._gpu_lock = asyncio.Lock()
        self._ml_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _get_or_create_analyzer(self) -> TubeAnalyzer:
        shared = getattr(self, "_shared_analyzer", None)
        if shared is not None:
            return shared

        path = _DEFAULT_MODEL_PATH
        local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "best.pt")
        if not os.path.exists(path):
            if os.path.exists(local_path):
                path = local_path
            else:
                raise FileNotFoundError(f"Model not found at {path} or {local_path}")
        
        logger.info("Initializing TubeAnalyzer for video session...")
        is_cuda = bool(torch and torch.cuda.is_available())
        analyzer = TubeAnalyzer(model_path=path, device="0" if is_cuda else "cpu")
        self._shared_analyzer = analyzer
        return analyzer

    def stop_all_sessions(self):
        with self._sessions_lock:
            sessions_list = list(self.sessions.values())
        for session in sessions_list:
            if not session["stop_event"].is_set():
                session["stop_event"].set()

    def delete_session(self, session_id: str):
        with self._sessions_lock:
            session = self.sessions.pop(session_id, None)
        if session:
            self.clear_session_files(session_id)
            session["stop_event"].set()
            session["analyzer"] = None
            logger.info(f"Released video session {session_id}")

    def cleanup_stale_sessions(self, max_idle_seconds: int = 3600):
        now = time.time()
        with self._sessions_lock:
            stale = [
                sid for sid, s in list(self.sessions.items())
                if s.get("status") in ("completed", "stopped", "error")
                and (now - s.get("last_active", now)) > max_idle_seconds
            ]
        for sid in stale:
            logger.info(f"Evicting stale video session {sid}")
            self.delete_session(sid)

    def create_session(self, original_filename: Optional[str] = None, session_id: Optional[str] = None) -> str:
        if app_state.get_mode() == "TRAINING":
            raise RuntimeError("Training is in progress -- please wait for it to finish before starting inference")
        if app_state.get_active_inference_kind() == "camera":
            raise RuntimeError("Live camera inference is currently active -- please stop it before starting a video upload")

        self.cleanup_stale_sessions()
        self.stop_all_sessions()
        
        session_id = session_id or str(uuid.uuid4())
        session_data = {
            "status": "queued",
            "queue": asyncio.Queue(maxsize=2),
            "original_filename": original_filename,
            "analyzer": None,
            "temp_files_to_cleanup": [],
            "created_at": time.time(),
            "last_active": time.time(),
            "stats": {
                "session_id": session_id,
                "original_filename": original_filename,
                "status": "queued",
                "frames_processed": 0,
                "total_frames": 0,
                "progress": 0.0,
                "fps": 0.0,
                "heads_count": 0,
                "tails_count": 0,
                "detections": [],
                "bigger_tube": None,
                "smaller_tube": None,
                "video_width": 0,
                "video_height": 0,
            },
            "stop_event": asyncio.Event(),
            "run_seq": 0,
            "temp_file": None
        }
        with self._sessions_lock:
            self.sessions[session_id] = session_data
        return session_id

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._sessions_lock:
            session = self.sessions.get(session_id)
        if not session:
            return None
        return session["stats"]

    async def get_stream_generator(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return

        if session.get("last_frame_bytes"):
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + session["last_frame_bytes"] + b"\r\n")

        queue = session["queue"]
        try:
            while not session["stop_event"].is_set():
                try:
                    frame_bytes = await asyncio.wait_for(queue.get(), timeout=1.5)
                except asyncio.TimeoutError:
                    if session["stop_event"].is_set() or session.get("status") in ("stopped", "completed", "error"):
                        break
                    continue
                if frame_bytes is None:
                    break
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        except asyncio.CancelledError:
            logger.info(f"Stream disconnected for session {session_id}")
        finally:
            logger.debug(f"Stream generator finished for session {session_id}")

    async def stop_session(self, session_id: str, timeout: float = 0.5):
        session = self.sessions.get(session_id)
        if session:
            session["stop_event"].set()
            session["status"] = "stopped"
            session["stats"]["status"] = "stopped"
            try:
                if session["queue"].full():
                    session["queue"].get_nowait()
                session["queue"].put_nowait(None)
            except Exception:
                pass
            start_wait = time.time()
            while session.get("is_task_active", False) and (time.time() - start_wait < timeout):
                await asyncio.sleep(0.01)
            # If task is still active after timeout, cancel it so it never hangs
            task = session.get("task")
            if task and not task.done() and session.get("is_task_active", False):
                task.cancel()
            session_dir = session.get("session_dir")
            if session_dir and session.get("last_frame_bytes"):
                try:
                    with open(os.path.join(session_dir, "last_frame.jpg"), "wb") as lf_out:
                        lf_out.write(session["last_frame_bytes"])
                except Exception:
                    pass

    def clear_session_files(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return
        for tf in list(session.get("temp_files_to_cleanup", [])):
            try:
                if tf and os.path.exists(tf):
                    os.remove(tf)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {tf}: {e}")
        session["temp_files_to_cleanup"] = []

    def start_run(self, session_id: str) -> int:
        session = self.sessions[session_id]
        session["run_seq"] += 1
        session["stop_event"].set()
        old_queue = session["queue"]
        try:
            if old_queue.full():
                old_queue.get_nowait()
            old_queue.put_nowait(None)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass
        session["queue"] = asyncio.Queue(maxsize=2)
        session["stop_event"] = asyncio.Event()
        session["status"] = "processing"
        
        session["stats"].update({
            "status": "processing",
            "frames_processed": 0,
            "progress": 0.0,
            "fps": 0.0,
            "heads_count": 0,
            "tails_count": 0,
            "detections": [],
            "bigger_tube": None,
            "smaller_tube": None,
        })
        return session["run_seq"]

    def _is_current_run(self, session_id: str, run_seq: int) -> bool:
        session = self.sessions.get(session_id)
        return bool(session and session.get("run_seq") == run_seq)

    async def process_video_task(self, session_id: str, temp_file_path: str, original_filename: Optional[str] = None, run_seq: Optional[int] = None):
        session = self.sessions.get(session_id)
        if not session or (run_seq is not None and not self._is_current_run(session_id, run_seq)):
            return

        if run_seq is None:
            run_seq = session.get("run_seq", 0)

        session["temp_file"] = temp_file_path
        if original_filename:
            session["original_filename"] = original_filename
            session["stats"]["original_filename"] = original_filename

        if not self._is_current_run(session_id, run_seq):
            return
        
        claimed_inference_lock = False

        async with self._gpu_lock:
            if not self._is_current_run(session_id, run_seq):
                return
                
            if app_state.get_mode() == "TRAINING":
                session["status"] = "error"
                session["stats"]["status"] = "error"
                session["stats"]["reasons"] = ["Training is in progress -- try again after it finishes"]
                return

            if session["stop_event"].is_set():
                session["status"] = "stopped"
                session["stats"]["status"] = "stopped"
                return

            if not app_state.try_enter_inference("video"):
                session["status"] = "error"
                session["stats"]["status"] = "error"
                session["stats"]["reasons"] = ["GPU is currently in use by another inference session (camera or training) -- try again shortly"]
                return
                
            claimed_inference_lock = True
            session["status"] = "processing"
            session["stats"]["status"] = "processing"
            session["is_task_active"] = True
            
            logger.info(f"Starting inference for session {session_id}")
            
            cap = None
            out_writer = None
            try:
                analyzer = self._get_or_create_analyzer()
                analyzer.reset()
                session["analyzer"] = analyzer

                cap = cv2.VideoCapture(temp_file_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video file: {temp_file_path}")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                session["stats"]["total_frames"] = max(1, total_frames)
                
                try:
                    from app.core.app_paths import get_ml_output_dir
                    base_output_dir = str(get_ml_output_dir())
                except Exception:
                    base_output_dir = os.path.join(tempfile.gettempdir(), "vision_monitor_output")
                    
                session_dir = os.path.join(base_output_dir, session_id)
                os.makedirs(session_dir, exist_ok=True)
                session["session_dir"] = session_dir

                orig_name = original_filename or session.get("original_filename")
                if orig_name:
                    base_name = os.path.splitext(os.path.basename(orig_name))[0]
                    video_filename = f"{base_name}.mp4" if base_name.startswith("annotated_") else f"annotated_{base_name}.mp4"
                else:
                    video_filename = f"annotated_{session_id}.mp4"

                output_path = os.path.join(session_dir, video_filename)
                session["stats"]["output_dir"] = session_dir
                session["stats"]["output_file"] = output_path

                raw_fps = cap.get(cv2.CAP_PROP_FPS)
                video_fps = raw_fps if (raw_fps and 10.0 <= raw_fps <= 120.0) else 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                session["stats"]["video_width"] = width
                session["stats"]["video_height"] = height
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
                
                frame_idx = 0
                stride = 2 if video_fps > 45.0 else 1
                last_result = {}
                last_annotated_frame = None
                infer_fps = 0.0
                loop = asyncio.get_running_loop()

                while not session["stop_event"].is_set() and self._is_current_run(session_id, run_seq):
                    ret, frame = cap.read()
                    if not self._is_current_run(session_id, run_seq) or session["stop_event"].is_set():
                        break

                    if not ret or frame is None:
                        logger.info(f"Session {session_id}: reached EOF at frame {frame_idx}")
                        session["status"] = "completed"
                        session["stats"]["status"] = "completed"
                        session["stats"]["progress"] = 100.0
                        session_dir = session.get("session_dir")
                        if session_dir and session.get("last_frame_bytes"):
                            try:
                                with open(os.path.join(session_dir, "last_frame.jpg"), "wb") as lf_out:
                                    lf_out.write(session["last_frame_bytes"])
                            except Exception as write_err:
                                logger.warning(f"Could not persist final last_frame.jpg: {write_err}")
                        break
                    
                    frame_idx += 1
                    session["last_active"] = time.time()
                    if frame.shape[1] > 0 and frame.shape[0] > 0:
                        session["stats"]["video_width"] = frame.shape[1]
                        session["stats"]["video_height"] = frame.shape[0]
                    session["stats"]["total_frames"] = max(session["stats"]["total_frames"], frame_idx)

                    if stride > 1 and (frame_idx % stride != 0) and last_annotated_frame is not None:
                        result = last_result
                        annotated_frame = frame
                    else:
                        def _infer_frame(analyzer_inst, frame_in):
                            if torch is not None:
                                with torch.inference_mode():
                                    return analyzer_inst.process_frame(frame_in, frame_idx)
                            return analyzer_inst.process_frame(frame_in, frame_idx)

                        try:
                            t_infer_start = time.perf_counter()
                            result, annotated_frame = await loop.run_in_executor(
                                self._ml_executor, _infer_frame, analyzer, frame.copy()
                            )
                            infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000
                            infer_fps = round(1000.0 / infer_latency_ms, 1) if infer_latency_ms > 0 else 0.0
                            last_result = result
                            last_annotated_frame = annotated_frame
                        except Exception as frame_err:
                            logger.warning(f"Session {session_id}: frame {frame_idx} inference failed: {frame_err}")
                            result = last_result
                            annotated_frame = frame.copy()
                            infer_fps = 0.0

                    session["stats"]["frames_processed"] = frame_idx
                    if session["stats"]["total_frames"] > 0:
                        session["stats"]["progress"] = round((frame_idx / session["stats"]["total_frames"]) * 100, 1)
                    session["stats"]["fps"] = infer_fps
                    
                    # Update stats with latest frame results
                    for k, v in result.items():
                        session["stats"][k] = v

                    if out_writer:
                        out_writer.write(annotated_frame)

                    try:
                        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        frame_bytes = buffer.tobytes()
                        session["last_frame_bytes"] = frame_bytes
                        try:
                            if session["queue"].full():
                                session["queue"].get_nowait()
                            session["queue"].put_nowait(frame_bytes)
                        except (asyncio.QueueFull, asyncio.QueueEmpty):
                            pass
                    except Exception as e:
                        logger.warning(f"Failed to encode frame for SSE in session {session_id}: {e}")

                    # Yield control briefly
                    await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in video inference task for session {session_id}: {e}", exc_info=True)
                session["status"] = "error"
                session["stats"]["status"] = "error"
                session["stats"]["reasons"] = [str(e)]
            finally:
                if cap:
                    cap.release()
                if out_writer:
                    out_writer.release()
                if claimed_inference_lock:
                    app_state.exit_inference("video")
                session_dir = session.get("session_dir")
                if session_dir and session.get("last_frame_bytes"):
                    try:
                        with open(os.path.join(session_dir, "last_frame.jpg"), "wb") as lf_out:
                            lf_out.write(session["last_frame_bytes"])
                    except Exception:
                        pass
                session["is_task_active"] = False
                try:
                    session["queue"].put_nowait(None)
                except Exception:
                    pass

video_inference_service = VideoInferenceService()
ml_inference_service = video_inference_service

def is_cuda_operational() -> bool:
    if torch is None:
        return False
    return torch.cuda.is_available() and torch.cuda.device_count() > 0

def _set_shared_analyzer(analyzer):
    video_inference_service._shared_analyzer = analyzer

