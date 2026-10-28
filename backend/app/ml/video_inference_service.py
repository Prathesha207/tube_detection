"""
video_inference_service.py
==========================
Batch / offline video-file inference for Tube tracing.

How a run works
---------------
    router:  session_id = create_session(...)
             run_seq    = start_run(session_id)
             await process_video_task(session_id, path, name, run_seq)

    process_video_task() (async) hands the whole job to ONE worker thread, so reading
    frames, running the model, writing the annotated mp4 and JPEG-encoding never block
    the FastAPI event loop. Per frame the worker does:

        rvf.open_video() -> cap.read() -> analyzer.infer(annotate=True)  <- the wheel's analyze_frame:
                                              detect, track, sizes, result, fps AND the drawn picture
                   -> annotated picture (drawn by the wheel for THIS frame) -> VideoWriter (annotated mp4)
                   -> result                      -> history.jsonl (the wheel's dict, one line per processed frame)
                   -> analyzer.get_clean_frame()  -> JPEG -> live stream queue
                   -> session["stats"]            (polled by get_status)

    Nothing here decides what a frame contains: counts, widths, Bigger/Smaller, fps, frame_total and
    video_fps are the wheel's. Frame skipping is the wheel's ``stride`` setting from config.yaml.

    One video job runs at a time (single worker thread). The model is loaded when a job
    starts and freed when it ends, so VRAM is empty whenever app_state says the GPU is free.
"""

import asyncio
import base64
import json
import os
import queue
import tempfile
import threading
import time
import uuid
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

try:
    import torch
except Exception:
    torch = None
from app.ml import app_state
from app.core.logger import setup_logger
from .tube_analyzer import (
    TubeAnalyzer,
    default_device,
    new_stats,
    release_gpu_memory,
    rvf,                      # the wheel's run_video_frames module (same one TubeAnalyzer uses)
)

logger = setup_logger("tube-video-inference")

_ML_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(_ML_DIR / "model" / "best.pt")
_CONFIG_PATH = str(_ML_DIR / "config" / "config.yaml")

_FINISHED_STATES = ("completed", "stopped", "error")
_JPEG_QUALITY = 85
_SAVE_PREVIEW_EVERY_N_FRAMES = 15


# ------------------------------------------------------------------- helpers
def _put_latest(queue: asyncio.Queue, item) -> None:
    """Put item in the queue, dropping the oldest entry when full. Must run on the event loop."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


def _json_default(obj: Any) -> Any:
    # 1. Converts single NumPy numbers (np.int64, np.float32) -> standard Python int or float
    if hasattr(obj, "item"):
        return obj.item()

    # 2. Converts NumPy arrays (np.ndarray bounding boxes or masks) -> standard Python list [...]
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # 3. Fallback for objects like Path or UUID -> converts to string
    return str(obj)



def _write_bytes(path: str, data: bytes) -> None:
    try:
        with open(path, "wb") as f:
            f.write(data)
    except Exception:
        pass


def _output_video_name(original_filename: Optional[str], session_id: str) -> str:
    if not original_filename:
        return f"annotated_{session_id}.mp4"
    base = os.path.splitext(os.path.basename(original_filename))[0]
    return f"{base}.mp4" if base.startswith("annotated_") else f"annotated_{base}.mp4"


def _output_base_dir() -> str:
    try:
        from app.core.app_paths import get_ml_output_dir
        return str(get_ml_output_dir())
    except Exception:
        return os.path.join(tempfile.gettempdir(), "vision_monitor_output")


class VideoInferenceService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()
        # ONE worker thread = one video job at a time, and the model is only ever used from it.
        self._ml_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-infer")
        self._shared_analyzer: Optional[TubeAnalyzer] = None   # only set via _set_shared_analyzer()

    # ------------------------------------------------------------- analyzer
    def _acquire_analyzer(self) -> Tuple[TubeAnalyzer, bool]:
        """(analyzer, owned). An analyzer injected with _set_shared_analyzer() is reused and never
        freed by us; otherwise a fresh one is created and the job frees it when it ends."""
        if self._shared_analyzer is None:
            try:
                from app.ml.camera_inference_service import _shared_camera_analyzer
                if _shared_camera_analyzer is not None:
                    self._shared_analyzer = _shared_camera_analyzer
            except Exception:
                pass

        if self._shared_analyzer is not None:
            if getattr(self._shared_analyzer, "has_weights_changed", lambda: False)():
                logger.info(f"[HOT-RELOAD] Detected updated weights at {self._shared_analyzer.model_path}. Reloading TubeAnalyzer for video...")
                try:
                    new_analyzer = TubeAnalyzer(
                        model_path=self._shared_analyzer.model_path,
                        config_path=_CONFIG_PATH,
                        device=default_device(),
                        draw_overlay=False,
                    )
                    self._shared_analyzer = new_analyzer
                    try:
                        from app.ml.camera_inference_service import _set_camera_analyzer
                        _set_camera_analyzer(new_analyzer)
                    except Exception:
                        pass
                    logger.info("[HOT-RELOAD] TubeAnalyzer successfully hot-reloaded for video.")
                except Exception as reload_err:
                    logger.error(f"[HOT-RELOAD] Failed to reload model weights for video: {reload_err}", exc_info=True)
            return self._shared_analyzer, False
        logger.info(f"Loading TubeAnalyzer for video job (model={_DEFAULT_MODEL_PATH})")
        analyzer = TubeAnalyzer(
            model_path=_DEFAULT_MODEL_PATH,
            config_path=_CONFIG_PATH,
            device=default_device(),
            draw_overlay=False,
        )
        self._shared_analyzer = analyzer
        try:
            from app.ml.camera_inference_service import _set_camera_analyzer
            _set_camera_analyzer(analyzer)
        except Exception:
            pass
        return analyzer, False

    def _get_or_create_analyzer(self) -> TubeAnalyzer:
        """Return an analyzer for one-off frame processing (e.g. frame 0 preview on upload)."""
        if self._shared_analyzer is not None:
            return self._shared_analyzer
        analyzer, _ = self._acquire_analyzer()
        return analyzer

    # ------------------------------------------------------------- sessions
    def stop_all_sessions(self):
        with self._sessions_lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            session["stop_event"].set()

    def delete_session(self, session_id: str):
        with self._sessions_lock:
            session = self.sessions.pop(session_id, None)
        if session:
            session["stop_event"].set()
            self._remove_temp_files(session)
            session["analyzer"] = None
            logger.info(f"Released video session {session_id}")

    def cleanup_stale_sessions(self, max_idle_seconds: int = 3600):
        now = time.time()
        with self._sessions_lock:
            stale = [sid for sid, s in self.sessions.items()
                     if s.get("status") in _FINISHED_STATES and now - s.get("last_active", now) > max_idle_seconds]
        for sid in stale:
            logger.info(f"Evicting stale video session {sid}")
            self.delete_session(sid)

    def create_session(self, original_filename: Optional[str] = None, session_id: Optional[str] = None) -> str:
        if app_state.get_mode() == "TRAINING":
            raise RuntimeError("Training is in progress -- please wait for it to finish before starting inference")
        if app_state.get_active_inference_kind() == "camera":
            raise RuntimeError("Live camera inference is currently active -- please stop it before starting a video upload")

        self.cleanup_stale_sessions()
        if session_id and session_id in self.sessions:
            return session_id
        self.stop_all_sessions()

        session_id = session_id or str(uuid.uuid4())
        session = {
            "status": "queued",
            "queue": asyncio.Queue(maxsize=2),
            "original_filename": original_filename,
            "analyzer": None,
            "temp_files_to_cleanup": [],
            "created_at": time.time(),
            "last_active": time.time(),
            "stats": new_stats(session_id=session_id, original_filename=original_filename,
                               total_frames=0, progress=0.0),
            "stop_event": threading.Event(),
            "run_seq": 0,
            "temp_file": None,
        }
        with self._sessions_lock:
            self.sessions[session_id] = session
        return session_id

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._sessions_lock:
            session = self.sessions.get(session_id)
        if not session:
            return None
        res = dict(session["stats"])
        # Frame-accurate synchronization: return pre-cached base64 frame (0ms event-loop cost)
        b64 = session.get("last_frame_b64")
        if b64:
            res["frame"] = b64
        elif session.get("last_frame_bytes"):
            b64 = "data:image/jpeg;base64," + base64.b64encode(session["last_frame_bytes"]).decode("ascii")
            session["last_frame_b64"] = b64
            res["frame"] = b64
        return res

    # --------------------------------------------------------------- stream
    async def get_stream_generator(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return

        if session.get("last_frame_bytes"):
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + session["last_frame_bytes"] + b"\r\n"

        queue = session["queue"]
        try:
            while not session["stop_event"].is_set():
                try:
                    frame_bytes = await asyncio.wait_for(queue.get(), timeout=1.5)
                except asyncio.TimeoutError:
                    if session["stop_event"].is_set() or session.get("status") in _FINISHED_STATES:
                        break
                    continue
                if frame_bytes is None:
                    break
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        except asyncio.CancelledError:
            logger.info(f"Stream disconnected for session {session_id}")
        finally:
            logger.debug(f"Stream generator finished for session {session_id}")

    async def stop_session(self, session_id: str, timeout: float = 1.0):
        session = self.sessions.get(session_id)
        if not session:
            return
        session["stop_event"].set()
        session["status"] = "stopped"
        session["stats"]["status"] = "stopped"
        session["last_frame_bytes"] = None
        session["last_frame_b64"] = None
        _put_latest(session["queue"], None)
        start_wait = time.time()
        while session.get("is_task_active", False) and time.time() - start_wait < timeout:
            await asyncio.sleep(0.01)

    # ---------------------------------------------------------------- files
    def _remove_temp_files(self, session: Dict[str, Any]) -> None:
        for path in list(session.get("temp_files_to_cleanup", [])):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {path}: {e}")
        session["temp_files_to_cleanup"] = []

    def clear_session_files(self, session_id: str):
        session = self.sessions.get(session_id)
        if session:
            self._remove_temp_files(session)

    # ----------------------------------------------------------------- runs
    def start_run(self, session_id: str) -> int:
        """Begin a new run of this session: stops the previous run, resets stats, new queue."""
        session = self.sessions[session_id]
        session["run_seq"] += 1
        session["stop_event"].set()                       # ends the previous run's loop
        _put_latest(session["queue"], None)               # ends the previous run's stream
        session["queue"] = asyncio.Queue(maxsize=2)
        session["stop_event"] = threading.Event()
        session["last_frame_bytes"] = None
        session["last_frame_b64"] = None
        # Preserve user ROI across runs; only clear ROI when user explicitly resets it
        session["status"] = "processing"
        session["stats"].update({
            **new_stats(),
            "status": "processing",
            "total_frames": session["stats"].get("total_frames", 0),
            "progress": 0.0,
            "video_width": session["stats"].get("video_width", 0),
            "video_height": session["stats"].get("video_height", 0),
        })
        return session["run_seq"]

    def _is_current_run(self, session_id: str, run_seq: int) -> bool:
        session = self.sessions.get(session_id)
        return bool(session and session.get("run_seq") == run_seq)

    async def process_video_task(self, session_id: str, temp_file_path: str,
                                 original_filename: Optional[str] = None, run_seq: Optional[int] = None):
        session = self.sessions.get(session_id)
        if not session or (run_seq is not None and not self._is_current_run(session_id, run_seq)):
            return
        if run_seq is None:
            run_seq = session.get("run_seq", 0)

        session["temp_file"] = temp_file_path
        if original_filename:
            session["original_filename"] = original_filename
            session["stats"]["original_filename"] = original_filename

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._ml_executor, self._run_video_job,
            session_id, run_seq, temp_file_path, original_filename, loop,
        )

    # ------------------------------------------------- the job (worker thread)
    def _run_video_job(self, session_id: str, run_seq: int, video_path: str,
                       original_filename: Optional[str], loop: asyncio.AbstractEventLoop) -> None:
        session = self.sessions.get(session_id)
        if session is None or not self._is_current_run(session_id, run_seq):
            return
        # Captured now: start_run() replaces these for the NEXT run, and this run must keep
        # using (and finishing) its own.
        stop_event = session["stop_event"]
        frame_queue = session["queue"]
        stats = session["stats"]

        def set_status(status: str, reasons=None) -> None:
            if not self._is_current_run(session_id, run_seq):
                return                                    # a newer run owns the session now
            session["status"] = status
            stats["status"] = status
            if reasons:
                stats["reasons"] = reasons

        def publish(item) -> None:
            try:
                loop.call_soon_threadsafe(_put_latest, frame_queue, item)
            except RuntimeError:
                pass                                      # event loop already closed

        if app_state.get_mode() == "TRAINING":
            set_status("error", ["Training is in progress -- try again after it finishes"])
            return
        if stop_event.is_set():
            set_status("stopped")
            return
        if not app_state.try_enter_inference("video"):
            set_status("error", ["GPU is currently in use by another inference session (camera or training) -- try again shortly"])
            return

        session["is_task_active"] = True
        set_status("processing")
        logger.info(f"Starting inference for session {session_id}")

        cap = writer = analyzer = last_stats = None
        owns_analyzer = False
        reached_eof = False
        try:
            analyzer, owns_analyzer = self._acquire_analyzer()   # model load happens here, off the event loop
            analyzer.reset_tracking()
            if hasattr(analyzer, "cfg"):
                analyzer.cfg.history_jsonl = None
            if session.get("roi_points"):
                try:
                    analyzer.set_roi(session["roi_points"], session.get("roi_frame_size"))
                    logger.info(f"Re-applied {len(session['roi_points'])}-point ROI to session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to re-apply ROI for session {session_id}: {e}")
            session["analyzer"] = analyzer

            # Open the video exactly like the wheel's own CLI does (rvf.open_video), so frames are decoded
            # the same way and fps / frame count come from the file, not from an assumption here.
            cap, video_fps, total = rvf.open_video(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            stride = max(int(getattr(analyzer.cfg, "stride", 1) or 1), 1)   # config.yaml (ML team), not a guess from fps
            stats.update(total_frames=max(1, int(total or 0)), video_width=width, video_height=height)
            # Same two values HeadTailAnalyzer.process_video gives the wheel: it then reports
            # frame_total / video_fps / fps itself and draws "FRAME n / total".
            analyzer.set_video_info(total, round(video_fps, 2))
            analyzer.warm_up((height, width))                    # so frame 1 is not slow

            session_dir = os.path.join(_output_base_dir(), session_id)
            os.makedirs(session_dir, exist_ok=True)
            output_path = os.path.join(session_dir, _output_video_name(
                original_filename or session.get("original_filename"), session_id))
            history_path = os.path.join(session_dir, "history.jsonl")
            session["session_dir"] = session_dir
            session["history_path"] = history_path
            stats["output_dir"] = session_dir
            stats["output_file"] = output_path
            stats["history_file"] = history_path
            # fps / stride: with stride > 1 only every Nth frame is written, same as the wheel's CLI
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps / stride, (width, height))
            if not writer.isOpened():
                logger.warning(f"Session {session_id}: cannot write {output_path}; continuing without annotated video")
                writer = None

            history_f = None
            try:
                history_f = open(history_path, "w", encoding="utf-8")
            except Exception as h_err:
                logger.warning(f"Session {session_id}: cannot open history file {history_path}: {h_err}")

            # Asynchronous video writer thread: only MP4 encoding happens there. The picture was already
            # drawn by the wheel inside analyze_frame, so it always matches that frame's result.
            writer_queue = queue.Queue(maxsize=32)
            writer_thread = None
            if writer is not None:
                def _writer_worker():
                    while True:
                        item = writer_queue.get()
                        if item is None:
                            break
                        try:
                            writer.write(item)
                        except Exception as w_err:
                            logger.warning(f"Session {session_id}: writer worker error: {w_err}")
                        finally:
                            writer_queue.task_done()

                writer_thread = threading.Thread(target=_writer_worker, daemon=True, name=f"writer-{session_id[:8]}")
                writer_thread.start()

            frame_number = 0
            processed = 0

            while not stop_event.is_set():
                ok, frame = cap.read()
                if not self._is_current_run(session_id, run_seq):
                    break
                if not ok or frame is None:
                    reached_eof = True
                    break

                frame_number += 1
                session["last_active"] = time.time()
                stats["total_frames"] = max(stats["total_frames"], frame_number)

                if frame_number % stride:
                    # Skipped by the wheel's stride setting: not analysed, not written, no history line.
                    stats["progress"] = round(frame_number / stats["total_frames"] * 100, 1)
                    continue

                try:
                    last_stats = analyzer.infer(frame, frame_number, annotate=writer is not None)   # <- the ONE ML call
                except Exception as frame_err:
                    raise RuntimeError(f"Inference failed on frame {frame_number}: {frame_err}") from frame_err
                analysis = analyzer._last_analysis
                processed += 1

                # Stats for the frontend (one update() so readers never see a half-written frame).
                # "status" is left out: set_status() owns it, a late frame must not undo "stopped".
                stats.update({k: v for k, v in last_stats.items() if k != "status"},
                             frame=frame_number,
                             frames_processed=processed,
                             progress=round(frame_number / stats["total_frames"] * 100, 1))

                # Saved video: the wheel's own drawing of THIS frame; never dropped (the queue blocks
                # when the encoder is behind, so the video has every processed frame).
                annotated = analysis.annotated if analysis is not None else None
                if writer is not None and annotated is not None:
                    if writer_thread is not None and writer_thread.is_alive():
                        while not stop_event.is_set():
                            try:
                                writer_queue.put(annotated, timeout=0.5)
                                break
                            except queue.Full:
                                continue
                    else:
                        writer.write(annotated)

                # history.jsonl = the wheel's result dict, untouched, one line per processed frame
                # (same content the CLI's --history writes), so two runs can be diffed line by line.
                if history_f is not None and analysis is not None and analysis.result:
                    try:
                        history_f.write(json.dumps(analysis.result, default=_json_default) + "\n")
                    except Exception as h_write_err:
                        logger.warning(f"Session {session_id}: history write error on frame {frame_number}: {h_write_err}")

                # Live stream: clean frame; the frontend draws boxes from stats["detections"].
                try:
                    clean_f = analyzer.get_clean_frame(frame)
                    h, w = clean_f.shape[:2]
                    if w > 1280:
                        new_h = int(h * (1280.0 / w))
                        preview_f = cv2.resize(clean_f, (1280, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        preview_f = clean_f
                    encoded, buffer = cv2.imencode(".jpg", preview_f,
                                                   [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
                    if encoded:
                        frame_bytes = buffer.tobytes()
                        session["last_frame_bytes"] = frame_bytes
                        session["last_frame_b64"] = "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode("ascii")
                        if frame_number == 1 or frame_number % _SAVE_PREVIEW_EVERY_N_FRAMES == 0:
                            _write_bytes(os.path.join(session_dir, "last_frame.jpg"), frame_bytes)
                        publish(frame_bytes)
                except Exception as enc_err:
                    logger.warning(f"Session {session_id}: could not encode frame {frame_number}: {enc_err}")

            # Flush and release the background video writer
            if writer_thread is not None:
                try:
                    while writer_thread.is_alive():          # let the encoder drain every queued frame
                        try:
                            writer_queue.put(None, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                    writer_thread.join(timeout=120.0)
                    if writer_thread.is_alive():
                        logger.warning(f"Session {session_id}: video encoder still busy after 120 s; output may be cut short")
                except Exception as flush_err:
                    logger.warning(f"Session {session_id}: writer flush warning: {flush_err}")
                writer_thread = None

            if history_f is not None:
                try:
                    history_f.flush()
                    history_f.close()
                except Exception as h_close_err:
                    logger.warning(f"Session {session_id}: history file close warning: {h_close_err}")
                history_f = None

            # Finalise the output file BEFORE reporting "completed", so a client that sees
            # "completed" can open a fully written mp4.
            cap.release(); cap = None
            if writer is not None:
                writer.release(); writer = None

            if reached_eof:
                logger.info(f"Session {session_id}: reached EOF at frame {frame_number}")
                stats["progress"] = 100.0
                if session.get("last_frame_bytes"):
                    _write_bytes(os.path.join(session_dir, "last_frame.jpg"), session["last_frame_bytes"])
                try:
                    with open(os.path.join(session_dir, "results.json"), "w") as rf:
                        json.dump({**stats, "status": "completed"}, rf, indent=2, default=_json_default)
                except Exception as json_err:
                    logger.warning(f"Could not save results.json: {json_err}")
                set_status("completed")
            elif session.get("status") == "processing":
                set_status("stopped")                         # stopped via stop_all_sessions / delete_session

        except Exception as e:
            logger.error(f"Error in video inference task for session {session_id}: {e}", exc_info=True)
            set_status("error", [str(e)])
        finally:
            if 'history_f' in locals() and history_f is not None:
                try:
                    history_f.close()
                except Exception:
                    pass
            if cap is not None:
                cap.release()
            if 'writer_thread' in locals() and writer_thread is not None and writer_thread.is_alive():
                try:
                    writer_queue.put_nowait(None)
                except Exception:
                    pass
            if writer is not None:
                writer.release()
            session["analyzer"] = None
            analyzer = last_stats = None                      # drop our references to the model
            if owns_analyzer:
                release_gpu_memory()                          # VRAM back BEFORE the claim is released
            app_state.exit_inference("video")
            session["is_task_active"] = False
            if self._is_current_run(session_id, run_seq):
                publish(None)                                 # ends this run's live stream


video_inference_service = VideoInferenceService()
ml_inference_service = video_inference_service


def is_cuda_operational() -> bool:
    from .tube_analyzer import cuda_is_available
    return cuda_is_available()


def set_video_roi(session_id: Optional[str] = None, points: Any = None, frame_size: Optional[Tuple[int, int]] = None) -> None:
    """Hot-swap active video session analyzer in-memory (per-session ROI, no disk persistence)."""
    with video_inference_service._sessions_lock:
        if session_id and session_id in video_inference_service.sessions:
            targets = [video_inference_service.sessions[session_id]]
        else:
            targets = list(video_inference_service.sessions.values())
        for s in targets:
            s["roi_points"] = points or []
            s["roi_frame_size"] = frame_size
            analyzer = s.get("analyzer")
            if isinstance(analyzer, TubeAnalyzer):
                analyzer.set_roi(points, frame_size)


def _set_shared_analyzer(analyzer):
    """Optional: pre-loaded analyzer that video jobs reuse instead of loading one per job.
    NOTE: it stays in VRAM permanently, which defeats the video/camera exclusion in app_state."""
    video_inference_service._shared_analyzer = analyzer
