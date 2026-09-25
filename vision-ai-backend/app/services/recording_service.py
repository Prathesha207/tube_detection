"""
Recording service — uses PyAV for per-frame timestamps.

Why PyAV instead of cv2.VideoWriter:
  cv2.VideoWriter stamps every frame at (frame_number / declared_fps).
  When the OAK camera delivers 3.2 fps instead of the configured 10 fps
  (auto-exposure in low light), the result is a 19-second video for a
  60-second recording.

  PyAV records each frame with the wall-clock time it actually arrived,
  so the saved video is always the correct duration regardless of the
  camera's delivery rate.  No fps parameter needed at recording start,
  no post-recording patching, no ffmpeg subprocess.
"""

import av
import cv2
import os
import queue
import threading
import time
import logging
from datetime import datetime
from app.core.app_paths import get_desktop_dir

logger = logging.getLogger("recording-service")

# session_id → RecordingSession
active_recordings: dict = {}


class RecordingSession:
    def __init__(
        self,
        session_id: str,
        width: int,
        height: int,
        fps: float = 30.0,           # kept for API compatibility; not used for timestamps
        root_path: str = None,
        recording_format: str = "MP4",
    ):
        now = datetime.now()
        if not root_path:
            desktop = get_desktop_dir()
            root_path = str(desktop / "recordings")
        folder = os.path.join(root_path, now.strftime("%Y-%m-%d"))
        os.makedirs(folder, exist_ok=True)

        fmt = (recording_format or "MP4").upper()

        options = {}
        if fmt in ("AVI", "MJPEG"):
            fmt        = "AVI"
            codec_name = "mjpeg"
            pix_fmt    = "yuvj420p"
            ext        = ".avi"
        elif fmt in ("FFV1", "MKV"):
            fmt        = "FFV1"
            codec_name = "ffv1"
            pix_fmt    = "yuv420p"
            ext        = ".mkv"
            options    = {"level": "3"}
        else:  # Default: MP4 (H.264)
            fmt        = "MP4"
            codec_name = "libx264"
            pix_fmt    = "yuv420p"
            ext        = ".mp4"
            options    = {"preset": "veryfast", "crf": "16"}

        self.filename = now.strftime(f"session_%Y-%m-%d_%H-%M-%S{ext}")
        self.video_path = os.path.join(folder, self.filename)
        self.width  = width
        self.height = height
        # BUGFIX: `fmt` was only ever a local variable in __init__ -- stop()
        # reads self.fmt when building its return dict, which raised
        # AttributeError on every single recording stop (this was thrown
        # after the file was already correctly flushed/closed by the worker
        # thread, so the .mp4/.avi/.mkv on disk was actually fine -- but the
        # exception then propagated out of RecordingSession.stop() ->
        # recording_service.stop_recording() -> oak_camera_service's
        # stop_recording()/stop(), which has no try/except around this call.
        # When stop() is the camera-shutdown path, that exception aborted
        # shutdown BEFORE _close_stream_queue()/_stop_capture_threads()/
        # _cleanup_pipeline()/disconnect() could run -- leaving the pipeline
        # and device connected, which is exactly the kind of state that makes
        # the next camera start fail or the app look wedged.
        self.fmt = fmt
        # Initialize all state stop()/add_frame() touch BEFORE attempting to open
        # the container, so a failed open leaves a fully-formed (just inert)
        # object instead of a half-built one that crashes on the first stop().
        self.frame_queue = queue.Queue(maxsize=1000)
        self.is_running  = False
        # Set start clock at session creation — not on first frame — so the
        # container duration matches the wall-clock recording time exactly.
        self._start_mono: float = time.monotonic()
        self._frames_written = 0
        self._last_bgr = None       # last frame received, used for pad-frame on stop
        self._last_pts_ms: int = 0  # PTS of that frame
        self.thread = None

        try:
            self.container = av.open(self.video_path, mode="w")
            # rate=1000 → time_base = 1/1000 s = 1 ms per PTS unit.
            # VFR timestamps allow precise wall-clock synchronization.
            self.stream = self.container.add_stream(codec_name, rate=1000)
            self.stream.width   = width
            self.stream.height  = height
            self.stream.pix_fmt = pix_fmt
            if options:
                self.stream.options = options
        except Exception as e:
            logger.error(f"[RECORD] Failed to open container: {e}")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info(f"[RECORD] Started: {self.video_path} | format={fmt} ({codec_name}) | {width}x{height}")

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker(self):
        try:
            while self.is_running or not self.frame_queue.empty():
                try:
                    bgr, pts_ms = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                try:
                    # BGR (OpenCV) → RGB → target pixel format
                    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame = frame.reformat(format=self.stream.pix_fmt)

                    # pts_ms is milliseconds since recording start.
                    # With time_base = 1/1000, one PTS unit = 1 ms.
                    frame.pts = pts_ms

                    for pkt in self.stream.encode(frame):
                        self.container.mux(pkt)

                    self._frames_written += 1
                    if self._frames_written % 100 == 0:
                        logger.info(f"[RECORD] Frames written: {self._frames_written}")

                except Exception as e:
                    logger.warning(f"[RECORD] Frame encode error: {e}")

        except Exception as e:
            logger.error(f"[RECORD] Worker error: {e}")
        finally:
            # Flush encoder and close container inside the worker so that
            # stop() can join() and know the file is fully written.
            try:
                for pkt in self.stream.encode():
                    self.container.mux(pkt)
                self.container.close()
                logger.info(
                    f"[RECORD] Closed — {self._frames_written} frames → {self.video_path}"
                )
            except Exception as e:
                logger.error(f"[RECORD] Container close error: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def add_frame(self, bgr):
        if not self.is_running:
            return

        pts_ms = int((time.monotonic() - self._start_mono) * 1000)

        self._last_bgr = bgr
        self._last_pts_ms = pts_ms

        try:
            self.frame_queue.put_nowait((bgr, pts_ms))
        except queue.Full:
            # Drop oldest frame to make room for the newest
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((bgr, pts_ms))
            except Exception:
                pass

    def stop(self) -> dict:
        # Compute the actual wall-clock recording duration right now.
        # The last real frame arrived some time ago (depends on camera fps).
        # Adding a duplicate of the last frame at "now" pads the container so
        # the video duration matches the real recording time exactly.
        if self._last_bgr is not None:
            pad_pts_ms = int((time.monotonic() - self._start_mono) * 1000)
            gap_ms = pad_pts_ms - self._last_pts_ms
            logger.info(
                f"[RECORD] Padding video by {gap_ms} ms "
                f"(last frame at {self._last_pts_ms} ms → pad to {pad_pts_ms} ms)"
            )
            try:
                self.frame_queue.put_nowait((self._last_bgr, pad_pts_ms))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((self._last_bgr, pad_pts_ms))
                except Exception:
                    pass

        logger.info(
            f"[RECORD] Stopping — {self.frame_queue.qsize()} frames still in queue"
        )
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=30.0)
            if self.thread.is_alive():
                logger.warning("[RECORD] Worker thread did not finish within 30 s")
        else:
            logger.warning(
                f"[RECORD] stop() called on a session whose container never "
                f"opened successfully: {self.video_path}"
            )
        # Container is already closed by the worker's finally block
        duration = time.monotonic() - self._start_mono
        return {
            "recording_path": self.video_path,
            "filename": self.filename,
            "duration": round(duration, 2),
            "frames_written": self._frames_written,
            "format": self.fmt,
            "width": self.width,
            "height": self.height,
        }


def _get_default_recording_path() -> str:
    desktop = get_desktop_dir()
    return str(desktop / "recordings")


def _resolve_recording_path(root_path: str = None) -> str:
    if root_path:
        return root_path
    try:
        from app.core.database import SessionLocal
        from app.models.camera_model import Camera
        

        db = SessionLocal()
        try:
            camera    = db.query(Camera).first()
            inference = None
            mode = getattr(inference, "mode", None) or "production"
            path = (
                getattr(camera, "recording_video_testing_path", None)
                if mode == "testing"
                else None
            ) or getattr(camera, "recording_video_path", None)
            default_path = _get_default_recording_path()
            return path or default_path
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"[RECORD] Could not resolve path from DB: {e}")
        return _get_default_recording_path()


def start_recording(
    session_id: str,
    width: int,
    height: int,
    fps: float = 30.0,
    root_path: str = None,
    recording_format: str = "MP4",
) -> str:
    logger.info(f"[RECORD] Creating session: {session_id} | format={recording_format}")
    if session_id in active_recordings:
        logger.warning("[RECORD] Session already exists — returning existing path")
        return active_recordings[session_id].video_path

    root_path = _resolve_recording_path(root_path)
    session = RecordingSession(session_id, width, height, fps, root_path, recording_format)

    # BUGFIX: if av.open() failed inside RecordingSession.__init__, it logs
    # and returns early, leaving is_running=False and thread=None -- but this
    # function used to store that broken session into active_recordings and
    # return session.video_path unconditionally, exactly as if recording had
    # started. The caller (oak_camera_service.start_recording) then logs
    # "[RECORD] Session active" and happily feeds it frames for the entire
    # session -- every one of them silently dropped by add_frame()'s
    # `if not self.is_running: return` guard -- producing a recording button
    # that appears to work but writes nothing, with no error anywhere.
    if not session.is_running:
        raise RuntimeError(
            f"Recording container failed to open for session {session_id} "
            "(see the '[RECORD] Failed to open container' log line above for "
            "the underlying cause) -- refusing to report a successful start "
            "for a session that would silently record zero frames."
        )

    active_recordings[session_id] = session
    return session.video_path


def write_frame(session_id: str, frame) -> None:
    session = active_recordings.get(session_id)
    if session:
        session.add_frame(frame)


def stop_recording(session_id: str) -> dict:
    session = active_recordings.pop(session_id, None)
    if not session:
        logger.warning(f"[RECORD] No active session: {session_id}")
        return {}
    return session.stop()