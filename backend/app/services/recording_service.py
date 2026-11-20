"""
Recording service — uses PyAV for per-frame hardware/wall-clock timestamps.

Key design goals (aligned with dualcam_monitor / vision-monitor):
  - Zero-drop recording queue (unbounded) so encoder stalls never silently evict frames.
  - Hardware H.264 acceleration (h264_nvenc) when available, with low-latency
    libx264 (ultrafast + zerolatency) fallback for MP4.
  - Direct raw sensor NV12 -> yuv420p encoding with multi-slice Golomb-Rice FFV1
    (coder=0, slices=8, SLICE threading) to bypass CPU NV12->BGR->RGB conversion.
  - Persistent per-session VideoReformatter using bgr24 directly (eliminates
    redundant cv2.cvtColor BGR->RGB full-frame copies).
  - Explicit Fraction(1, 1000) time_base on stream, codec_context, and frames,
    plus strict monotonic PTS/DTS enforcement to prevent choppy playback or
    muxer drops.
"""

import av
from av.video.reformatter import (
    VideoReformatter,
    ColorRange,
    Colorspace,
    ColorPrimaries,
    ColorTrc,
)
import numpy as np
import os
import queue
import threading
import time
import logging
from fractions import Fraction
from datetime import datetime
from typing import Any
from app.core.app_paths import get_desktop_dir

logger = logging.getLogger("recording-service")

# session_id → RecordingSession
active_recordings: dict = {}

_NVENC_FUNCTIONAL: bool | None = None
_NVENC_LOCK = threading.Lock()


def _is_nvenc_functional() -> bool:
    """Probe once whether NVIDIA h264_nvenc is genuinely usable on this machine."""
    global _NVENC_FUNCTIONAL
    if _NVENC_FUNCTIONAL is not None:
        return _NVENC_FUNCTIONAL
    with _NVENC_LOCK:
        if _NVENC_FUNCTIONAL is not None:
            return _NVENC_FUNCTIONAL
        if "h264_nvenc" not in av.codec.codecs_available:
            _NVENC_FUNCTIONAL = False
            return False
        try:
            devnull = "nul" if os.name == "nt" else "/dev/null"
            c: Any = av.open(devnull, mode="w", format="mp4")
            s: Any = c.add_stream("h264_nvenc", rate=30)
            s.width = 64
            s.height = 64
            s.pix_fmt = "yuv420p"
            s.options = {"preset": "fast"}
            f = av.VideoFrame(64, 64, "yuv420p")
            s.encode(f)
            c.close()
            _NVENC_FUNCTIONAL = True
            logger.info("[RECORD] Hardware acceleration verified: h264_nvenc is functional")
            return True
        except Exception as e:
            logger.info(f"[RECORD] NVENC not functional ({e}) — using low-latency libx264")
            _NVENC_FUNCTIONAL = False
            return False


class RecordingSession:
    def __init__(
        self,
        session_id: str,
        width: int,
        height: int,
        fps: float = 30.0,
        root_path: str | None = None,
        recording_format: str = "MP4",
        camera_settings: dict | None = None,
        **kwargs: Any,
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
            options    = {"qmin": "1", "qmax": "2"}
        elif fmt in ("FFV1", "MKV"):
            fmt        = "FFV1"
            codec_name = "ffv1"
            pix_fmt    = "yuv420p"
            ext        = ".mkv"
            options    = {"level": "3", "coder": "0", "slices": "8", "slicecrc": "0"}
        else:  # Default: MP4 (H.264)
            fmt        = "MP4"
            pix_fmt    = "yuv420p"
            ext        = ".mp4"
            if _is_nvenc_functional():
                codec_name = "h264_nvenc"
                options    = {"preset": "fast", "cq": "18"}
            else:
                codec_name = "libx264"
                options    = {"preset": "ultrafast", "tune": "zerolatency", "crf": "19"}

        self.session_id = session_id
        self.filename = now.strftime(f"session_%Y-%m-%d_%H-%M-%S{ext}")
        self.video_path = os.path.join(folder, self.filename)
        self.width  = width
        self.height = height
        self.fps    = float(fps or 30.0)
        self.fmt = fmt
        self.codec_name = codec_name
        self.pix_fmt = pix_fmt
        self.codec_options = options
        self.is_raw_nv12 = (fmt == "FFV1")
        self._saw_nv12 = False

        self.camera_settings = camera_settings or {}
        self.settings_history = [{"elapsed_seconds": 0.0, "settings": self.camera_settings.copy()}]

        # Per-session reformatter avoids repeated SwsContext creation and BGR->RGB copies
        self._reformatter = VideoReformatter()

        # Unbounded queue preserves every captured camera frame while encoder drains.
        # Live preview runs on a separate path and never waits on this queue.
        self.frame_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._start_mono: float = time.monotonic()
        self._first_cam_ts_sec: float | None = None
        self._first_cam_offset_ms: int = 0
        self._frames_written: int = 0
        self._dropped_frames: int = 0
        self._camera_frames_missing: int = 0
        self._last_seq_num: int | None = None
        self._last_backlog_log: float = 0.0
        self._last_frame_payload: Any = None
        self._last_pts_ms: int = -1
        self._last_encoded_pts: int = -1
        self._last_mux_dts: int = -1
        self._capture_packets_pending: int = 0
        self.container: Any = None
        self.stream: Any = None
        self.thread: threading.Thread | None = None

        if not self._open_container():
            return

        self.is_running = True
        self.thread = threading.Thread(
            target=self._worker,
            name=f"record-enc-{session_id[:12]}",
            daemon=True,
        )
        self.thread.start()
        logger.info(
            f"[RECORD] Started: {self.video_path} | format={self.fmt} "
            f"({self.codec_name}) | {width}x{height} @ {self.fps}fps | raw_nv12={self.is_raw_nv12}"
        )

    def _configure_stream(self, container, codec_name: str, options: dict):
        # AVI requires constant frame rate (CFR) matching capture rate so the container header
        # correctly reflects true frame count and framerate. MP4/MKV support VFR with rate=1000.
        stream_rate = int(self.fps or 30) if self.fmt == "AVI" else 1000
        stream = container.add_stream(codec_name, rate=stream_rate)
        stream.width   = self.width
        stream.height  = self.height
        stream.pix_fmt = self.pix_fmt

        if self.fmt == "AVI":
            stream.time_base = Fraction(1, int(self.fps or 30))
            stream.codec_context.time_base = Fraction(1, int(self.fps or 30))
        else:
            # Keep encoder and muxer on the same 1ms clock as our PTS values
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)

        if options:
            stream.options = options
        if codec_name == "ffv1":
            try:
                stream.codec_context.thread_type = "SLICE"
                stream.codec_context.thread_count = min(16, os.cpu_count() or 4)
            except Exception:
                pass
        if self.fmt == "MP4":
            try:
                stream.codec_context.colorspace = Colorspace.ITU709
                stream.codec_context.color_primaries = ColorPrimaries.BT709
                stream.codec_context.color_trc = ColorTrc.BT709
                stream.codec_context.color_range = ColorRange.JPEG
            except Exception:
                pass
        elif self.fmt == "FFV1":
            try:
                stream.codec_context.colorspace = Colorspace.ITU601
                stream.codec_context.color_primaries = ColorPrimaries.SMPTE170M
                stream.codec_context.color_trc = ColorTrc.BT709
                stream.codec_context.color_range = ColorRange.MPEG
            except Exception:
                pass
        return stream

    def _open_container(self) -> bool:
        container_options = {}
        if self.fmt == "MP4":
            # Fragmented MP4 writes playable chunks at each keyframe so video is crash-resilient
            container_options = {"movflags": "frag_keyframe+empty_moov"}

        try:
            self.container = av.open(self.video_path, mode="w", options=container_options)
            self.stream = self._configure_stream(self.container, self.codec_name, self.codec_options)
            return True
        except Exception as e:
            if self.container is not None:
                try:
                    self.container.close()
                except Exception:
                    pass
                self.container = None
                self.stream = None

            if self.fmt == "FFV1":
                logger.warning(f"[RECORD] FFV1 open with fast slice options failed ({e}); retrying minimal FFV1")
                self.codec_options = {"level": "3"}
                try:
                    self.container = av.open(self.video_path, mode="w", options=container_options)
                    self.stream = self._configure_stream(self.container, "ffv1", self.codec_options)
                    return True
                except Exception as ex2:
                    logger.error(f"[RECORD] Failed to open container: {ex2}")
                    self.container = None
                    self.stream = None
                    return False

            if self.fmt == "MP4" and self.codec_name != "libx264":
                logger.warning(f"[RECORD] Failed to open container with {self.codec_name} ({e}); falling back to libx264")
                self.codec_name = "libx264"
                self.codec_options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "19"}
                try:
                    self.container = av.open(self.video_path, mode="w", options=container_options)
                    self.stream = self._configure_stream(self.container, self.codec_name, self.codec_options)
                    return True
                except Exception as ex2:
                    logger.error(f"[RECORD] Failed to open container: {ex2}")
                    self.container = None
                    self.stream = None
                    return False

            logger.error(f"[RECORD] Failed to open container: {e}")
            return False

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _build_video_frame(self, payload) -> av.VideoFrame:
        if isinstance(payload, tuple) and len(payload) == 3 and payload[0] == "nv12":
            _, y_plane, uv_plane = payload
            frame = av.VideoFrame(self.width, self.height, "yuv420p")
            frame.planes[0].update(np.ascontiguousarray(y_plane).tobytes())
            frame.planes[1].update(np.ascontiguousarray(uv_plane[:, 0::2]).tobytes())
            frame.planes[2].update(np.ascontiguousarray(uv_plane[:, 1::2]).tobytes())
            return frame

        # BGR ndarray (OpenCV / camera)
        bgr = payload
        stream = self.stream
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        if self.fmt == "MP4":
            frame = self._reformatter.reformat(
                frame,
                format=stream.pix_fmt,
                src_color_range=ColorRange.JPEG,
                dst_color_range=ColorRange.JPEG,
                src_colorspace=Colorspace.ITU709,
                dst_colorspace=Colorspace.ITU709,
            )
            frame.color_range = ColorRange.JPEG
        elif frame.format.name != stream.pix_fmt:
            frame = self._reformatter.reformat(frame, format=stream.pix_fmt)
        return frame

    def _mux_packet(self, pkt) -> None:
        if self.container is None:
            return
        if pkt.dts is not None:
            if pkt.dts <= self._last_mux_dts:
                pkt.dts = self._last_mux_dts + 1
            self._last_mux_dts = pkt.dts
        if pkt.pts is not None and pkt.dts is not None and pkt.pts < pkt.dts:
            pkt.pts = pkt.dts
        self.container.mux(pkt)

    def _worker(self):
        try:
            while self.is_running or not self.frame_queue.empty():
                try:
                    payload, pts_ms = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if self.stream is None or self.container is None:
                    continue

                try:
                    if self.fmt == "AVI":
                        frame = self._build_video_frame(payload)
                        frame.pts = self._frames_written
                        frame.time_base = Fraction(1, int(self.fps or 30))
                    else:
                        if pts_ms <= self._last_encoded_pts:
                            pts_ms = self._last_encoded_pts + 1
                        self._last_encoded_pts = pts_ms

                        frame = self._build_video_frame(payload)
                        frame.pts = pts_ms
                        frame.time_base = Fraction(1, 1000)

                    for pkt in self.stream.encode(frame):
                        self._mux_packet(pkt)

                    self._frames_written += 1
                    if self._frames_written % 100 == 0:
                        logger.info(
                            f"[RECORD] Frames written: {self._frames_written} | "
                            f"queue={self.frame_queue.qsize()} | "
                            f"missing_cam={self._camera_frames_missing}"
                        )

                except Exception as e:
                    logger.warning(f"[RECORD] Frame encode error: {e}")

        except Exception as e:
            logger.error(f"[RECORD] Worker error: {e}", exc_info=True)
        finally:
            # Flush encoder and close container inside the worker so that
            # stop() can join() and know the file is fully written.
            if self.container is not None and self.stream is not None:
                try:
                    for pkt in self.stream.encode():
                        self._mux_packet(pkt)
                    self.container.close()
                    logger.info(
                        f"[RECORD] Closed — {self._frames_written} frames "
                        f"(dropped={self._dropped_frames}, missing_cam={self._camera_frames_missing}) → {self.video_path}"
                    )
                except Exception as e:
                    logger.error(f"[RECORD] Container close error: {e}")
                finally:
                    self.container = None
                    self.stream = None

    # ── Timestamp & Enqueue Helpers ───────────────────────────────────────────

    def _compute_pts_ms(self, cam_ts_sec: float | None, seq_num: int | None) -> int:
        if seq_num is not None:
            if self._last_seq_num is not None and seq_num > self._last_seq_num + 1:
                self._camera_frames_missing += (seq_num - self._last_seq_num - 1)
            self._last_seq_num = seq_num

        if cam_ts_sec is not None:
            if self._first_cam_ts_sec is None:
                self._first_cam_ts_sec = cam_ts_sec
                self._first_cam_offset_ms = max(0, int((time.monotonic() - self._start_mono) * 1000))
            pts_ms = self._first_cam_offset_ms + int((cam_ts_sec - self._first_cam_ts_sec) * 1000)
        else:
            pts_ms = int((time.monotonic() - self._start_mono) * 1000)

        if pts_ms <= self._last_pts_ms:
            pts_ms = self._last_pts_ms + 1
        self._last_pts_ms = pts_ms
        return pts_ms

    def _enqueue_payload(self, payload, pts_ms: int) -> None:
        self._last_frame_payload = payload
        self.frame_queue.put_nowait((payload, pts_ms))
        backlog = self.frame_queue.qsize()
        now = time.monotonic()
        if backlog >= 300 and now - self._last_backlog_log >= 2.0:
            self._last_backlog_log = now
            logger.warning(
                f"[RECORD] Encoder backlog is {backlog} frames; preserving all frames while encoder catches up"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_nv12_frame(
        self,
        y_plane: np.ndarray,
        uv_plane: np.ndarray,
        cam_ts_sec: float | None = None,
        seq_num: int | None = None,
    ) -> None:
        """Feed raw sensor NV12 Y/UV planes directly for fast lossless FFV1 recording."""
        if not self.is_running:
            return
        self._saw_nv12 = True
        pts_ms = self._compute_pts_ms(cam_ts_sec, seq_num)
        self._enqueue_payload(("nv12", y_plane, uv_plane), pts_ms)

    def add_frame(
        self,
        bgr: np.ndarray,
        cam_ts_sec: float | None = None,
        seq_num: int | None = None,
        copy_frame: bool = True,
    ) -> None:
        """Add uncompressed BGR frame to the recording queue."""
        if not self.is_running:
            return
        if self.is_raw_nv12 and self._saw_nv12:
            return

        pts_ms = self._compute_pts_ms(cam_ts_sec, seq_num)
        color_payload = np.ascontiguousarray(bgr)
        if copy_frame:
            color_payload = color_payload.copy()
        self._enqueue_payload(color_payload, pts_ms)

    def add_settings_event(self, settings: dict) -> None:
        elapsed = round(time.monotonic() - self._start_mono, 3)
        event = {"elapsed_seconds": elapsed, "settings": settings.copy()}
        self.camera_settings = settings.copy()
        if self.settings_history and self.settings_history[-1]["settings"] == settings:
            return
        self.settings_history.append(event)

    def stop(self) -> dict:
        # Pad final frame to actual wall-clock recording duration with monotonic PTS.
        if self._last_frame_payload is not None:
            pad_pts_ms = int((time.monotonic() - self._start_mono) * 1000)
            if pad_pts_ms <= self._last_pts_ms:
                pad_pts_ms = self._last_pts_ms + 1
            gap_ms = pad_pts_ms - self._last_pts_ms
            self._last_pts_ms = pad_pts_ms
            logger.info(
                f"[RECORD] Padding video by {gap_ms} ms "
                f"(last frame → pad to {pad_pts_ms} ms)"
            )
            try:
                self.frame_queue.put_nowait((self._last_frame_payload, pad_pts_ms))
            except queue.Full:
                pass

        logger.info(
            f"[RECORD] Stopping — {self.frame_queue.qsize()} frames still in queue"
        )
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=60.0)
            if self.thread.is_alive():
                logger.warning("[RECORD] Worker thread did not finish within 60 s")
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
            "dropped_frames": self._dropped_frames,
            "camera_frames_missing": self._camera_frames_missing,
            "format": self.fmt,
            "codec": self.codec_name,
            "width": self.width,
            "height": self.height,
        }


def _get_default_recording_path() -> str:
    desktop = get_desktop_dir()
    return str(desktop / "recordings")


def _resolve_recording_path(root_path: str | None = None) -> str:
    if root_path:
        return root_path
    try:
        from app.core.database import SessionLocal
        from app.models.camera_model import Camera

        db = SessionLocal()
        try:
            camera    = db.query(Camera).first()
            path = getattr(camera, "recording_video_path", None)
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
    root_path: str | None = None,
    recording_format: str = "MP4",
    camera_settings: dict | None = None,
    **kwargs: Any,
) -> str:
    logger.info(f"[RECORD] Creating session: {session_id} | format={recording_format} @ {fps}fps")
    if session_id in active_recordings:
        logger.warning("[RECORD] Session already exists — returning existing path")
        return active_recordings[session_id].video_path

    root_path = _resolve_recording_path(root_path)
    session = RecordingSession(
        session_id, width, height, fps, root_path, recording_format, camera_settings, **kwargs
    )

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
