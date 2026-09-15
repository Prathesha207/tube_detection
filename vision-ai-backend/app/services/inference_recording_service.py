"""
inference_recording_service.py
===============================
Records raw and processed video for each inference cycle.

Directory layout
----------------
base_path/
  YYYY-MM-DD/
    processed/
      PENDING/   ← written here during recording
      NORMAL/    ← moved here on stop(status="NORMAL")
      ANOMALY/   ← moved here on stop(status="ANOMALY")
    raw/
      PENDING/
      NORMAL/
      ANOMALY/

Filename format
---------------
  session_YYYY-MM-DD_HH-MM-SS-ffffff_<STATUS>_<session_id_safe>_cycle<N>.mp4

  e.g.
    session_2025-01-15_10-30-00-123456_NORMAL_cam0_cycle1.mp4
    session_2025-01-15_10-31-05-654321_ANOMALY_cam0_cycle2.mp4

Lifecycle
---------
  1. InferenceRecorder(status="PENDING") is created when recording starts.
     Writers open under PENDING/.
  2. write(frame, result) is called by run_inference() for EVERY frame.
     This is the only write point; the WebSocket handler
     must NOT call recorder.write() separately.
  3. update_status("NORMAL" | "ANOMALY") is called when inference resolves.
  4. stop() flushes the queue, releases writers, then renames both files
     from PENDING/ into the correct NORMAL/ or ANOMALY/ subfolder.
"""

import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import av
import cv2
import numpy as np

from app.core.app_paths import get_desktop_dir

logger = logging.getLogger("inference-recorder")



# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    radius: int,
    color: tuple,
    alpha: float = 0.72,
) -> None:
    overlay = img.copy()
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_hourglass(img: np.ndarray, cx: int, cy: int, r: int, color: tuple) -> None:
    pts_top = np.array([[cx - r, cy - r], [cx + r, cy - r], [cx, cy]], np.int32)
    pts_bot = np.array([[cx - r, cy + r], [cx + r, cy + r], [cx, cy]], np.int32)
    cv2.fillPoly(img, [pts_top], color)
    cv2.fillPoly(img, [pts_bot], color)


# ---------------------------------------------------------------------------
# Public draw_overlay
# ---------------------------------------------------------------------------

def draw_overlay(
    frame,
    detections,
    status,
):
    overlay_frame = frame.copy()

    # ── Bounding boxes ─────────────────────────────────────────────────
    for d in detections:
        # Check if it's the old contour style or new bbox style
        contour = d.get("contour")
        if contour and len(contour) > 2:
            pts = np.array(contour, np.int32)
            cv2.polylines(overlay_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
        bbox = d.get("bbox") or d.get("box")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Change color if anomaly
            is_anomaly = d.get("isAnomaly", False) or d.get("species") not in ["Duck", "duck"]
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
            
            label = d.get("species", "Duck")
            conf = d.get("confidence", d.get("conf", 0.0))
            text = f"{label} {int(conf * 100)}%"
            cv2.putText(overlay_frame, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── Status badge ────────────────────────────────────────
    STATUS_STYLES = {
        "NORMAL":  {"dot": (57, 255, 20),  "text": (57, 255, 20),  "bg": (30, 27, 40)},
        "ANOMALY": {"dot": (50, 50, 255),  "text": (50, 50, 255),  "bg": (30, 27, 40)},
        "PENDING": {"dot": (250, 204, 21), "text": (250, 204, 21), "bg": (30, 27, 40)},
    }
    style = STATUS_STYLES.get(status, {"dot": (220, 220, 220), "text": (220, 220, 220), "bg": (30, 27, 40)})
    FS_STATUS = 1.0
    FT_STATUS = 2
    (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_PLAIN, FS_STATUS, FT_STATUS)
    PAD_X_S, PAD_Y_S = 20, 12
    DOT_R_S = 8
    
    H, W = frame.shape[:2]
    sx1 = W // 2 - sw // 2 - PAD_X_S - DOT_R_S - 10
    sy1 = 14
    sx2 = W // 2 + sw // 2 + PAD_X_S
    sy2 = sy1 + sh + PAD_Y_S * 2
    
    _draw_rounded_rect(overlay_frame, sx1, sy1, sx2, sy2, 14, style["bg"], alpha=0.92)
    cv2.rectangle(overlay_frame, (sx1 + 2, sy1 + 2), (sx2 - 2, sy2 - 2), style["dot"], 2)
    s_dot_x = sx1 + PAD_X_S + DOT_R_S
    s_dot_y = (sy1 + sy2) // 2
    cv2.circle(overlay_frame, (s_dot_x, s_dot_y), DOT_R_S, style["dot"], -1)
    cv2.circle(overlay_frame, (s_dot_x, s_dot_y), DOT_R_S + 3, style["dot"], 1)
    cv2.putText(overlay_frame, status, (s_dot_x + DOT_R_S + 10, s_dot_y + sh // 2),
                cv2.FONT_HERSHEY_PLAIN, FS_STATUS, style["text"], FT_STATUS, cv2.LINE_AA)

    return overlay_frame


# ---------------------------------------------------------------------------
# InferenceRecorder
# ---------------------------------------------------------------------------

class InferenceRecorder:
    """
    Records one inference cycle using PyAV with per-frame wall-clock timestamps.

    Both MJPEG and FFV1 are stored in MKV (VFR-capable container).
    Duration is always correct regardless of actual camera delivery fps
    because each frame is stamped with its real arrival time, not a
    fixed-fps counter.
    """

    def __init__(
        self,
        session_id: str,
        width: int,
        height: int,
        fps: int,                    # kept for API compatibility; not used for timestamps
        status: str = "PENDING",
        video_name: str = "",
        recording_format: str = "FFV1",
        record_raw: bool = False,
        record_processed: bool = True,
    ):
        self._width  = width
        self._height = height

        self._final_status     = "ANOMALY"   # safe fallback if stop() fires early
        self._status_finalized = False
        self._is_running       = True

        # Wall-clock start — PTS is ms since this moment (set at creation, not first frame)
        self._start_mono: float = time.monotonic()
        self._last_pts_ms: int  = 0
        self._last_raw: Optional[np.ndarray]       = None
        self._last_proc: Optional[np.ndarray]      = None

        now         = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        _sid_safe   = session_id.replace("/", "_").replace(":", "-").replace(" ", "_")

        fmt = (recording_format or "FFV1").upper()
        options = {}
        if fmt in ("MP4", "H264", "LIBX264"):
            fmt        = "MP4"
            codec_name = "libx264"
            pix_fmt    = "yuv420p"
            _ext       = ".mp4"
            options    = {"preset": "ultrafast", "crf": "18"}
        elif fmt in ("MJPEG", "AVI"):
            fmt        = "MJPEG"
            codec_name = "mjpeg"
            pix_fmt    = "yuvj420p"
            _ext       = ".avi"
        else:  # Default: FFV1
            fmt        = "FFV1"
            codec_name = "ffv1"
            pix_fmt    = "yuv420p"
            _ext       = ".mkv"
            options    = {"level": "3"}

        if video_name:
            video_stem = os.path.splitext(video_name)[0]
            base_name  = f"{video_stem}_PENDING_{_sid_safe}{_ext}"
        else:
            base_name = now.strftime("session_%Y-%m-%d_%H-%M-%S-%f_PENDING") + f"_{_sid_safe}{_ext}"

        base_path = str(get_desktop_dir() / "inference_results")

        self._root      = os.path.join(base_path, date_folder)
        self._base_name = base_name
        os.makedirs(self._root, exist_ok=True)

        # ── open PyAV containers ──────────────────────────────────────────
        self._processed_pending_path: Optional[str] = None
        self._raw_pending_path:       Optional[str] = None

        self._processed_container = None
        self._processed_stream    = None
        self._raw_container       = None
        self._raw_stream          = None

        def _open_av(path: str):
            """Open a container with the configured codec. rate=1000 → 1ms PTS unit."""
            try:
                container = av.open(path, mode="w")
                stream    = container.add_stream(codec_name, rate=1000)
                stream.width   = width
                stream.height  = height
                stream.pix_fmt = pix_fmt
                if options:
                    stream.options = options
                return container, stream
            except Exception as e:
                logger.error(f"[INFERENCE RECORDER] Failed to open {path}: {e}")
                return None, None

        if record_processed:
            pending_dir = os.path.join(self._root, "processed", "PENDING")
            os.makedirs(pending_dir, exist_ok=True)
            path = os.path.join(pending_dir, base_name)
            self._processed_pending_path = path
            self._processed_container, self._processed_stream = _open_av(path)
            if self._processed_container:
                logger.info(f"[INFERENCE RECORDER] Processed (PENDING) → {path}")
            else:
                logger.error(f"[INFERENCE RECORDER] Processed container failed: {path}")

        if record_raw:
            pending_dir = os.path.join(self._root, "raw", "PENDING")
            os.makedirs(pending_dir, exist_ok=True)
            path = os.path.join(pending_dir, base_name)
            self._raw_pending_path = path
            self._raw_container, self._raw_stream = _open_av(path)
            if self._raw_container:
                logger.info(f"[INFERENCE RECORDER] Raw (PENDING) → {path}")
            else:
                logger.error(f"[INFERENCE RECORDER] Raw container failed: {path}")

        # ── background writer thread ──────────────────────────────────────
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._thread = threading.Thread(
            target=self._worker, daemon=True,
            name=f"inf-rec-{session_id[:8]}",
        )
        self._thread.start()
        logger.info(
            f"[INFERENCE RECORDER] Started — session={session_id} "
            f"codec={codec_name} container={_ext.lstrip('.')} size={width}×{height}"
        )

    # ------------------------------------------------------------------
    # Background writer thread
    # ------------------------------------------------------------------

    def _encode_frame(self, bgr: np.ndarray, stream, pts_ms: int) -> None:
        if bgr is None or stream is None:
            return
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame = frame.reformat(format=stream.pix_fmt)
        frame.pts = pts_ms
        container = stream.container
        for pkt in stream.encode(frame):
            container.mux(pkt)

    def _flush_and_close(self, stream, container) -> None:
        if stream and container:
            try:
                for pkt in stream.encode():
                    container.mux(pkt)
                container.close()
            except Exception as e:
                logger.error(f"[INFERENCE RECORDER] Container close error: {e}")

    def _worker(self) -> None:
        frames_written = 0
        try:
            while self._is_running or not self._queue.empty():
                try:
                    raw_bgr, proc_bgr, pts_ms = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                try:
                    self._encode_frame(raw_bgr,  self._raw_stream,       pts_ms)
                    self._encode_frame(proc_bgr, self._processed_stream, pts_ms)
                except Exception as e:
                    logger.warning(f"[INFERENCE RECORDER] Encode error: {e}")

                frames_written += 1
                if frames_written % 100 == 0:
                    logger.info(
                        f"[INFERENCE RECORDER] Written={frames_written} "
                        f"queue={self._queue.qsize()}"
                    )
        except Exception as exc:
            logger.error(f"[INFERENCE RECORDER] Worker error: {exc}", exc_info=True)
        finally:
            self._flush_and_close(self._raw_stream,       self._raw_container)
            self._flush_and_close(self._processed_stream, self._processed_container)
            logger.info(f"[INFERENCE RECORDER] Worker ended — total frames: {frames_written}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write(self, raw_frame: np.ndarray, result: dict) -> None:
        """Enqueue raw + processed frame pair with wall-clock PTS (non-blocking)."""
        if not self._is_running:
            return

        pts_ms = int((time.monotonic() - self._start_mono) * 1000)

        display_status = self._final_status if self._status_finalized else "PENDING"

        processed_frame = draw_overlay(
            raw_frame,
            detections=result.get("detections", []),
            status=display_status,
        )

        self._last_pts_ms = pts_ms
        self._last_raw    = raw_frame
        self._last_proc   = processed_frame

        try:
            self._queue.put_nowait((raw_frame, processed_frame, pts_ms))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((raw_frame, processed_frame, pts_ms))
            except Exception:
                pass

    @property
    def is_finalized(self) -> bool:
        return self._status_finalized

    def update_status(self, final_status: str) -> None:
        self._final_status     = final_status.upper()
        self._status_finalized = True
        logger.info(f"[INFERENCE RECORDER] Status updated → {self._final_status}")

    def _pad_and_stop(self) -> None:
        """Add a pad frame at current wall-clock time, then signal worker to stop."""
        if self._last_raw is not None:
            pad_pts_ms = int((time.monotonic() - self._start_mono) * 1000)
            gap_ms     = pad_pts_ms - self._last_pts_ms
            logger.info(
                f"[INFERENCE RECORDER] Padding by {gap_ms} ms "
                f"(last frame {self._last_pts_ms} ms → {pad_pts_ms} ms)"
            )
            try:
                self._queue.put_nowait((self._last_raw, self._last_proc, pad_pts_ms))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait((self._last_raw, self._last_proc, pad_pts_ms))
                except Exception:
                    pass
        self._is_running = False
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            logger.warning("[INFERENCE RECORDER] Worker did not finish within 30 s")

    def stop(self) -> None:
        """Flush, pad to actual duration, close containers, move files to NORMAL/ANOMALY."""
        logger.info(
            f"[INFERENCE RECORDER] Stopping — "
            f"{self._queue.qsize()} frames still in queue"
        )
        self._pad_and_stop()

        final    = self._final_status
        # rsplit replaces the LAST _PENDING_ (the one we inserted, before the session id).
        # Using replace(..., 1) would corrupt names like "my_PENDING_video_PENDING_sid.mkv".
        new_name = f"_{final}_".join(self._base_name.rsplit("_PENDING_", 1))

        for pending_path, subfolder in [
            (self._raw_pending_path,       "raw"),
            (self._processed_pending_path, "processed"),
        ]:
            if not pending_path or not os.path.exists(pending_path):
                continue
            dest_dir  = os.path.join(self._root, subfolder, final)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, new_name)
            try:
                os.rename(pending_path, dest_path)
                logger.info(
                    f"[INFERENCE RECORDER] Moved {subfolder}: "
                    f"PENDING/{os.path.basename(pending_path)} → {final}/{new_name}"
                )
            except Exception as exc:
                logger.error(
                    f"[INFERENCE RECORDER] Failed to move {pending_path} → {dest_path}: {exc}"
                )

        logger.info(f"[INFERENCE RECORDER] Stopped — final status: {final}")

    def discard(self) -> None:
        """Stop recording and delete the PENDING files without moving them."""
        logger.info(
            f"[INFERENCE RECORDER] Discarding — "
            f"{self._queue.qsize()} frames still in queue"
        )
        self._pad_and_stop()

        for pending_path in [self._raw_pending_path, self._processed_pending_path]:
            if not pending_path or not os.path.exists(pending_path):
                continue
            try:
                os.remove(pending_path)
                logger.info(f"[INFERENCE RECORDER] Discarded: {pending_path}")
            except Exception as exc:
                logger.error(f"[INFERENCE RECORDER] Failed to discard {pending_path}: {exc}")

        logger.info("[INFERENCE RECORDER] Discarded — no files saved")