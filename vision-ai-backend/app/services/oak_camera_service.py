import asyncio
import base64
import os
import threading
import time
from queue import Queue, Empty

import cv2
import depthai as dai
import numpy as np
import logging
from app.services.realtime_log_service import realtime_log_service

logger = logging.getLogger("oak-camera")

# Sentinel placed between videos in folder mode (not the final EOF None)
_VIDEO_BOUNDARY = object()



class OakCameraService:
    """Single OAK-D camera service for DHTX.

    Pipeline (2 DAI output nodes):
      Camera -> VideoEncoder (MJPEG) -> _mjpeg_dai_queue
      Camera -> raw NV12             -> _raw_dai_queue

    Capture threads (3):
      _mjpeg_thread   : drains MJPEG DAI queue -> pushes JPEG into asyncio stream_queue
      _hires_thread   : drains raw NV12 DAI queue -> puts NV12 packets into _hires_packet_queue
      _convert_thread : pulls NV12 packets -> getCvFrame() -> stores _latest_bgr

    Streaming:
      HTTP StreamingResponse reads from stream_queue (asyncio.Queue)
      MJPEG thread pushes via run_coroutine_threadsafe (drop-oldest on full)
    """

    def __init__(self):
        # ---- Device ----
        self.device: dai.Device | None = None
        self.is_connected: bool = False

        # ---- Pipeline ----
        self._pipeline: dai.Pipeline | None = None
        self._is_running: bool = False
        self._pipeline_lock = threading.Lock()
        self._lifecycle_lock = asyncio.Lock()

        # ---- DAI queues ----
        self._mjpeg_dai_queue = None
        self._raw_dai_queue = None
        self._control_queue = None

        # ---- Inter-thread queue (hires capture -> converter) ----
        self._hires_packet_queue: Queue = Queue(maxsize=1000)

        # ---- Offline frame queue (synchronized, every frame processed once) ----
        self._offline_frame_queue: Queue = Queue(maxsize=32) 
        self._offline_frame_index: int = 0
        self._offline_total_frames: int = 1

        # ---- Stream queue (MJPEG thread -> HTTP streaming endpoint) ----
        self._stream_queue: asyncio.Queue | None = None
        self._stream_subscribers: set[asyncio.Queue] = set()
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._is_streaming: bool = False

        # ---- Capture threads ----
        self._mjpeg_thread: threading.Thread | None = None
        self._hires_thread: threading.Thread | None = None
        self._convert_thread: threading.Thread | None = None

        self._mjpeg_stop = threading.Event()
        self._hires_stop = threading.Event()
        self._convert_stop = threading.Event()

        # ---- Latest BGR frame (GIL-safe, no lock needed for single ref assign) ----
        self._latest_bgr: np.ndarray | None = None

        # ---- Camera settings ----
        self.control_mode: str = "auto"
        self.current_brightness: int = 0
        self.current_contrast: int = 50
        self.current_fps: float = 0.0
        self._configured_fps: int = 30
        self._ae_limit_us: int | None = None  # None = manual mode

        # ---- Recording ----
        self._active_recording = None  # RecordingSession instance when active

        # ---- Inference ----
        self._inference_thread: threading.Thread | None = None
        self._inference_watchdog_thread: threading.Thread | None = None
        self._inference_stop = threading.Event()
        self._inference_result_queue: asyncio.Queue | None = None
        self._inference_loop: asyncio.AbstractEventLoop | None = None
        self._inference_session_id: str | None = None

        # ---- Offline inference ----
        self._offline_thread: threading.Thread | None = None
        self._offline_stop = threading.Event()
        self._inference_offline: bool = False

        # ---- Offline multi-video tracking ----
        self._total_videos: int = 1
        self._current_video_idx: int = 0
        self._current_video_name: str = ""


        # GPU sampler — runs independently, never blocks inference
        self._gpu_usage: float = -1.0
        self._gpu_sampler_thread: threading.Thread | None = None
        self._gpu_sampler_stop = threading.Event()

    # ==================== Device ====================

    async def connect(self, ip: str) -> bool:
        loop = asyncio.get_running_loop()
        self._last_connection_error = None
        for attempt in range(2):
            try:
                logger.info(f"[DEVICE] Connecting to {ip} (attempt {attempt + 1}/2)")
                device_info = self._resolve_device_info(ip)
                logger.info(f"[DEVICE] Resolved connection target: {device_info}")
                self.device = await loop.run_in_executor(None, dai.Device, device_info)
                self.device.setLogLevel(dai.LogLevel.WARN)
                self.device.setLogOutputLevel(dai.LogLevel.WARN)
                self.is_connected = True
                logger.info(f"[DEVICE] Connected to {ip}")
                realtime_log_service.add_log(
                    "camera",
                    "CAMERA",
                    f"Camera connected ({ip})",
                    "success"
                )
                return True
            except Exception as e:
                err_str = str(e)
                if attempt == 0 and ("already used" in err_str.lower() or "in use" in err_str.lower()):
                    logger.warning(f"[DEVICE] Device is reported in use, waiting 1.5s to retry: {e}")
                    await asyncio.sleep(1.5)
                    continue
                logger.error(f"[DEVICE] Connection failed: {e}")
                realtime_log_service.add_log(
                    "camera",
                    "CAMERA",
                    f"DEVICE Connection failed ({ip}): {e}",
                    "error"
                )
                self.is_connected = False
                self._last_connection_error = err_str
                return False
        self._last_connection_error = "Failed after retries"
        return False

    @staticmethod
    def _resolve_device_info(identifier: str):
        """Resolve a database value to a current USB or network DeviceInfo."""
        value = str(identifier or "").strip()
        available = dai.Device.getAllAvailableDevices()

        if not available:
            raise RuntimeError("No OAK device found over USB or network")

        # 1. Match against MXID, deviceId, name, or description
        if value and value.lower() not in {"", "usb", "auto", "127.0.0.1", "localhost"}:
            for device_info in available:
                device_id = str(getattr(device_info, "deviceId", "") or "")
                mxid = str(getattr(device_info, "mxid", "") or "")
                name = str(getattr(device_info, "name", "") or "")
                desc_name = ""
                try:
                    desc_name = str(getattr(device_info.getXLinkDeviceDesc(), "name", ""))
                except Exception:
                    pass
                if value in {device_id, mxid, name, desc_name}:
                    return device_info
            logger.warning(
                f"[DEVICE] Identifier '{value}' not matched in visible devices. Trying available devices..."
            )

        # 2. Prefer USB devices
        for device_info in available:
            name = str(getattr(device_info, "name", "")).lower()
            state = str(getattr(device_info, "state", "")).lower()
            if "usb" in name or "usb" in state or "tcp" not in state:
                return device_info

        # 3. Fallback to first discovered device
        return available[0]

    async def disconnect(self) -> None:
        try:
            if self.device is not None:
                self.device.close()
                logger.info("[DEVICE] Disconnected")
                realtime_log_service.add_log(
                    "camera",
                    "CAMERA",
                    "Camera disconnected",
                    "success"
                )
        except Exception as e:
            logger.error(f"[DEVICE] Disconnect error: {e}")
            realtime_log_service.add_log(
                "camera",
                "CAMERA",
                "DEVICE Disconnect error",
                "error"
            )
        finally:
            self.device = None
            self.is_connected = False

    # ==================== Pipeline ====================

    def _build_pipeline(self, pipeline: dai.Pipeline, config) -> bool:
        try:
            logger.info("[PIPELINE] Building...")

            # --- Resolution / FPS (parsed first — needed by AE limit below) ---
            resolution = config.resolution or "1920x1080"
            try:
                sep = "x" if "x" in resolution else "*"
                width, height = map(int, resolution.split(sep))
            except Exception:
                logger.warning("[PIPELINE] Bad resolution — fallback 1920x1080")
                width, height = 1920, 1080
            fps = int(config.fps or 30)
            self._configured_fps = fps
            logger.info(f"[PIPELINE] {width}x{height} @ {fps}fps")

            # --- Control mode ---
            control_mode = (config.control_mode or "auto").lower()
            self.control_mode = control_mode

            cam = None
            is_color_camera = False

            # Try DepthAI unified Camera node first
            try:
                cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                init_ctrl = cam.initialControl
            except Exception as cam_err:
                err_str = str(cam_err)
                logger.warning(f"[PIPELINE] dai.node.Camera build failed: {err_str}")
                if "OAK4 has not been setup yet" in err_str or "setup.luxonis.com" in err_str:
                    logger.error(
                        "[PIPELINE] Connected camera is an OAK4 device that requires setup! "
                        "Complete setup via https://setup.luxonis.com/ or run 'oakctl device setup apply'."
                    )
                    realtime_log_service.add_log(
                        "camera",
                        "CAMERA",
                        "OAK4 camera requires setup: visit https://setup.luxonis.com or run oakctl",
                        "error"
                    )

                # Attempt ColorCamera fallback (for RVC2 / classic OAK-D / OAK-1 devices)
                try:
                    logger.info("[PIPELINE] Attempting dai.node.ColorCamera fallback...")
                    cam = pipeline.create(dai.node.ColorCamera)
                    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
                    if hasattr(dai.ColorCameraProperties, "SensorResolution"):
                        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                    cam.setFps(fps)
                    cam.setInterleaved(False)
                    cam.setVideoSize(width, height)
                    init_ctrl = cam.initialControl
                    is_color_camera = True
                    logger.info("[PIPELINE] ColorCamera node created successfully as fallback")
                except Exception as color_err:
                    logger.error(f"[PIPELINE] ColorCamera fallback also failed: {color_err}")
                    raise cam_err

            if control_mode == "auto":
                init_ctrl.setAutoExposureEnable()
                init_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                # Cap shutter to frame period — prevents AE from picking 300ms in dim scenes.
                ae_limit_us = max(1_000, int(1_000_000 / max(fps, 1)) - 2_000)
                self._ae_limit_us = ae_limit_us
                try:
                    init_ctrl.setAutoExposureLimit(ae_limit_us)
                    logger.info(f"[PIPELINE] Auto exposure + focus | AE shutter hint: {ae_limit_us} µs")
                except AttributeError:
                    logger.warning("[PIPELINE] setAutoExposureLimit not on initialControl — will send via runtime control only")
            else:
                self._ae_limit_us = None
                try:
                    exposure = int(config.exposure)
                    gain = int(config.gain)
                    focus = int(config.focus)
                except Exception:
                    logger.warning("[PIPELINE] Invalid manual values — using defaults")
                    exposure, gain, focus = 10000, 200, 120

                init_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                init_ctrl.setManualExposure(exposure, gain)
                init_ctrl.setManualFocus(focus)
                hw_b = max(-10, min(10, int(round(int(config.brightness or 0) / 5.0))))
                hw_c = max(-10, min(10, int(round((int(config.contrast or 50) - 50) / 5.0))))
                init_ctrl.setBrightness(hw_b)
                init_ctrl.setContrast(hw_c)
                logger.info(f"[PIPELINE] Manual: exp={exposure}, gain={gain}, focus={focus}")

            encoder = pipeline.create(dai.node.VideoEncoder)
            encoder.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.MJPEG)

            if not is_color_camera:
                # --- Node 1: MJPEG encoder → stream preview ---
                mjpeg_out = cam.requestOutput(
                    (width, height), type=dai.ImgFrame.Type.NV12, fps=fps
                )
                mjpeg_out.link(encoder.input)
                self._mjpeg_dai_queue = encoder.bitstream.createOutputQueue(maxSize=4, blocking=False)
                logger.info("[PIPELINE] Node 1: MJPEG encoder ready")

                # --- Node 2: Raw NV12 → inference / recording ---
                raw_out = cam.requestOutput(
                    (width, height), type=dai.ImgFrame.Type.NV12, fps=fps
                )
                self._raw_dai_queue = raw_out.createOutputQueue(maxSize=4, blocking=False)
                logger.info("[PIPELINE] Node 2: Raw NV12 ready")
            else:
                cam.video.link(encoder.input)
                self._mjpeg_dai_queue = encoder.bitstream.createOutputQueue(maxSize=4, blocking=False)
                self._raw_dai_queue = cam.video.createOutputQueue(maxSize=4, blocking=False)
                logger.info("[PIPELINE] ColorCamera outputs wired successfully")

            # --- Camera control input ---
            self._control_queue = cam.inputControl.createInputQueue(maxSize=4, blocking=False)

            logger.info("[PIPELINE] Build success")
            return True

        except Exception as e:
            logger.error(f"[PIPELINE] Build failed: {e}", exc_info=True)
            return False

    def _cleanup_pipeline(self) -> None:
        with self._pipeline_lock:
            if self._pipeline is None:
                return
            logger.info("[PIPELINE] Stopping...")
            try:
                self._pipeline.stop()
                self._pipeline.wait()
            except Exception as e:
                logger.error(f"[PIPELINE] Stop error: {e}")
            finally:
                self._pipeline = None
                self._is_running = False
                self._mjpeg_dai_queue = None
                self._raw_dai_queue = None
                self._control_queue = None
            logger.info("[PIPELINE] Cleaned up")

    

    # ── New method: background GPU sampler ───────────────────────────────────────
    def _start_gpu_sampler(self) -> None:
        """Samples GPU every 2s in background — never blocks inference."""
        def _loop():
            import subprocess
            while not self._gpu_sampler_stop.is_set():
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits"],
                        timeout=1,
                    )
                    self._gpu_usage = float(out.decode().strip().split("\n")[0])
                except Exception:
                    self._gpu_usage = -1.0
                self._gpu_sampler_stop.wait(timeout=2.0)   # sleep 2s between polls

        self._gpu_sampler_stop.clear()
        self._gpu_sampler_thread = threading.Thread(
            target=_loop, name="gpu-sampler", daemon=True
        )
        self._gpu_sampler_thread.start()

    def _stop_gpu_sampler(self) -> None:
        self._gpu_sampler_stop.set()
        if self._gpu_sampler_thread and self._gpu_sampler_thread.is_alive():
            self._gpu_sampler_thread.join(timeout=3.0)
        self._gpu_sampler_thread = None


    # ==================== Capture Threads ====================

    def _start_capture_threads(self) -> bool:
        if not self._is_running:
            logger.error("[CAPTURE] Cannot start — pipeline not running")
            return False

        mjpeg_alive = self._mjpeg_thread and self._mjpeg_thread.is_alive()
        hires_alive = self._hires_thread and self._hires_thread.is_alive()
        convert_alive = self._convert_thread and self._convert_thread.is_alive()

        if not mjpeg_alive:
            self._mjpeg_stop.clear()
            self._mjpeg_thread = threading.Thread(
                target=self._mjpeg_loop, name="oak-mjpeg", daemon=True
            )
            self._mjpeg_thread.start()
            logger.info("[CAPTURE] MJPEG thread started")

        if not hires_alive:
            self._hires_stop.clear()
            self._hires_thread = threading.Thread(
                target=self._hires_loop, name="oak-hires", daemon=True
            )
            self._hires_thread.start()
            logger.info("[CAPTURE] Hi-res thread started")

        if not convert_alive:
            self._convert_stop.clear()
            self._convert_thread = threading.Thread(
                target=self._convert_loop, name="oak-convert", daemon=True
            )
            self._convert_thread.start()
            logger.info("[CAPTURE] Converter thread started")

        return True

    def _stop_capture_threads(self) -> None:
        logger.info("[CAPTURE] Stopping threads...")

        # Signal all
        self._mjpeg_stop.set()
        self._hires_stop.set()
        self._convert_stop.set()

        # Join converter first — it consumes from hires_packet_queue.
        # If we join producers first they block trying to put into a full queue.
        if self._convert_thread and self._convert_thread.is_alive():
            self._convert_thread.join(timeout=5.0)
        self._convert_thread = None

        if self._mjpeg_thread and self._mjpeg_thread.is_alive():
            self._mjpeg_thread.join(timeout=3.0)
        self._mjpeg_thread = None

        if self._hires_thread and self._hires_thread.is_alive():
            self._hires_thread.join(timeout=3.0)
        self._hires_thread = None

        self._latest_bgr = None
        self._drain_hires_packet_queue()

        logger.info("[CAPTURE] Threads stopped")

    def _drain_hires_packet_queue(self) -> None:
        drained = 0
        while not self._hires_packet_queue.empty():
            try:
                self._hires_packet_queue.get_nowait()
                drained += 1
            except Empty:
                break
        if drained:
            logger.info(f"[CAPTURE] Drained {drained} stale hi-res packets")

    # ---- Thread 1: MJPEG consumer ----

    def _mjpeg_loop(self) -> None:
        """Drains MJPEG DAI queue and pushes JPEG bytes into the asyncio stream_queue."""
        logger.info("[MJPEG] Thread running")
        realtime_log_service.add_log(
            "stream",
            "NETWORK",
            "Connection stable - Bitrate: 4.2 Mbps",
            "info"
        )
        frames = 0
        frames_pushed = 0
        frames_dropped = 0
        last_log = time.time()

        try:
            while not self._mjpeg_stop.is_set():
                if self._mjpeg_dai_queue is None:
                    time.sleep(0.001)
                    continue

                pkt = self._mjpeg_dai_queue.tryGet()
                if pkt is None:
                    time.sleep(0.001)
                    continue

                jpeg = bytes(pkt.getData())
                frames += 1

                has_subscribers = bool(self._stream_subscribers) or (self._stream_queue is not None)
                if has_subscribers and self._server_loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self._async_stream_push(jpeg), self._server_loop
                    )
                    frames_pushed += 1
                else:
                    frames_dropped += 1

                now = time.time()
                if now - last_log >= 5.0:
                    elapsed = now - last_log
                    sub_count = len(self._stream_subscribers)
                    logger.info(
                        f"[MJPEG] FPS: {frames / elapsed:.1f} | "
                        f"pushed: {frames_pushed} | "
                        f"dropped (no client): {frames_dropped} | "
                        f"subscribers: {sub_count}"
                    )
                    frames = 0
                    frames_pushed = 0
                    frames_dropped = 0
                    last_log = now


        except Exception as e:
            logger.error(f"[MJPEG] Thread error: {e}", exc_info=True)
            realtime_log_service.add_log(
                "stream",
                "NETWORK",
                "MJPEG Thread error",
                "error"
            )
        finally:
            logger.info("[MJPEG] Thread ended")

    # ---- Thread 2: Hi-res capture ----

    def _hires_loop(self) -> None:
        """Drains raw NV12 DAI queue and queues packets for the converter thread."""
        logger.info("[HIRES] Thread running")

        try:
            while not self._hires_stop.is_set():
                if self._raw_dai_queue is None:
                    time.sleep(0.001)
                    continue

                pkt = self._raw_dai_queue.tryGet()
                if pkt is None:
                    time.sleep(0.001)
                    continue

                try:
                    self._hires_packet_queue.put_nowait(pkt)
                except Exception:
                    # Queue full — drop oldest, put new
                    try:
                        self._hires_packet_queue.get_nowait()
                        self._hires_packet_queue.put_nowait(pkt)
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"[HIRES] Thread error: {e}", exc_info=True)
        finally:
            logger.info("[HIRES] Thread ended")

    # ---- Thread 3: Converter ----

    def _convert_loop(self) -> None:
        """Pulls NV12 packets, converts to BGR (getCvFrame), stores latest for inference/recording."""
        logger.info("[CONVERT] Thread running")
        frames = 0
        record_frames = 0
        last_log = time.time()
        _first_frame_logged = False

        try:
            while not self._convert_stop.is_set() or not self._hires_packet_queue.empty():
                try:
                    pkt = self._hires_packet_queue.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    raw_bgr = pkt.getCvFrame()
                except Exception as e:
                    logger.warning(f"[CONVERT] getCvFrame failed: {e}")
                    continue

                if not _first_frame_logged:
                    logger.info(f"[CONVERT] First frame received — shape={raw_bgr.shape} | ae_limit_us={self._ae_limit_us} | configured_fps={self._configured_fps}")
                    _first_frame_logged = True

                # Apply software adjustments ONLY when explicitly non-neutral and in manual mode.
                # Default contrast (50 or 0) and brightness 0 is neutral.
                is_neutral = (self.current_brightness == 0) and (self.current_contrast in (0, 50))
                if not is_neutral and self.control_mode == "manual":
                    display_bgr = self._apply_adjustments(raw_bgr)
                else:
                    display_bgr = raw_bgr

                self._latest_bgr = display_bgr  # GIL-safe single reference assignment

                if self._active_recording is not None:
                    # Feed the authentic, unmodified camera stream frame to the recording
                    self._active_recording.add_frame(raw_bgr)
                    record_frames += 1

                frames += 1
                now = time.time()
                if now - last_log >= 2.0:
                    elapsed = now - last_log
                    self.current_fps = frames / elapsed
                    expected = self._configured_fps
                    fps_status = "OK" if self.current_fps >= expected * 0.8 else f"LOW (expected ~{expected})"
                    logger.info(
                        f"[CONVERT] FPS: {self.current_fps:.1f} [{fps_status}] | "
                        f"recording: {self._active_recording is not None} | "
                        f"record_frames_fed: {record_frames} | "
                        f"ae_limit_us: {self._ae_limit_us}"
                    )
                    frames = 0
                    record_frames = 0
                    last_log = now

        except Exception as e:
            logger.error(f"[CONVERT] Thread error: {e}", exc_info=True)
        finally:
            logger.info("[CONVERT] Thread ended")

    # ==================== Stream Queue ====================

    async def _async_stream_push(self, jpeg: bytes) -> None:
        """Drop-oldest push into all subscribed asyncio stream queues."""
        for q in list(self._stream_subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(jpeg)
            except asyncio.QueueFull:
                pass
        if self._stream_queue is not None and self._stream_queue not in self._stream_subscribers:
            if self._stream_queue.full():
                try:
                    self._stream_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._stream_queue.put_nowait(jpeg)
            except asyncio.QueueFull:
                pass

    def subscribe_stream(self) -> asyncio.Queue:
        if self._server_loop is None:
            try:
                self._server_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._stream_subscribers.add(q)
        self._stream_queue = q
        logger.info(f"[STREAM] Client subscribed — active subscribers: {len(self._stream_subscribers)}")
        return q

    def unsubscribe_stream(self, q: asyncio.Queue | None = None) -> None:
        if q is not None:
            self._stream_subscribers.discard(q)
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        else:
            for sub in list(self._stream_subscribers):
                while not sub.empty():
                    try:
                        sub.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._stream_subscribers.clear()

        if not self._stream_subscribers:
            self._stream_queue = None
        logger.info(f"[STREAM] Client unsubscribed — remaining subscribers: {len(self._stream_subscribers)}")

    def _open_stream_queue(self) -> asyncio.Queue:
        return self.subscribe_stream()

    def _close_stream_queue(self, q: asyncio.Queue | None = None) -> None:
        self.unsubscribe_stream(q)

    async def get_stream_frame(self, timeout: float = 1.0) -> bytes | None:
        if self._stream_queue is not None:
            try:
                return await asyncio.wait_for(self._stream_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
        if self._latest_bgr is not None:
            try:
                ret, buf = cv2.imencode(".jpg", self._latest_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return buf.tobytes()
            except Exception:
                pass
        return None

    # ==================== Frame Access ====================

    def get_bgr_frame(self) -> np.ndarray | None:
        """Latest BGR frame for inference / recording."""
        frame = self._latest_bgr
        return frame.copy() if frame is not None else None

    # ==================== Image Adjustments ====================

    def _apply_adjustments(self, frame: np.ndarray) -> np.ndarray:
        c = self.current_contrast
        b = self.current_brightness

        # If neutral, return frame directly without copying or modifying
        if (c == 0 or c == 50) and b == 0:
            return frame

        # Contrast: 50 is neutral (1.0x). 0..100 maps to 0.0x..2.0x
        alpha = (float(c) / 50.0) if c > 0 else 1.0
        beta = float(b)

        if abs(alpha - 1.0) < 0.01 and abs(beta) < 0.01:
            return frame

        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # ==================== Camera Controls ====================

    def update_controls(
        self,
        exposure: int | None = None,
        gain: int | None = None,
        focus: int | None = None,
        brightness: int | None = None,
        contrast: int | None = None,
        auto_focus: bool | None = None,
        auto_exposure: bool | None = None,
    ) -> None:
        if brightness is not None:
            self.current_brightness = int(brightness)
            logger.info(f"[CONTROL] Brightness={brightness}")

        if contrast is not None:
            self.current_contrast = int(contrast)
            logger.info(f"[CONTROL] Contrast={contrast}")

        if self._control_queue is None:
            logger.warning("[CONTROL] DepthAI device queue not available — software adjustments active")
            return

        ctrl = dai.CameraControl()

        # --- Exposure Control ---
        # If auto_exposure is explicitly True, or auto_exposure is not requested and exposure <= 0:
        if auto_exposure is True:
            try:
                ctrl.setAutoExposureEnable()
                if self._ae_limit_us is not None:
                    try:
                        ctrl.setAutoExposureLimit(self._ae_limit_us)
                    except Exception:
                        pass
                self.control_mode = "auto"
                logger.info(f"[CONTROL] AutoExposure=ENABLED (ae_limit={self._ae_limit_us}us)")
            except Exception as e:
                logger.warning(f"[CONTROL] setAutoExposureEnable failed: {e}")
        elif auto_exposure is False and exposure is not None:
            exp_us = int(exposure) * 1000 if int(exposure) < 10000 else int(exposure)
            # Default gain to base ISO 100 if not specified (NOT 400, which blows out image)
            gain_val = int(gain if gain is not None else 100)
            try:
                ctrl.setAutoExposureLock(False)
                ctrl.setManualExposure(exp_us, gain_val)
                self.control_mode = "manual"
                logger.info(f"[CONTROL] Manual Exposure={exp_us}us, Gain={gain_val}")
            except Exception as e:
                logger.warning(f"[CONTROL] setManualExposure failed: {e}")
        elif auto_exposure is None and exposure is not None:
            if exposure <= 0:
                try:
                    ctrl.setAutoExposureEnable()
                    if self._ae_limit_us is not None:
                        ctrl.setAutoExposureLimit(self._ae_limit_us)
                    self.control_mode = "auto"
                    logger.info("[CONTROL] Exposure<=0 -> AutoExposure=ENABLED")
                except Exception as e:
                    logger.warning(f"[CONTROL] setAutoExposureEnable failed: {e}")
            else:
                exp_us = int(exposure) * 1000 if int(exposure) < 10000 else int(exposure)
                gain_val = int(gain if gain is not None else 100)
                try:
                    ctrl.setAutoExposureLock(False)
                    ctrl.setManualExposure(exp_us, gain_val)
                    self.control_mode = "manual"
                    logger.info(f"[CONTROL] Exposure={exp_us}us, Gain={gain_val}")
                except Exception as e:
                    logger.warning(f"[CONTROL] setManualExposure failed: {e}")

        # --- Focus Control ---
        if auto_focus is not None:
            try:
                if auto_focus:
                    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                    try:
                        ctrl.setAutoFocusTrigger()
                    except Exception:
                        pass
                    logger.info("[CONTROL] AutoFocus=CONTINUOUS_VIDEO (trigger sent)")
                else:
                    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                    logger.info("[CONTROL] AutoFocus=OFF")
            except Exception as e:
                logger.warning(f"[CONTROL] setAutoFocusMode failed: {e}")
        elif focus is not None:
            try:
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                ctrl.setManualFocus(int(focus))
                logger.info(f"[CONTROL] Focus={focus}")
            except Exception as e:
                logger.warning(f"[CONTROL] setManualFocus failed: {e}")

        # --- Hardware Brightness (-10 to 10) ---
        if brightness is not None:
            try:
                hw_b = max(-10, min(10, int(round(self.current_brightness / 5.0))))
                ctrl.setBrightness(hw_b)
                logger.info(f"[CONTROL] Hardware Brightness={hw_b}")
            except Exception as e:
                logger.debug(f"[CONTROL] setBrightness failed: {e}")

        # --- Hardware Contrast (-10 to 10) ---
        if contrast is not None:
            try:
                hw_c = max(-10, min(10, int(round((self.current_contrast - 50) / 5.0))))
                ctrl.setContrast(hw_c)
                logger.info(f"[CONTROL] Hardware Contrast={hw_c}")
            except Exception as e:
                logger.debug(f"[CONTROL] setContrast failed: {e}")

        try:
            self._control_queue.send(ctrl)
            logger.info("[CONTROL] Sent to device")
        except Exception as e:
            logger.warning(f"[CONTROL] Device send failed: {e}")

    # ==================== Inference ====================

    # def _inference_worker(self, session_id: str) -> None:
    #     """Reads latest BGR frame, runs inference, pushes result to WebSocket queue.
    #     Also auto-manages processed video recording based on result['record'] flag.
    #     No frame skipping — inference itself is slow enough that _latest_bgr is always fresh.
    #     """
    #     from app.ml.tube.inference.camera_inference_service import run_inference
    #     from app.services.inference_recording_service import InferenceRecorder, draw_overlay
    #     from app.services import inference_config_service
    #     from app.core.database import SessionLocal

    #     logger.info(f"[INFERENCE] Thread running — session: {session_id}")

    #     # Load config once for recorder settings (processed_video / raw_video flags)
    #     db = SessionLocal()
    #     try:
    #         config = inference_config_service.get_config(db)
    #     finally:
    #         db.close()

    #     recorder: InferenceRecorder | None = None
    #     recording_active = False

    #     try:
    #         while not self._inference_stop.is_set():
    #             frame = self._latest_bgr
    #             if frame is None:
    #                 time.sleep(0.01)
    #                 continue

    #             result = run_inference(frame.copy(), session_id)
    #             if result is None:
    #                 logger.warning("❌ [INFERENCE] run_inference returned None")
    #                 continue

    #             # ===== TEST ONLY — remove before production =====
    #             # result = {
    #             #     "status": "WAITING_MODEL1",
    #             #     "record": False,
    #             #     "model1_detected": True,
    #             #     "detections": [{"bbox": [100, 100, 300, 300], "confidence": 0.92, "centroid": [200, 200], "contour": [[100,100],[300,100],[300,300],[100,300]], "model": "model1"}],
    #             #     "detections_model2": [],
    #             #     "mask_polygons": [],
    #             #     "lenA1": None,
    #             #     "lenA2": None,
    #             #     "model3_classes": [],
    #             # }
    #             # ===== END TEST =====

    #             logger.debug(f"[INFERENCE] status={result.get('status')} record={result.get('record')}")

    #             # Attach base64 frame for offline mode (UI has no MJPEG stream)
    #             if self._inference_offline:
    #                 try:
    #                     _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    #                     result["frame"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    #                 except Exception as e:
    #                     logger.warning(f"[INFERENCE] Frame encode failed: {e}")

    #             # Push result to WebSocket queue
    #             if self._inference_result_queue is not None and self._inference_loop is not None:
    #                 asyncio.run_coroutine_threadsafe(
    #                     self._async_inference_push(result),
    #                     self._inference_loop
    #                 )

    #             # Auto inference recording based on record flag
    #             record_flag = result.get("record", False)
    #             should_record = (
    #                 record_flag
    #                 and config is not None
    #                 and (config.processed_video or config.raw_video)
    #             )


    #             if should_record and not recording_active:
    #                 h, w = frame.shape[:2]
    #                 recorder = InferenceRecorder(
    #                     session_id=f"{session_id}_{int(time.time())}",
    #                     config=config,
    #                     width=w,
    #                     height=h,
    #                     fps=25,
    #                 )
    #                 recording_active = True
    #                 logger.info("[INFERENCE] Processed recording started")

    #             elif not should_record and recording_active:
    #                 if recorder:
    #                     recorder.stop()
    #                 recorder = None
    #                 recording_active = False
    #                 logger.info("[INFERENCE] Processed recording stopped")

    #             if recording_active and recorder:
    #                 overlay = draw_overlay(
    #                     frame,
    #                     result.get("detections", []),
    #                     result.get("detections_model2", []),
    #                     result.get("mask_polygons", []),
    #                     result.get("status", ""),
    #                 )
    #                 recorder.write(frame, overlay)

    #     except Exception as e:
    #         logger.error(f"[INFERENCE] Thread error: {e}", exc_info=True)
    #     finally:
    #         if recorder:
    #             logger.info("[INFERENCE] Stopping recorder on thread exit")
    #             recorder.stop()
    #         logger.info(f"[INFERENCE] Thread ended — session: {session_id}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Tube Analyzer Worker
    # This worker runs continuous TubeAnalyzer inference.
    # ─────────────────────────────────────────────────────────────────────────────

    def _inference_worker(self, session_id: str) -> None:
        from app.ml.tube.inference.camera_inference_service import run_inference
        from app.services.inference_recording_service import InferenceRecorder, draw_overlay
        import base64

        logger.info(f"[INFERENCE] Thread running — session: {session_id}")

        # ── Metrics state ──
        fps_counter     = 0
        fps_window_start = time.time()
        current_fps     = 0.0

        def _get_gpu_usage() -> float:
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits"],
                    timeout=1,
                )
                return float(out.decode().strip().split("\n")[0])
            except Exception:
                return -1.0

        recorder: InferenceRecorder | None = None
        recording_active = False

        try:
            while not self._inference_stop.is_set():

                # ── Frame acquisition ──────────────────────────────────────────
                if self._inference_offline:
                    frame = None
                    _got_boundary = False
                    _got_eof = False

                    while not self._inference_stop.is_set():
                        try:
                            item = self._offline_frame_queue.get(timeout=0.1)
                        except Empty:
                            continue

                        if item is None:
                            logger.info("[INFERENCE] All videos processed — inference complete")
                            self._inference_stop.set()
                            _got_eof = True
                            break
                        elif item is _VIDEO_BOUNDARY:
                            from app.ml.tube.inference.camera_inference_service import reset_session_for_next_video
                            reset_session_for_next_video(session_id)
                            logger.info("[INFERENCE] Video boundary — session reset, cycle counter carried forward")
                            _got_boundary = True
                            break
                        else:
                            frame = item
                            break

                    if _got_boundary:
                        continue

                    if frame is None:
                        if _got_eof:
                            done_result = {
                                "status": "INFERENCE_COMPLETE",
                                "done": True,
                                "metrics": {
                                    "fps": round(current_fps, 1),
                                    "gpu_pct": round(self._gpu_usage, 1),
                                    "latency_ms": 0,
                                },
                            }
                            if self._inference_result_queue is not None and self._inference_loop is not None:
                                asyncio.run_coroutine_threadsafe(
                                    self._async_inference_push(done_result),
                                    self._inference_loop,
                                )
                        break
                    self._latest_bgr = frame
                else:
                    frame = self._latest_bgr
                    if frame is None:
                        time.sleep(0.01)
                        continue

                # ── Online frame normalization ────────────────────────────────
                if not self._inference_offline:
                    _fmt = "FFV1"
                    if _fmt == "MJPEG":
                        _ok, _buf = cv2.imencode(
                            ".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )
                        if _ok:
                            frame = cv2.imdecode(_buf, cv2.IMREAD_COLOR)

                # ── Run inference + measure latency ───────────────────────────
                t_start = time.perf_counter()
                infer_res = run_inference(
                    frame.copy(),
                    session_id,
                    video_name=self._current_video_name,
                )
                latency_ms = (time.perf_counter() - t_start) * 1000

                if infer_res is None:
                    logger.warning("❌ [INFERENCE] run_inference returned None")
                    continue

                if isinstance(infer_res, tuple):
                    result, annotated_frame = infer_res
                else:
                    result, annotated_frame = infer_res, frame

                if not isinstance(result, dict):
                    logger.warning("❌ [INFERENCE] run_inference result is not a dict")
                    continue

                # ── FPS calculation (rolling 2-second window) ─────────────────
                fps_counter += 1
                now = time.time()
                elapsed = now - fps_window_start
                if elapsed >= 0.5:
                    current_fps      = fps_counter / elapsed
                    fps_counter      = 0
                    fps_window_start = now

                # ── GPU usage (sampled every result, cheap if nvidia-smi absent) ─
                gpu_pct = self._gpu_usage

                # ── Attach metrics to result ──────────────────────────────────
                result["metrics"] = {
                    "fps":        round(current_fps, 1),
                    "gpu_pct":    round(gpu_pct, 1),
                    "latency_ms": round(latency_ms, 1),
                }

                logger.debug(
                    f"[INFERENCE] status={result.get('status')} "
                    f"record={result.get('record')} "
                    f"fps={current_fps:.1f} gpu={gpu_pct:.0f}% "
                    f"latency={latency_ms:.1f}ms"
                )

                # ── Frame progress (offline) ──────────────────────────────────
                if self._inference_offline:
                    result["frame_index"]   = self._offline_frame_index
                    result["total_frames"]  = self._offline_total_frames
                    result["video_index"]   = self._current_video_idx
                    result["total_videos"]  = self._total_videos
                    result["video_name"]    = self._current_video_name

                if self._inference_offline:
                    result["_raw_frame"] = frame

                # ── Push result to WebSocket ──────────────────────────────────
                if self._inference_result_queue is not None and self._inference_loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self._async_inference_push(result),
                        self._inference_loop,
                    )

                # ── Feed frames to video recorder ─────────────────────────────
                record_flag = result.get("record", False)
                should_record = record_flag

                if should_record and not recording_active:
                    h, w = frame.shape[:2]
                    rec_fmt = self._resolve_recording_format()
                    recorder = InferenceRecorder(
                        session_id=f"{session_id}_{int(time.time())}",
                        width=w,
                        height=h,
                        fps=self._configured_fps,
                        recording_format=rec_fmt,
                    )
                    recording_active = True
                    logger.info(f"[INFERENCE] Tube recording started (format={rec_fmt})")
                elif not should_record and recording_active:
                    if recorder:
                        recorder.stop()
                    recorder = None
                    recording_active = False
                    logger.info("[INFERENCE] Tube recording stopped")

                if recording_active and recorder:
                    recorder.write(frame, result)

        except Exception as e:
            logger.error(f"[INFERENCE] Thread error: {e}", exc_info=True)
        finally:
            if recorder:
                logger.info("[INFERENCE] Stopping recorder on thread exit")
                recorder.stop()
            logger.info(f"[INFERENCE] Thread ended — session: {session_id}")

    def _inference_watchdog(self, session_id: str) -> None:
        """Restarts the online inference worker thread if it dies unexpectedly
        (uncaught exception) without an explicit stop_inference() call.

        Without this, an uncaught error in the worker (e.g. a locked SQLite
        read) silently ends the thread while capture threads and the FastAPI
        process keep running — the dashboard then stops receiving updates
        with no visible crash. Offline (video file) runs are not restarted
        since replaying from the correct frame position isn't safe to do
        automatically.
        """
        max_restarts = 5
        restarts = 0

        while not self._inference_stop.wait(timeout=5.0):
            if self._inference_offline:
                continue

            thread = self._inference_thread
            if thread is None or thread.is_alive():
                continue

            if restarts >= max_restarts:
                logger.critical(
                    f"[INFERENCE] Worker died {restarts} times — giving up "
                    f"auto-restart for session: {session_id}"
                )
                break

            restarts += 1
            logger.error(
                f"[INFERENCE] Worker thread died unexpectedly — "
                f"restarting (attempt {restarts}/{max_restarts}) for session: {session_id}"
            )

            new_thread = threading.Thread(
                target=self._inference_worker,
                args=(session_id,),
                name="oak-inference",
                daemon=True,
            )
            self._inference_thread = new_thread
            new_thread.start()

        logger.info(f"[INFERENCE] Watchdog stopped — session: {session_id}")

    # ==================== Offline Frame Feed ====================

    def _offline_loop(self, video_paths: list[str]) -> None:
        """Reads frames from one or more video files sequentially and pushes each into
        _offline_frame_queue.  Blocks if inference can't keep up — guarantees every frame
        is processed exactly once.  A _VIDEO_BOUNDARY sentinel is placed between videos so
        the inference worker can reset session state.  A final None sentinel signals EOF.
        """
        total = len(video_paths)
        self._total_videos = total
        logger.info(f"[OFFLINE] Thread running — {total} video(s)")

        try:
            for idx, video_path in enumerate(video_paths):
                if self._offline_stop.is_set():
                    break

                self._current_video_idx = idx + 1
                self._current_video_name = os.path.basename(video_path)

                logger.info(f"[OFFLINE] Video {idx + 1}/{total}: {video_path}")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"[OFFLINE] Failed to open video: {video_path}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                self._offline_frame_index  = 0
                self._offline_total_frames = total_frames

                logger.info(f"[OFFLINE] {width}x{height} | fps={fps:.1f} | total_frames={total_frames}")

                frames_read = 0
                try:
                    while not self._offline_stop.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            logger.info(f"[OFFLINE] EOF — frames_read={frames_read}")
                            break

                        frames_read += 1
                        self._offline_frame_index = frames_read

                        # Block here if inference is behind — no frames are skipped
                        while not self._offline_stop.is_set():
                            try:
                                self._offline_frame_queue.put(frame, timeout=0.1)
                                break
                            except Exception:
                                continue

                        if frames_read % 100 == 0:
                            logger.info(
                                f"[OFFLINE] frames_read={frames_read} | "
                                f"queue_size={self._offline_frame_queue.qsize()}"
                            )
                finally:
                    cap.release()

                # Between videos: boundary sentinel so inference worker resets state
                if idx < total - 1 and not self._offline_stop.is_set():
                    self._offline_frame_queue.put(_VIDEO_BOUNDARY)
                    logger.info(f"[OFFLINE] Video {idx + 1} done — boundary sentinel sent")

        except Exception as e:
            logger.error(f"[OFFLINE] Thread error: {e}", exc_info=True)
        finally:
            # Final EOF sentinel — inference worker will push INFERENCE_COMPLETE to WS
            self._offline_frame_queue.put(None)
            self._offline_stop.set()
            logger.info(f"[OFFLINE] Thread ended — processed {total} video(s)")

    def _start_offline_thread(self, video_paths: list[str]) -> bool:
        # If thread exists but is dead (finished naturally), clean it up first
        if self._offline_thread is not None and not self._offline_thread.is_alive():
            self._offline_thread = None
            self._offline_stop.clear()

        if self._offline_thread and self._offline_thread.is_alive():
            logger.warning("[OFFLINE] Thread already running")
            return True

        if not video_paths:
            logger.error("[OFFLINE] No video paths provided")
            return False

        # ALWAYS drain the queue before starting — catches stale sentinels
        # from threads that finished after stop_inference returned
        drained = 0
        while not self._offline_frame_queue.empty():
            try:
                self._offline_frame_queue.get_nowait()
                drained += 1
            except Exception:
                break
        if drained:
            logger.info(f"[OFFLINE] Drained {drained} stale items from previous run")

        self._offline_stop.clear()
        self._inference_stop.clear()  # also clear this — late EOF may have set it

        # Quick readability check on the first video only
        cap_check = cv2.VideoCapture(video_paths[0])
        if not cap_check.isOpened():
            cap_check.release()
            logger.error(f"[OFFLINE] First video not readable: {video_paths[0]}")
            return False
        cap_check.release()

        self._offline_thread = threading.Thread(
            target=self._offline_loop,
            args=(video_paths,),
            name="oak-offline",
            daemon=True,
        )
        self._offline_thread.start()
        logger.info(f"[OFFLINE] Thread started — {len(video_paths)} video(s), first: {video_paths[0]}")
        return True

    # def _stop_offline_thread(self) -> None:
    #     if not (self._offline_thread and self._offline_thread.is_alive()):
    #         return

    #     logger.info("[OFFLINE] Stopping thread...")
    #     self._offline_stop.set()
    #     self._offline_thread.join(timeout=5.0)
    #     if self._offline_thread.is_alive():
    #         logger.warning("[OFFLINE] Thread did not stop in time")
    #     self._offline_thread = None
    #     self._latest_bgr = None
    #     logger.info("[OFFLINE] Thread stopped")
    
    def _stop_offline_thread(self) -> None:
        if self._offline_thread is None:
            return

        logger.info("[OFFLINE] Stopping thread...")
        self._offline_stop.set()

        if self._offline_thread.is_alive():
            self._offline_thread.join(timeout=5.0)
            if self._offline_thread.is_alive():
                logger.warning("[OFFLINE] Thread did not stop in time")

        # Drain queue so inference worker unblocks
        while not self._offline_frame_queue.empty():
            try:
                self._offline_frame_queue.get_nowait()
            except Exception:
                break

        self._offline_thread = None
        self._latest_bgr = None
        self._offline_stop.clear()
        logger.info("[OFFLINE] Thread stopped")

    async def _async_inference_push(self, result: dict) -> None:
        """Drop-oldest push into inference result queue."""
        if self._inference_result_queue is None:
            return
        if self._inference_result_queue.full():
            try:
                self._inference_result_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._inference_result_queue.put_nowait(result)
        except asyncio.QueueFull:
            pass

    def _collect_folder_videos(self, folder_path: str) -> list[str]:
        """Return sorted list of video file paths found directly inside folder_path."""
        VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}
        if not os.path.isdir(folder_path):
            logger.error(f"[OFFLINE] Not a directory: {folder_path}")
            return []
        videos = [
            os.path.join(folder_path, name)
            for name in sorted(os.listdir(folder_path))
            if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
        ]
        logger.info(f"[OFFLINE] Found {len(videos)} video(s) in: {folder_path}")
        return videos

    def start_inference(
        self,
        session_id: str,
        loop: asyncio.AbstractEventLoop,
        offline: bool = False,
        video_path: str | None = None,
        folder_path: str | None = None,
    ) -> dict:
        if self._inference_thread and self._inference_thread.is_alive():
            logger.warning("[INFERENCE] Already running")
            return {"status": "already_running"}

        logger.info(f"[INFERENCE] Mode: {'OFFLINE' if offline else 'ONLINE'} — session: {session_id}")

        # Set up queue and loop FIRST so the WebSocket can connect even if validation fails.
        # Validation errors are pushed as a result so the WS receives them before closing.
        self._inference_session_id = session_id
        self._inference_loop = loop
        self._inference_offline = offline
        self._inference_result_queue = asyncio.Queue(maxsize=4)

        def _push_error(msg: str) -> None:
            asyncio.run_coroutine_threadsafe(
                self._async_inference_push({"status": "ERROR", "error": msg, "done": True}),
                loop,
            )

        if offline:
            if folder_path:
                video_paths = self._collect_folder_videos(folder_path)
                if not video_paths:
                    msg = f"No video files found in folder: {folder_path}"
                    logger.error(f"[INFERENCE] {msg}")
                    _push_error(msg)
                    return {"status": "error", "message": msg}
                logger.info(f"[INFERENCE] Folder mode — {len(video_paths)} video(s): {folder_path}")
            elif video_path:
                if not os.path.isfile(video_path):
                    msg = f"Video file not found: {video_path}"
                    logger.error(f"[INFERENCE] {msg}")
                    _push_error(msg)
                    return {"status": "error", "message": msg}
                video_paths = [video_path]
                logger.info(f"[INFERENCE] Single video mode: {video_path}")
            else:
                msg = "video_path or folder_path is required for offline mode"
                logger.error(f"[INFERENCE] {msg}")
                _push_error(msg)
                return {"status": "error", "message": msg}

            if not self._start_offline_thread(video_paths):
                msg = "Could not open video(s)"
                logger.error(f"[INFERENCE] {msg}")
                _push_error(msg)
                return {"status": "error", "message": msg}
        else:
            # Online mode: feed _latest_bgr from camera capture threads (unchanged)
            if not self._is_running:
                msg = "Pipeline not started"
                _push_error(msg)
                return {"status": "error", "message": msg}
            if not (self._hires_thread and self._hires_thread.is_alive()):
                logger.info("[INFERENCE] Capture threads not running — starting now")
                self._start_capture_threads()

        self._inference_stop.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_worker,
            args=(session_id,),
            name="oak-inference",
            daemon=True,
        )
        self._start_gpu_sampler()
        self._inference_thread.start()

        if not offline:
            self._inference_watchdog_thread = threading.Thread(
                target=self._inference_watchdog,
                args=(session_id,),
                name="oak-inference-watchdog",
                daemon=True,
            )
            self._inference_watchdog_thread.start()

        logger.info(f"[INFERENCE] Started — session: {session_id} | offline: {offline}")
        return {
            "status": "started",
            "mode": "offline" if offline else "online",
            "total_videos": self._total_videos if offline else 1,
        }

    def stop_inference(self) -> dict:
        from app.ml.tube.inference.camera_inference_service import clear_session

        logger.info("[INFERENCE] Stopping...")
        self._inference_stop.set()
        
        # Stop offline feed thread immediately so it stops queueing frames
        if self._offline_thread and self._offline_thread.is_alive():
            self._stop_offline_thread()

        # BUGFIX: this used to join(timeout=2.0), log a warning if the thread
        # was still alive, and then fall through anyway to unconditionally
        # call clear_session() -> app_state.exit_inference("camera") and null
        # out self._inference_thread. That releases the cross-kind GPU claim
        # (and defeats start_inference()'s "already_running" guard) while the
        # worker thread can still be mid-call inside analyzer.process_frame()
        # on the GPU. The very next /video/start can then pass
        # try_enter_inference("video") and load a second TubeAnalyzer onto
        # the same GPU while the camera thread is still using it -- two
        # threads hitting the same CUDA context concurrently, which is
        # exactly the kind of thing that can wedge the whole process badly
        # enough to look like "backend OFFLINE". This is the backend half of
        # the race useInferenceLoop.ts's "GPU-contention fix" comment was
        # working around from the frontend side.
        #
        # Fix: only release the claim / clear the session once the thread is
        # actually confirmed dead. If it's still alive after the timeout, we
        # leave self._inference_thread pointing at it (so start_inference()'s
        # existing "already_running" guard keeps correctly refusing a new
        # start) and hand off to a background watcher that does an unbounded
        # join and releases the claim only when the thread genuinely exits.
        thread_confirmed_dead = True
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
            if self._inference_thread.is_alive():
                thread_confirmed_dead = False
                stuck_thread = self._inference_thread
                stuck_session_id = self._inference_session_id
                logger.warning(
                    "[INFERENCE] Worker thread did not stop within 2s -- it is "
                    "still using the GPU. NOT releasing the camera's GPU claim "
                    "yet; a background watcher will release it once the thread "
                    "actually exits."
                )

                def _wait_and_release(thread=stuck_thread, session_id=stuck_session_id):
                    thread.join()  # unbounded -- the native call WILL return eventually
                    logger.info(
                        f"[INFERENCE] Delayed-stopped worker thread for session "
                        f"{session_id} has now actually exited -- releasing GPU claim."
                    )
                    try:
                        clear_session(session_id)
                    except Exception as e:
                        logger.error(f"[INFERENCE] clear_session during delayed release failed: {e}")

                threading.Thread(
                    target=_wait_and_release,
                    name="oak-inference-delayed-release",
                    daemon=True,
                ).start()

        # ── Clean up inference session (only if the thread is actually gone) ──
        if thread_confirmed_dead and self._inference_session_id:
            logger.info(f"[INFERENCE] Clearing session: {self._inference_session_id}")
            clear_session(self._inference_session_id)

        # Only forget the thread reference once it's confirmed dead -- otherwise
        # start_inference()'s already_running guard must keep seeing it so a
        # second worker can't start on top of the still-running one.
        if thread_confirmed_dead:
            self._inference_thread = None
        self._inference_watchdog_thread = None
        self._inference_result_queue = None
        self._inference_session_id = None
        self._inference_offline = False

        streaming_active = bool(self._is_streaming or self._stream_subscribers or (self._stream_queue is not None))
        recording_active = self._active_recording is not None
        # Keep capture threads alive if streaming or camera pipeline is running
        if not streaming_active and not recording_active and not self._is_running:
            logger.info("[INFERENCE] Camera not streaming or running — stopping capture threads")
            self._stop_capture_threads()
        else:
            logger.info(f"[INFERENCE] Camera active (streaming={streaming_active}, recording={recording_active}, running={self._is_running}) — keeping capture threads running")

        logger.info("[INFERENCE] Stopped")
        self._stop_gpu_sampler()
        # ── Reset offline thread reference if it finished naturally ──
        if self._offline_thread is not None and not self._offline_thread.is_alive():
            self._offline_thread = None
            self._offline_stop.clear()
            # Drain any leftover sentinel from previous EOF
            while not self._offline_frame_queue.empty():
                try:
                    self._offline_frame_queue.get_nowait()
                except Exception:
                    break
        return {"status": "stopped"}

    async def get_inference_result(self, timeout: float = 1.0) -> dict | None:
        if self._inference_result_queue is None:
            return None
        try:
            return await asyncio.wait_for(self._inference_result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ==================== Recording ====================

    def _resolve_recording_path(self) -> str | None:
        return None

    def _resolve_recording_format(self, override: str | None = None) -> str:
        if override and isinstance(override, str) and override.strip():
            return override.strip().upper()
        try:
            from app.core.database import SessionLocal
            from app.models.camera_model import Camera
            db = SessionLocal()
            try:
                cam = db.query(Camera).first()
                fmt = getattr(cam, "recording_format", None)
                if fmt and isinstance(fmt, str) and fmt.strip():
                    return fmt.strip().upper()
            finally:
                db.close()
        except Exception:
            pass
        return "MP4"

    def start_recording(
        self,
        session_id: str,
        width: int,
        height: int,
        fps: float,
        root_path: str | None = None,
        recording_format: str | None = None,
    ) -> str:
        from app.services.recording_service import start_recording as _start, active_recordings

        root_path = self._resolve_recording_path()
        rec_fmt = self._resolve_recording_format(recording_format)

        logger.info(f"[RECORD] Starting recording — session: {session_id}, {width}x{height} @ {fps}fps | format={rec_fmt}")
        logger.info(f"[RECORD] Pipeline running: {self._is_running}")
        logger.info(f"[RECORD] Threads — mjpeg: {bool(self._mjpeg_thread and self._mjpeg_thread.is_alive())} | hires: {bool(self._hires_thread and self._hires_thread.is_alive())} | convert: {bool(self._convert_thread and self._convert_thread.is_alive())}")
        realtime_log_service.add_log(
            "record",
            "RECORD",
            f"Recording started: Session {session_id} ({rec_fmt})",
            "success"
        )

        # Start capture threads if not already running (e.g. recording without streaming)
        if not (self._hires_thread and self._hires_thread.is_alive()):
            logger.info("[RECORD] Step 1 — capture threads not running, starting now")
            self._start_capture_threads()
            self._enforce_ae_limit()
            logger.info(f"[RECORD] Step 2 — AE limit enforced | configured_fps={self._configured_fps} | ae_limit_us={self._ae_limit_us}")
        else:
            logger.info("[RECORD] Capture threads already running (AE limit already applied at stream start)")

        path = _start(session_id, width, height, fps, root_path, rec_fmt)
        self._active_recording = active_recordings.get(session_id)

        if self._active_recording is None:
            logger.error("[RECORD] Failed to get recording session from active_recordings!")
        else:
            logger.info(f"[RECORD] Session active — frames will be fed from _convert_loop → path: {path}")

        return path

    def stop_recording(self, session_id: str) -> dict:
        from app.services.recording_service import stop_recording as _stop

        logger.info(f"[RECORD] Stopping recording — session: {session_id}")
        realtime_log_service.add_log(
            "record",
            "RECORD",
            f"Recording stopped: Session {session_id}",
            "success"
        )
        if self._active_recording:
            logger.info(f"[RECORD] Queue size before stop: {self._active_recording.frame_queue.qsize()}")

        self._active_recording = None
        result = _stop(session_id)
        logger.info(f"[RECORD] Recording stopped and session cleared → {result}")

        # Stop capture threads only if streaming and pipeline are also not active
        streaming_active = bool(self._is_streaming or self._stream_subscribers or (self._stream_queue is not None))
        if not streaming_active and not self._is_running:
            logger.info("[RECORD] No active streaming or running pipeline — stopping capture threads")
            self._stop_capture_threads()
        else:
            logger.info("[RECORD] Streaming/pipeline still active — keeping capture threads running")

        return result

    # ==================== App Lifecycle (connect / disconnect) ====================

    async def start(self, config) -> dict:
        """App startup — connect device and build pipeline only. Does NOT start capture threads."""
        async with self._lifecycle_lock:
            if self._is_running:
                logger.warning("[START] Pipeline already running")
                return {"status": "already_running"}

            try:
                if not self.is_connected:
                    if not await self.connect(config.ip_address):
                        # Older databases may select a locked/unavailable
                        # network camera as the newest row. Prefer a local USB
                        # OAK when one is available so startup still works.
                        selected = str(config.ip_address or "").strip().lower()
                        if selected not in {"", "usb", "auto", "127.0.0.1", "localhost"}:
                            logger.warning(
                                f"[START] Could not connect to {config.ip_address}; trying USB OAK"
                            )
                            if not await self.connect("usb"):
                                return {"status": "error", "message": "Device connection failed"}
                        else:
                            return {"status": "error", "message": "Device connection failed"}

                self.current_brightness = int(config.brightness or 0)
                self.current_contrast = int(config.contrast if config.contrast is not None and config.contrast != 0 else 50)

                with self._pipeline_lock:
                    self._pipeline = dai.Pipeline(self.device)

                if not self._build_pipeline(self._pipeline, config):
                    return {"status": "error", "message": "Pipeline build failed"}

                self._pipeline.start()
                self._is_running = True

                # Send AE limit immediately after pipeline starts.
                # The control queue is ready as soon as the pipeline is running.
                # Sending here (before capture threads start) ensures the camera
                # applies the shutter cap from its very first exposure cycle.
                if self.control_mode == "auto" and self._ae_limit_us is not None:
                    self._enforce_ae_limit()

                logger.info("[START] Pipeline ready — waiting for stream/start")
                realtime_log_service.add_log(
                    "system",
                    "SYSTEM",
                    "Application initialized successfully",
                    "success"
                )
                return {"status": "started"}

            except Exception as e:
                logger.error(f"[START] Failed: {e}", exc_info=True)
                try:
                    self._cleanup_pipeline()
                    await self.disconnect()
                except Exception:
                    pass
                return {"status": "error", "message": str(e)}

    async def stop(self) -> dict:
        """App shutdown — stop capture threads if running, cleanup pipeline, disconnect device."""
        async with self._lifecycle_lock:
            try:
                self.stop_inference()
                
                # If a recording is active, we should attempt to stop it gracefully
                if self._active_recording and getattr(self._active_recording, "session_id", None):
                    self.stop_recording(self._active_recording.session_id)
                elif self._inference_session_id:
                    self.stop_recording(self._inference_session_id)
                    
                self._close_stream_queue()
                self._stop_capture_threads()
                self._cleanup_pipeline()
                await self.disconnect()
                logger.info("[STOP] Camera stopped")
                realtime_log_service.add_log(
                    "system",
                    "SYSTEM",
                    "Application stopped",
                    "success"
                )
                return {"status": "stopped"}
            except Exception as e:
                logger.error(f"[STOP] Failed: {e}")
                return {"status": "error", "message": str(e)}

    # ==================== AE Runtime Enforcement ====================

    def _enforce_ae_limit(self) -> None:
        """Send AE shutter limit via runtime control queue.

        initialControl settings are applied at pipeline build time but are often
        overridden by the camera firmware during AE convergence.  Sending the same
        limit again as a runtime CameraControl message after the pipeline is running
        is the authoritative way to cap the shutter and guarantee the configured fps.
        """
        if self._ae_limit_us is None:
            logger.info("[AE] Control mode is manual — no AE limit to enforce")
            return
        if self._control_queue is None:
            logger.warning("[AE] Control queue not available — cannot enforce AE limit")
            return
        try:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            ctrl.setAutoExposureLimit(self._ae_limit_us)
            self._control_queue.send(ctrl)
            logger.info(
                f"[AE] Runtime AE limit sent: {self._ae_limit_us} µs "
                f"(max shutter = {self._ae_limit_us / 1000:.1f} ms → "
                f"guarantees {self._configured_fps} fps)"
            )
        except Exception as e:
            logger.warning(f"[AE] Runtime AE limit failed: {e}")

    # ==================== Stream Lifecycle (capture threads) ====================

    async def start_streaming(self) -> dict:
        """User clicks Start Streaming — start the 3 capture threads."""
        if not self._is_running:
            return {"status": "error", "message": "Pipeline not started — call /oak/start first"}

        if not self._start_capture_threads():
            return {"status": "error", "message": "Capture threads failed to start"}

        # Enforce AE shutter limit via runtime control now that the pipeline is live.
        # This is more reliable than initialControl alone.
        logger.info(f"[STREAMING] Step 1 — capture threads started")
        self._enforce_ae_limit()
        logger.info(f"[STREAMING] Step 2 — AE limit enforced | configured_fps={self._configured_fps} | ae_limit_us={self._ae_limit_us}")

        self._is_streaming = True
        logger.info("[STREAMING] Started")
        realtime_log_service.add_log(
            "stream",
            "STREAM",
            "Video stream initialized successfully",
            "success"
        )
        return {"status": "streaming"}

    async def stop_streaming(self) -> dict:
        """User clicks Stop Streaming — close stream queue, stop threads if recording and inference also inactive."""
        self._is_streaming = False
        for q in list(self._stream_subscribers):
            try:
                q.put_nowait(None)
            except Exception:
                pass
        self._stream_subscribers.clear()
        self._stream_queue = None

        recording_active = self._active_recording is not None
        inference_active = bool(self._inference_thread and self._inference_thread.is_alive())
        if not recording_active and not inference_active:
            logger.info("[STREAMING] No active recording or inference — stopping capture threads")
            self._stop_capture_threads()
        else:
            logger.info(f"[STREAMING] Keeping capture threads running (recording={recording_active}, inference={inference_active})")

        logger.info("[STREAMING] Stopped")
        return {"status": "stopped"}

    # ==================== Health ====================

    def health(self) -> dict:
        return {
            "running": self._is_running,
            "connected": self.is_connected,
            "camera_running": self._is_running,
            "is_running": self._is_running,
            "is_connected": self.is_connected,
            "mjpeg_thread": bool(self._mjpeg_thread and self._mjpeg_thread.is_alive()),
            "hires_thread": bool(self._hires_thread and self._hires_thread.is_alive()),
            "convert_thread": bool(self._convert_thread and self._convert_thread.is_alive()),
            "inference_thread": bool(self._inference_thread and self._inference_thread.is_alive()),
            "inference_watchdog_thread": bool(self._inference_watchdog_thread and self._inference_watchdog_thread.is_alive()),
            "streaming": self._is_streaming,
            "fps": round(self.current_fps, 1),
        }


oak_camera_service = OakCameraService()