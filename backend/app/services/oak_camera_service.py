from app.ml.camera_inference_service import close_camera_session
import asyncio
import base64
import os
import threading
import time
from collections import deque
from datetime import timedelta
from queue import Queue, Empty

import cv2
import depthai as dai
import numpy as np
import logging
from app.services.realtime_log_service import realtime_log_service

logger = logging.getLogger("oak-camera")

# DepthAI XLink native layer is NOT thread-safe for device enumeration.
# Concurrent calls to dai.Device.getAllAvailableDevices() from multiple
# threads will corrupt internal state and crash the process with a segfault
# ("Attempt to access invalid address" in XLinkConnection).
# This lock serialises every call that touches device discovery.
_depthai_device_enum_lock = threading.Lock()

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
        self._last_device_scan_time: float = 0.0
        self._cached_device_detected: bool = False
        self._cached_devices_count: int = 0
        self._is_scanning_devices: bool = False

        # ---- Pipeline ----
        self._pipeline: dai.Pipeline | None = None
        self._is_running: bool = False
        self._pipeline_lock = threading.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._streaming_lock = asyncio.Lock()
        self._capture_lock = threading.Lock()
        self._camera_calibration_lock = threading.Lock()
        self._calibrating = False
        self._active_camera_id: int | None = None
        self._legacy_preset_migration = False

        # ---- DAI queues ----
        self._mjpeg_dai_queue = None
        self._raw_dai_queue = None
        self._control_queue = None
        self._sync_dai_queue = None
        self._sync_thread: threading.Thread | None = None
        self._sync_stop = threading.Event()

        # ---- Depth (optional stereo depth; isolated from the RGB path) ----
        # OAK_DEPTH_ENABLED=0 turns depth off completely (RGB behaves exactly as before).
        # OAK_DEPTH_ALIGN_RGB=1 aligns depth to the RGB camera (step 2, after depth looks good).
        self._depth_enabled: bool = os.getenv("OAK_DEPTH_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
        self._depth_align_rgb: bool = os.getenv("OAK_DEPTH_ALIGN_RGB", "1").strip().lower() not in {"0", "false", "no", "off"}
        # Depth resolution: If None, automatically adopts the active RGB stream resolution (e.g. 1920x1080)
        self._depth_width: int | None = int(os.getenv("OAK_DEPTH_WIDTH")) if os.getenv("OAK_DEPTH_WIDTH") else None
        self._depth_height: int | None = int(os.getenv("OAK_DEPTH_HEIGHT")) if os.getenv("OAK_DEPTH_HEIGHT") else None
        self._depth_min_mm: int = 300      # colormap preview range only, does not limit measurement
        self._depth_max_mm: int = 1500
        self._depth_dai_queue = None
        self._depth_thread: threading.Thread | None = None
        self._depth_stop = threading.Event()
        self._depth_last_view: float = 0.0   # last time a client asked for the colorized view
        self._latest_depth: np.ndarray | None = None   # uint16, millimetres, 0 = invalid
        self._latest_depth_jpeg: bytes | None = None
        self._depth_buffer: deque = deque(maxlen=30)
        self._depth_buffer_lock = threading.Lock()
        self._hires_packet_queue: Queue = Queue(maxsize=16)

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

        # ---- Latest BGR & JPEG frame caches (GIL-safe single ref assign) ----
        self._latest_bgr: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_bgr_seq: int = 0
        self._latest_frame_data: tuple[np.ndarray, int, float] | None = None  # (bgr, seq_num, capture_ts)
        self._pipeline_width: int = 1920
        self._pipeline_height: int = 1080

        # ---- Camera settings ----
        self.control_mode: str = "auto"
        self.current_brightness: int = 0
        self.current_contrast: int = 50
        self.current_fps: float = 0.0
        self._configured_fps: int = 30
        self.current_exposure_us: int = 8000  # Starter preset; tune/calibrate on the installed setup.
        self._ae_limit_us = 10000
        self.current_gain: int = 400
        self.current_focus: int | None = None
        self.focus_available: bool = False
        # AE/AF may run only during the explicit calibration action.
        self.auto_exposure_enabled: bool = False
        self.auto_focus_enabled: bool = False

        # ---- Recording ----
        self._active_recording = None  # RecordingSession instance when active

        # ---- Inference ----
        self._inference_thread: threading.Thread | None = None
        self._inference_watchdog_thread: threading.Thread | None = None
        self._inference_stop = threading.Event()
        self._inference_result_queue: asyncio.Queue | None = None
        self._inference_subscribers: set[asyncio.Queue] = set()
        self._inference_loop: asyncio.AbstractEventLoop | None = None
        self._inference_session_id: str | None = None
        self._latest_inference_result: dict | None = None

        # ---- Offline inference ----
        self._offline_thread: threading.Thread | None = None
        self._offline_stop = threading.Event()
        self._inference_offline: bool = False

        # ---- Offline multi-video tracking ----
        self._total_videos: int = 1
        self._current_video_idx: int = 0
        self._current_video_name: str = ""
        self._latest_metadata: tuple[int, int, int | None] | None = None

        # ---- Auto-calibration monitor ----
        self._monitor_last_calib_time: float = 0.0
        self._monitor_last_check: float = 0.0
        self._monitor_baseline_sharpness: float | None = None
        self._monitor_baseline_brightness: float | None = None
        self._monitor_bad_start_time: float | None = None

        # GPU sampler — runs independently, never blocks inference
        self._gpu_usage: float = -1.0
        self._gpu_sampler_thread: threading.Thread | None = None
        self._gpu_sampler_stop = threading.Event()

        # Load persisted defaults from vision_ai.db so camera always starts with saved defaults
        self._load_saved_db_settings()

    def _load_saved_db_settings(self) -> None:
        """Load persisted camera settings from vision_ai.db into memory."""
        try:
            from app.core.database import SessionLocal
            from app.models.camera_model import Camera
            db = SessionLocal()
            try:
                cam = db.query(Camera).filter(Camera.id == self._active_camera_id).first() if self._active_camera_id else None
                cam = cam or db.query(Camera).order_by(Camera.id.desc()).first()
                if cam:
                    if cam.control_mode:
                        self.control_mode = cam.control_mode
                    if cam.exposure is not None:
                        exp_val = float(cam.exposure)
                        self.current_exposure_us = int(exp_val * 1000) if exp_val <= 1000 else int(exp_val)
                    if cam.gain is not None:
                        self.current_gain = int(cam.gain)
                    if cam.focus is not None:
                        self.current_focus = int(cam.focus)
                        self.focus_available = True
                    if cam.brightness is not None:
                        self.current_brightness = int(cam.brightness)
                    if cam.contrast is not None:
                        self.current_contrast = int(cam.contrast)
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"[INIT] Could not load DB camera settings: {e}")

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ==================== Device ====================

    async def connect(self, ip: str) -> bool:
        loop = asyncio.get_running_loop()
        self._last_connection_error = None
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                logger.info(f"[DEVICE] Connecting to {ip} (attempt {attempt + 1}/{max_attempts})")
                try:
                    device_info = await loop.run_in_executor(None, self._resolve_device_info, ip)
                except RuntimeError as resolve_err:
                    # No OAK device found at all — don't retry, just report cleanly
                    err_msg = str(resolve_err)
                    logger.warning(f"[DEVICE] Connection failed: {err_msg}")
                    self._last_connection_error = err_msg
                    self.is_connected = False
                    realtime_log_service.add_log(
                        "camera", "CAMERA",
                        f"No OAK device found ({ip})",
                        "error"
                    )
                    return False
                logger.info(f"[DEVICE] Resolved connection target: {device_info}")
                self.device = await loop.run_in_executor(None, dai.Device, device_info)
                self.device.setLogLevel(dai.LogLevel.WARN)
                self.device.setLogOutputLevel(dai.LogLevel.WARN)
                self.is_connected = True

                try:
                    usb_speed = self.device.getUsbSpeed()
                    logger.info(f"[DEVICE] Connected link speed: {usb_speed}")
                    if usb_speed == dai.UsbSpeed.HIGH:
                        logger.warning(
                            "[DEVICE] ⚠️ Camera connected via USB 2.0 (HIGH speed: ~35-40 MB/s). "
                            "Uncompressed 1080p stream exceeds USB 2.0 bandwidth and will cause X_LINK_ERROR. "
                            "Please connect camera to a blue USB 3.0 port using a USB 3.0 SuperSpeed cable."
                        )
                        realtime_log_service.add_log(
                            "camera",
                            "WARN",
                            "OAK connected at USB 2.0 speed! Use a USB 3.0 port and cable for stability.",
                            "warning"
                        )
                except Exception as speed_err:
                    logger.debug(f"[DEVICE] Could not determine USB speed: {speed_err}")

                logger.info(f"[DEVICE] Connected to {ip}")
                realtime_log_service.add_log(
                    "camera",
                    "CAMERA",
                    f"Camera connected ({ip})",
                    "success"
                )
                self._last_connection_error = None
                return True
            except Exception as e:
                err_str = str(e)
                if "already used" in err_str.lower() or "in use" in err_str.lower():
                    clean_msg = "Device is already used by another application/process. Make sure to close all applications/processes using the device before starting a new one."
                    self._last_connection_error = clean_msg
                    if attempt < max_attempts - 1:
                        logger.warning(f"[DEVICE] Device is reported in use, waiting 2.0s to retry (attempt {attempt + 1}/{max_attempts})...")
                        await asyncio.sleep(2.0)
                        continue
                    else:
                        logger.warning(f"[DEVICE] Connection failed after {max_attempts} attempts: {clean_msg}")
                        realtime_log_service.add_log(
                            "camera",
                            "CAMERA",
                            clean_msg,
                            "error"
                        )
                        self.is_connected = False
                        return False

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
        self._last_connection_error = self._last_connection_error or "Failed after retries"
        return False


    @staticmethod
    def _resolve_device_info(identifier: str):
        """Resolve a database value to a current USB or network DeviceInfo."""
        value = str(identifier or "").strip()
        with _depthai_device_enum_lock:
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
                await asyncio.to_thread(self.device.close)
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

    def _build_pipeline(self, pipeline: dai.Pipeline, config, enable_depth: bool = True) -> bool:
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
            self._pipeline_width = width
            self._pipeline_height = height
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

            try:
                exposure = int(config.exposure if config.exposure is not None else 8)
                exposure = exposure * 1000 if exposure <= 1000 else exposure
                gain = max(100, min(1600, int(config.gain if config.gain is not None else 400)))
                focus = int(config.focus) if config.focus is not None else None
                if control_mode == "auto" and exposure == 16000 and gain == 400 and focus == 120:
                    # These were forced by the previous implementation, not a user-tuned calibration.
                    exposure, focus = 8000, None
                    self._legacy_preset_migration = True
            except (TypeError, ValueError):
                logger.warning("[PIPELINE] Invalid saved camera values; using 8ms/ISO400 preset")
                exposure, gain, focus = 8000, 400, None

            self.current_exposure_us = max(100, min(33000, exposure))
            self.current_gain = gain
            self.current_focus = focus
            self.focus_available = focus is not None
            self.current_brightness = int(config.brightness or 0)
            self.current_contrast = int(config.contrast if config.contrast is not None else 50)
            self.auto_exposure_enabled = self.auto_focus_enabled = False
            init_ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            if focus is not None:
                init_ctrl.setManualFocus(focus)
            init_ctrl.setManualExposure(self.current_exposure_us, gain)
            hw_b = max(-10, min(10, int(round(self.current_brightness / 5.0))))
            hw_c = max(-10, min(10, int(round((self.current_contrast - 50) / 5.0))))
            init_ctrl.setBrightness(hw_b)
            init_ctrl.setContrast(hw_c)
            logger.info(f"[PIPELINE] {control_mode} fixed preset: exp={self.current_exposure_us}us, gain={gain}, focus={focus}")

            is_usb2 = False
            if self.device is not None:
                try:
                    if self.device.getUsbSpeed() == dai.UsbSpeed.HIGH:
                        is_usb2 = True
                        logger.warning(
                            "[PIPELINE] ⚠️ Detected USB 2.0 connection. "
                            "Uncompressed 1080p @ 30fps requires ~93 MB/s which exceeds USB 2.0 bandwidth (~35 MB/s). "
                            "Throttling raw frame output rate to 10 FPS to prevent XLink communication buffer overflows. "
                            "Please connect camera to a blue USB 3.0 port with a SuperSpeed cable for full 30 FPS."
                        )
                except Exception:
                    pass

            raw_fps = min(fps, 10) if is_usb2 else fps

            # --- Depth node setup ---
            stereo_node = None
            if enable_depth and self._depth_enabled:
                stereo_node = self._build_depth(pipeline, width, height)

            # --- Unified Hardware Sync (CAM_A RGB + StereoDepth) ---
            self._sync_dai_queue = None
            if stereo_node is not None:
                try:
                    sync = pipeline.create(dai.node.Sync)
                    sync.setSyncThreshold(timedelta(milliseconds=25))

                    if not is_color_camera:
                        rgb_cam_out = cam.requestOutput((width, height), type=dai.ImgFrame.Type.NV12, fps=fps)
                    else:
                        rgb_cam_out = cam.video

                    rgb_cam_out.link(sync.inputs["rgb"])
                    stereo_node.depth.link(sync.inputs["depth"])

                    self._sync_dai_queue = sync.out.createOutputQueue(maxSize=4, blocking=False)
                    self._depth_dai_queue = self._sync_dai_queue
                    logger.info(f"[PIPELINE] Unified Hardware Sync Node ready ({width}x{height} @ {fps}fps, threshold=25ms)")
                except Exception as sync_err:
                    logger.warning(f"[PIPELINE] Sync node wiring failed ({sync_err}), falling back to direct queues")
                    self._sync_dai_queue = None

            # Fallback direct queues if Sync node is not used (e.g. depth disabled or sync failed)
            if self._sync_dai_queue is None:
                encoder = pipeline.create(dai.node.VideoEncoder)
                encoder.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.MJPEG)
                if not is_color_camera:
                    mjpeg_out = cam.requestOutput(
                        (width, height), type=dai.ImgFrame.Type.NV12, fps=fps
                    )
                    mjpeg_out.link(encoder.input)
                    self._mjpeg_dai_queue = encoder.bitstream.createOutputQueue(maxSize=4, blocking=False)

                    raw_out = cam.requestOutput(
                        (width, height), type=dai.ImgFrame.Type.NV12, fps=raw_fps
                    )
                    self._raw_dai_queue = raw_out.createOutputQueue(maxSize=2, blocking=False)
                    logger.info(f"[PIPELINE] Fallback direct queues wired (mjpeg + raw, fps={raw_fps})")
                else:
                    cam.video.link(encoder.input)
                    self._mjpeg_dai_queue = encoder.bitstream.createOutputQueue(maxSize=4, blocking=False)
                    self._raw_dai_queue = cam.video.createOutputQueue(maxSize=2, blocking=False)
                if stereo_node is not None:
                    self._depth_dai_queue = stereo_node.depth.createOutputQueue(maxSize=4, blocking=False)

            # --- Camera control input ---
            self._control_queue = cam.inputControl.createInputQueue(maxSize=4, blocking=False)

            logger.info("[PIPELINE] Build success")
            return True

        except Exception as e:
            logger.error(f"[PIPELINE] Build failed: {e}", exc_info=True)
            return False

    # ==================== Depth ====================

    def _build_depth(self, pipeline: dai.Pipeline, rgb_width: int = 1920, rgb_height: int = 1080):
        """Add left/right mono cameras + StereoDepth to the SAME pipeline as RGB.
        Automatically aligns to and outputs at the active RGB stream resolution (e.g. 1920x1080 or 1280x720).
        Returns the StereoDepth node on success, or None on failure.
        """
        # 1. Output resolution: Automatically matches the active RGB stream
        out_w = self._depth_width or rgb_width
        out_h = self._depth_height or rgb_height

        # 2. Input sensor mode: OV9282 physical sensors on OAK-D
        aspect = out_w / max(out_h, 1)
        if abs(aspect - 16 / 9) < 0.15:
            mono_w, mono_h = 1280, 720   # 16:9 native mode matching 720p/1080p
        elif out_w <= 640 or out_h <= 400:
            mono_w, mono_h = 640, 400    # 2x2 binned mode
        else:
            mono_w, mono_h = 1280, 800   # 16:10 full sensor mode

        stereo = pipeline.create(dai.node.StereoDepth)
        if hasattr(dai.node.StereoDepth, "PresetMode") and hasattr(dai.node.StereoDepth.PresetMode, "HIGH_DETAIL"):
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
        elif hasattr(dai.node.StereoDepth, "PresetMode") and hasattr(dai.node.StereoDepth.PresetMode, "DEFAULT"):
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

        stereo.setRectification(True)
        stereo.setLeftRightCheck(True)      # removes occlusion errors
        stereo.setSubpixel(True)            # subpixel accuracy for close-range tubing

        # 3. Capture mono stereo pair from physical sensors
        stereo_wired = False
        try:
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            left.requestOutput((mono_w, mono_h)).link(stereo.left)
            right.requestOutput((mono_w, mono_h)).link(stereo.right)
            stereo_wired = True
            logger.info(f"[PIPELINE] Stereo cameras created with unified dai.node.Camera ({mono_w}x{mono_h})")
        except Exception as cam_err:
            logger.info(f"[PIPELINE] dai.node.Camera stereo build unavailable ({cam_err}), attempting MonoCamera fallback...")

        # Fallback to dai.node.MonoCamera (classic OAK-D / OAK-D Lite / OAK-D Pro)
        if not stereo_wired:
            try:
                left = pipeline.create(dai.node.MonoCamera)
                left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
                if hasattr(dai.MonoCameraProperties, "SensorResolution"):
                    res = dai.MonoCameraProperties.SensorResolution.THE_720_P if mono_h == 720 else (
                        dai.MonoCameraProperties.SensorResolution.THE_800_P if mono_h == 800 else dai.MonoCameraProperties.SensorResolution.THE_400_P
                    )
                    left.setResolution(res)

                right = pipeline.create(dai.node.MonoCamera)
                right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
                if hasattr(dai.MonoCameraProperties, "SensorResolution"):
                    res = dai.MonoCameraProperties.SensorResolution.THE_720_P if mono_h == 720 else (
                        dai.MonoCameraProperties.SensorResolution.THE_800_P if mono_h == 800 else dai.MonoCameraProperties.SensorResolution.THE_400_P
                    )
                    right.setResolution(res)

                left.out.link(stereo.left)
                right.out.link(stereo.right)
                stereo_wired = True
                logger.info(f"[PIPELINE] Stereo cameras created with dai.node.MonoCamera fallback ({mono_w}x{mono_h})")
            except Exception as mono_err:
                logger.warning(f"[PIPELINE] Stereo cameras failed to initialize: {mono_err}")
                return None

        # 4. Align depth pixel-for-pixel to RGB camera (CAM_A)
        if self._depth_align_rgb:
            try:
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                logger.info("[PIPELINE] StereoDepth aligned to CAM_A (RGB)")
            except Exception as align_err:
                logger.warning(f"[PIPELINE] Stereo setDepthAlign to CAM_A failed: {align_err}")

            try:
                stereo.setOutputSize(out_w, out_h)
                stereo.setOutputKeepAspectRatio(False)
                logger.info(f"[PIPELINE] StereoDepth output size set to {out_w}x{out_h}")
            except Exception as size_err:
                logger.debug(f"[PIPELINE] Stereo setOutputSize not supported on this platform ({size_err})")

        logger.info(f"[PIPELINE] Stereo depth configured ({out_w}x{out_h}, mono_sensor={mono_w}x{mono_h}, align_rgb={self._depth_align_rgb})")
        return stereo

    def depth_available(self) -> bool:
        return bool(self._depth_dai_queue is not None and self._is_running and self.is_connected)

    def get_latest_depth_frame(self) -> np.ndarray | None:
        """Raw uint16 depth frame in millimeters (0 = invalid)."""
        return self._latest_depth

    def get_synchronized_depth(self, target_hw_ts: float, max_delta_s: float = 0.08) -> np.ndarray | None:
        """Retrieve the exact depth frame matching the hardware capture timestamp of an RGB frame.
        Both CAM_A and stereo cameras share the on-device hardware timestamp clock.
        """
        if not self._depth_enabled:
            return None
        with self._depth_buffer_lock:
            if not self._depth_buffer:
                return self._latest_depth
            best_diff = float("inf")
            best_depth = None
            for d_ts, d_seq, d_frame in self._depth_buffer:
                diff = abs(d_ts - target_hw_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_depth = d_frame
            if best_depth is not None and best_diff <= max_delta_s:
                return best_depth
            return best_depth if best_depth is not None else self._latest_depth

    def get_depth_jpeg(self) -> bytes | None:
        """Latest colorized depth JPEG. Calling this marks the depth view as 'being watched',
        so colorizing/encoding only costs CPU while someone is actually looking."""
        self._depth_last_view = time.time()
        if self._latest_depth_jpeg is None and self._latest_depth is not None:
            self._latest_depth_jpeg = self._render_depth_jpeg(self._latest_depth)
        return self._latest_depth_jpeg

    def get_depth_telemetry(self) -> dict | None:
        """Instant statistical summary of the latest depth frame (center, valid %, range)."""
        depth = self._latest_depth
        if depth is None:
            return None
        h, w = depth.shape[:2]
        valid_all = depth > 0
        valid_pct = round(100.0 * float(valid_all.mean()), 1)
        cy, cx = h // 2, w // 2
        cpatch = depth[max(0, cy - 10):cy + 11, max(0, cx - 10):cx + 11]
        cvalid = cpatch[cpatch > 0]
        center_z_mm = float(np.median(cvalid)) if cvalid.size else None
        return {
            "depth_width": int(w),
            "depth_height": int(h),
            "valid_pct": valid_pct,
            "center_z_mm": center_z_mm,
            "colormap_min_mm": self._depth_min_mm,
            "colormap_max_mm": self._depth_max_mm,
            "aligned_to_rgb": self._depth_align_rgb,
        }

    def get_depth_at(self, nx: float, ny: float, window: int = 7) -> dict | None:
        """Median depth (mm) around a normalized position (0..1, 0..1) of the depth frame."""
        depth = self._latest_depth
        if depth is None:
            return None
        h, w = depth.shape[:2]
        u = int(max(0, min(w - 1, round(nx * (w - 1)))))
        v = int(max(0, min(h - 1, round(ny * (h - 1)))))
        r = max(1, window) // 2
        patch = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        valid = patch[patch > 0]

        # Overall frame telemetry included so probe responses also refresh stream stats
        valid_all = depth > 0
        valid_pct = round(100.0 * float(valid_all.mean()), 1)
        cy, cx = h // 2, w // 2
        cpatch = depth[max(0, cy - 10):cy + 11, max(0, cx - 10):cx + 11]
        cvalid = cpatch[cpatch > 0]
        center_z_mm = float(np.median(cvalid)) if cvalid.size else None

        return {
            "x_px": u,
            "y_px": v,
            "depth_width": int(w),
            "depth_height": int(h),
            "z_mm": float(np.median(valid)) if valid.size else None,
            "valid_ratio": round(valid.size / max(patch.size, 1), 3),
            "valid_pct": valid_pct,
            "center_z_mm": center_z_mm,
            "colormap_min_mm": self._depth_min_mm,
            "colormap_max_mm": self._depth_max_mm,
            "window": int(window),
            "aligned_to_rgb": self._depth_align_rgb,
        }

    def _render_depth_jpeg(self, depth: np.ndarray) -> bytes | None:
        """Colorize uint16 mm depth (invalid = black) with center crosshair.
        All telemetry numbers are rendered by the frontend HUD at bottom-left without overlapping."""
        h, w = depth.shape[:2]
        valid = depth > 0

        lo, hi = self._depth_min_mm, self._depth_max_mm
        norm = (np.clip(depth, lo, hi).astype(np.float32) - lo) * (255.0 / max(hi - lo, 1))
        color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
        color[~valid] = 0

        cy, cx = h // 2, w // 2
        cv2.drawMarker(color, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

        ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def _depth_loop(self) -> None:
        """Drains the depth queue. Errors here are logged only - they never tear down RGB."""
        logger.info("[DEPTH] Thread running")
        frames = 0
        last_log = time.time()
        try:
            while not self._depth_stop.is_set():
                q = self._depth_dai_queue
                if q is None:
                    time.sleep(0.01)
                    continue
                pkt = q.tryGet()
                if pkt is None:
                    time.sleep(0.005)
                    continue

                depth = pkt.getFrame()           # uint16, millimetres
                self._latest_depth = depth       # GIL-safe single reference assignment
                try:
                    d_ts = pkt.getTimestamp().total_seconds()
                except Exception:
                    d_ts = time.time()
                try:
                    d_seq = pkt.getSequenceNum()
                except Exception:
                    d_seq = 0

                with self._depth_buffer_lock:
                    self._depth_buffer.append((d_ts, d_seq, depth))

                frames += 1

                if time.time() - self._depth_last_view < 3.0:
                    jpeg = self._render_depth_jpeg(depth)
                    if jpeg is not None:
                        self._latest_depth_jpeg = jpeg

                now = time.time()
                if now - last_log >= 5.0:
                    valid_pct = 100.0 * float((depth > 0).mean())
                    logger.info(f"[DEPTH] FPS: {frames / (now - last_log):.1f} | shape={depth.shape} | valid={valid_pct:.0f}%")
                    frames = 0
                    last_log = now
        except Exception as e:
            if not self._depth_stop.is_set():
                logger.warning(f"[DEPTH] Thread stopped: {e}")
        finally:
            logger.info("[DEPTH] Thread ended")

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
                self._sync_dai_queue = None
                self._control_queue = None
                self._latest_jpeg = None
                self._latest_bgr = None
                self._depth_dai_queue = None
                self._latest_depth = None
                self._latest_depth_jpeg = None
                with self._depth_buffer_lock:
                    self._depth_buffer.clear()
            logger.info("[PIPELINE] Cleaned up")

    def _on_device_lost(self, reason: str = "Device disconnected") -> None:
        """Central teardown when DepthAI device disconnects, XLink crashes, or queues close."""
        with self._pipeline_lock:
            if not self._is_running and not self.is_connected:
                return
            logger.warning(f"[DEVICE] Teardown triggered: {reason}")
            self._is_running = False
            self.is_connected = False
            self._is_streaming = False

            # Signal capture & inference loops to stop
            self._mjpeg_stop.set()
            self._hires_stop.set()
            self._convert_stop.set()
            self._depth_stop.set()
            self._inference_stop.set()

            # Invalidate queues so callers know device is unavailable
            self._mjpeg_dai_queue = None
            self._raw_dai_queue = None
            self._control_queue = None
            self._depth_dai_queue = None
            self._pipeline = None
            self._latest_jpeg = None
            self._latest_bgr = None

            # Wake up any streaming client queues with None sentinel so generators exit cleanly (thread-safe)
            loop = self._server_loop
            for q in list(self._stream_subscribers):
                if loop is not None and not loop.is_closed():
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, None)
                    except Exception:
                        pass
                else:
                    try:
                        q.put_nowait(None)
                    except Exception:
                        pass
            self._stream_subscribers.clear()
            self._stream_queue = None

            # If inference was active, push terminal error so frontend WebSocket knows immediately
            if self._inference_result_queue is not None and self._inference_loop is not None and not self._inference_loop.is_closed():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._async_inference_push({
                            "status": "error",
                            "error": f"Camera disconnected: {reason}",
                            "reasons": [f"Camera disconnected: {reason}"],
                            "done": True,
                        }),
                        self._inference_loop,
                    )
                except Exception:
                    pass

            if self._inference_session_id:
                try:
                    from app.ml.camera_inference_service import close_camera_session
                    close_camera_session(self._inference_session_id)
                except Exception as close_err:
                    logger.debug(f"[CAMERA] Error closing session on device lost: {close_err}")

            self._inference_result_queue = None
            self._inference_thread = None
            self._inference_watchdog_thread = None
            self._inference_session_id = None

            # Close device handle safely if not already closed
            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass
                self.device = None

            realtime_log_service.add_log(
                "camera",
                "CAMERA",
                f"Camera device disconnected: {reason}",
                "error"
            )

    

    # ── New method: background GPU sampler ───────────────────────────────────────
    def _start_gpu_sampler(self) -> None:
        """Samples GPU every 2s in background — uses pynvml with fallback, never blocks inference."""
        if self._gpu_sampler_thread and self._gpu_sampler_thread.is_alive():
            return

        def _loop():
            import subprocess
            nvml_handle = None
            try:
                import pynvml
                pynvml.nvmlInit()
                nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                nvml_handle = None

            while not self._gpu_sampler_stop.is_set():
                if nvml_handle is not None:
                    try:
                        import pynvml
                        rates = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                        self._gpu_usage = float(rates.gpu)
                    except Exception:
                        self._gpu_usage = -1.0
                else:
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
        with self._capture_lock:
            if not self._is_running or not self.is_connected:
                logger.warning("[CAPTURE] Cannot start — pipeline or device queue not running")
                return False

            # --- Priority 1: Unified Sync Thread (RGB + Depth paired) ---
            if self._sync_dai_queue is not None:
                sync_alive = self._sync_thread and self._sync_thread.is_alive()
                if not sync_alive:
                    self._sync_stop.clear()
                    self._sync_thread = threading.Thread(
                        target=self._sync_capture_loop, name="oak-sync-capture", daemon=True
                    )
                    self._sync_thread.start()
                    logger.info("[CAPTURE] Unified RGB+Depth Sync capture thread started")
                return True

            # --- Priority 2: Fallback separate threads (e.g. RGB only fallback) ---
            if self._mjpeg_dai_queue is None:
                logger.warning("[CAPTURE] Cannot start — no queue running")
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

            depth_alive = self._depth_thread and self._depth_thread.is_alive()
            if self._depth_dai_queue is not None and not depth_alive and self._depth_dai_queue is not self._sync_dai_queue:
                self._depth_stop.clear()
                self._depth_thread = threading.Thread(
                    target=self._depth_loop, name="oak-depth", daemon=True
                )
                self._depth_thread.start()
                logger.info("[CAPTURE] Depth thread started")

            return True

    def _stop_capture_threads(self) -> None:
        with self._capture_lock:
            logger.info("[CAPTURE] Stopping threads...")

            # Stop Sync thread
            self._sync_stop.set()
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=2.5)
            self._sync_thread = None

            # Signal legacy fallback threads
            self._mjpeg_stop.set()
            self._hires_stop.set()
            self._convert_stop.set()

            # Join converter first — it consumes from hires_packet_queue.
            # If we join producers first they block trying to put into a full queue.
            if self._convert_thread and self._convert_thread.is_alive():
                self._convert_thread.join(timeout=3.0)
            self._convert_thread = None

            if self._mjpeg_thread and self._mjpeg_thread.is_alive():
                self._mjpeg_thread.join(timeout=2.0)
            self._mjpeg_thread = None

            if self._hires_thread and self._hires_thread.is_alive():
                self._hires_thread.join(timeout=2.0)
            self._hires_thread = None

            self._depth_stop.set()
            if self._depth_thread and self._depth_thread.is_alive():
                self._depth_thread.join(timeout=2.0)
            self._depth_thread = None
            self._latest_depth = None
            self._latest_depth_jpeg = None
            with self._depth_buffer_lock:
                self._depth_buffer.clear()

            self._latest_bgr = None
            self._latest_bgr_seq = -1
            self._latest_frame_data = None
            self._latest_jpeg = None
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

    def _sync_capture_loop(self) -> None:
        """Unified capture loop draining the DepthAI Sync queue (RGB + Depth paired).
        Feeds live stream, depth stream, and AI inference from the exact same frames.
        """
        logger.info("[SYNC-CAPTURE] Loop started")
        q = self._sync_dai_queue
        seq = 0
        frames_count = 0
        last_log = time.time()

        try:
            while not self._sync_stop.is_set() and self._is_running:
                if q is None:
                    break
                try:
                    group = q.tryGet()
                except Exception as e:
                    logger.error(f"[SYNC-CAPTURE] tryGet failed: {e}")
                    break

                if group is None:
                    time.sleep(0.001)
                    continue

                rgb_pkt = None
                try:
                    rgb_pkt = group["rgb"]
                except Exception:
                    rgb_pkt = None

                depth_pkt = None
                try:
                    depth_pkt = group["depth"]
                except Exception:
                    depth_pkt = None

                if rgb_pkt is None:
                    continue

                # 1. RGB Frame conversion
                try:
                    bgr_frame = rgb_pkt.getCvFrame()
                except Exception as cv_err:
                    logger.error(f"[SYNC-CAPTURE] getCvFrame failed: {cv_err}")
                    continue

                # 2. Depth Frame extraction & shape verification
                depth_frame = None
                if depth_pkt is not None:
                    try:
                        depth_frame = depth_pkt.getFrame()  # uint16 mm
                        if depth_frame is not None and depth_frame.shape[:2] != bgr_frame.shape[:2]:
                            depth_frame = cv2.resize(
                                depth_frame,
                                (bgr_frame.shape[1], bgr_frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                    except Exception as d_err:
                        logger.debug(f"[SYNC-CAPTURE] Depth getFrame error: {d_err}")

                seq += 1
                frames_count += 1
                now = time.time()
                hw_ts = rgb_pkt.getTimestamp().total_seconds() if hasattr(rgb_pkt, "getTimestamp") else now

                # 3. Atomic state update
                with self._capture_lock:
                    self._latest_bgr = bgr_frame
                    self._latest_bgr_seq = seq
                    self._latest_depth = depth_frame
                    self._latest_frame_data = (bgr_frame, depth_frame, seq, now, hw_ts)

                # 4. Stream preview encoding (MJPEG for RAW view in browser)
                has_subscribers = len(self._stream_subscribers) > 0 or self._stream_queue is not None
                if has_subscribers:
                    ok, jpeg_buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        jpeg_bytes = jpeg_buf.tobytes()
                        self._latest_jpeg = jpeg_bytes
                        loop = self._server_loop
                        if loop is not None and not loop.is_closed():
                            try:
                                loop.call_soon_threadsafe(self._push_sync, jpeg_bytes)
                            except Exception:
                                pass

                # 5. Depth buffer and preview rendering
                if depth_frame is not None:
                    with self._depth_buffer_lock:
                        self._depth_buffer.append((hw_ts, seq, depth_frame))
                    if (now - self._depth_last_view < 5.0) or self._latest_depth_jpeg is None:
                        self._latest_depth_jpeg = self._render_depth_jpeg(depth_frame)

                # Periodic telemetry logging
                if now - last_log >= 5.0:
                    elapsed = now - last_log
                    real_fps = frames_count / max(elapsed, 0.001)
                    val_pct = 100.0 * float((depth_frame > 0).mean()) if depth_frame is not None else 0.0
                    logger.info(f"[SYNC-CAPTURE] FPS: {real_fps:.1f} | Shape: {bgr_frame.shape} | Depth valid: {val_pct:.0f}%")
                    frames_count = 0
                    last_log = now

        except Exception as e:
            if not self._sync_stop.is_set():
                logger.error(f"[SYNC-CAPTURE] Thread error: {e}", exc_info=True)
                self._on_device_lost(f"Sync capture error: {e}")
        finally:
            logger.info("[SYNC-CAPTURE] Thread ended")

    # ---- Thread 1: MJPEG consumer ----

    def _mjpeg_loop(self) -> None:
        """Drains MJPEG DAI queue and pushes JPEG bytes into the asyncio stream_queue."""
        logger.info("[MJPEG] Thread running")
        frames = 0
        frames_pushed = 0
        frames_dropped = 0
        last_log = time.time()
        last_frame_ts = time.time()

        try:
            while not self._mjpeg_stop.is_set():
                if self._mjpeg_dai_queue is None:
                    time.sleep(0.001)
                    continue

                pkt = self._mjpeg_dai_queue.tryGet()
                if pkt is None:
                    # Watchdog: ONLY monitor frozen camera if streaming is actively desired with subscribers,
                    # and allow 15.0s to accommodate DepthAI hardware auto-reconnect (10s internal timeout).
                    # Never trigger watchdog teardown when stream is stopped or has 0 subscribers!
                    has_subscribers = bool(self._stream_subscribers) or (self._stream_queue is not None)
                    if (
                        not self._mjpeg_stop.is_set()
                        and self._is_streaming
                        and has_subscribers
                        and frames > 0
                        and (time.time() - last_frame_ts > 15.0)
                    ):
                        logger.warning("[MJPEG] Watchdog: No MJPEG frames from camera for 15.0s during active stream — device connection frozen")
                        self._on_device_lost("No MJPEG frames from camera for 15.0s")
                        break
                    time.sleep(0.005)
                    continue

                last_frame_ts = time.time()
                jpeg = bytes(pkt.getData())
                self._latest_jpeg = jpeg
                frames += 1

                # Push to all subscribed client queues thread-safely
                # Keep stream/event loop errors completely isolated from camera hardware errors!
                has_subscribers = bool(self._stream_subscribers) or (self._stream_queue is not None)
                loop = self._server_loop
                if has_subscribers and loop is not None and not loop.is_closed():
                    try:
                        loop.call_soon_threadsafe(self._push_sync, jpeg)
                        frames_pushed += 1
                    except RuntimeError:
                        # Event loop closed / server reload — do NOT treat as camera device failure!
                        pass
                    except Exception as push_err:
                        logger.debug(f"[MJPEG] Stream push error: {push_err}")
                else:
                    frames_dropped += 1

                now = time.time()
                if now - last_log >= 5.0:
                    elapsed = now - last_log
                    sub_count = len(self._stream_subscribers)
                    fps_val = frames / max(elapsed, 0.001)
                    bitrate_mbps = (frames_pushed * len(jpeg) * 8) / (max(elapsed, 0.001) * 1_000_000)
                    logger.info(
                        f"[MJPEG] FPS: {fps_val:.1f} | "
                        f"bitrate: {bitrate_mbps:.1f} Mbps | "
                        f"pushed: {frames_pushed} | "
                        f"dropped (no client): {frames_dropped} | "
                        f"subscribers: {sub_count}"
                    )
                    realtime_log_service.add_log(
                        "stream",
                        "NETWORK",
                        f"Live Stream: {fps_val:.1f} FPS, {bitrate_mbps:.1f} Mbps",
                        "info"
                    )
                    frames = 0
                    frames_pushed = 0
                    frames_dropped = 0
                    last_log = now

        except Exception as e:
            if not self._mjpeg_stop.is_set():
                err_msg = str(e)
                if "closed" in err_msg.lower() or "xlink" in err_msg.lower():
                    logger.warning(f"[MJPEG] Device connection lost: {err_msg}")
                else:
                    logger.error(f"[MJPEG] Thread error: {e}", exc_info=True)
                self._on_device_lost(f"MJPEG queue closed: {err_msg}")
                realtime_log_service.add_log(
                    "stream",
                    "NETWORK",
                    "MJPEG Thread disconnected",
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
                    time.sleep(0.005)
                    continue

                try:
                    exp = int(pkt.getExposureTime().total_seconds() * 1_000_000)
                    iso = int(pkt.getSensitivity())
                    lens = int(pkt.getLensPosition())
                    self._latest_metadata = (exp, iso, lens if 0 <= lens <= 255 else None)
                except Exception:
                    pass

                # Zero-lag: Drop unconsumed frames so converter always converts the freshest frame
                while not self._hires_packet_queue.empty():
                    try:
                        self._hires_packet_queue.get_nowait()
                    except Exception:
                        break
                try:
                    self._hires_packet_queue.put_nowait(pkt)
                except Exception:
                    pass

        except Exception as e:
            if not self._hires_stop.is_set():
                err_msg = str(e)
                if "closed" in err_msg.lower() or "xlink" in err_msg.lower():
                    logger.warning(f"[HIRES] Device connection lost: {err_msg}")
                else:
                    logger.error(f"[HIRES] Thread error: {e}", exc_info=True)
                self._on_device_lost(f"Hi-res queue closed: {err_msg}")
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
                # Drain queue to only process the newest available packet
                pkt = None
                while not self._hires_packet_queue.empty():
                    try:
                        pkt = self._hires_packet_queue.get_nowait()
                    except Empty:
                        break
                if pkt is None:
                    try:
                        pkt = self._hires_packet_queue.get(timeout=0.05)
                    except Empty:
                        continue

                try:
                    raw_bgr = pkt.getCvFrame()
                except Exception as e:
                    logger.warning(f"[CONVERT] getCvFrame failed: {e}")
                    continue

                seq_num = self._latest_bgr_seq + 1
                ts_sec = time.time()
                try:
                    seq_num = pkt.getSequenceNum()
                except Exception:
                    pass
                try:
                    hw_ts = pkt.getTimestamp().total_seconds()
                except Exception:
                    hw_ts = ts_sec

                if not _first_frame_logged:
                    logger.info(f"[CONVERT] First frame received — shape={raw_bgr.shape} | ae_limit_us={self._ae_limit_us} | configured_fps={self._configured_fps}")
                    _first_frame_logged = True

                # Hardware ISP applies brightness and contrast directly to both MJPEG and NV12 outputs
                display_bgr = raw_bgr

                if self.control_mode == "auto" and not self._calibrating:
                    self._run_monitor(display_bgr)
                elif self.control_mode == "manual":
                    self._monitor_baseline_sharpness = None

                self._latest_bgr = display_bgr  # GIL-safe single reference assignment
                self._latest_bgr_seq = seq_num
                self._latest_frame_data = (display_bgr, seq_num, ts_sec, hw_ts)  # Overwrites with newest camera frame

                if self._active_recording is not None:
                    # Save the same adjusted image shown in the live stream.
                    self._active_recording.add_frame(display_bgr)
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

    def _push_sync(self, jpeg: bytes) -> None:
        """Runs on the asyncio event loop thread — drop-oldest push into all client queues.
        Invoked via loop.call_soon_threadsafe so no Futures/Tasks are created.
        """
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

    async def _async_stream_push(self, jpeg: bytes) -> None:
        """Async fallback for pushing into client queues."""
        self._push_sync(jpeg)

    def subscribe_stream(self) -> asyncio.Queue:
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

        # Bugfix: Point _stream_queue to an existing active subscriber, not a dead queue!
        if self._stream_subscribers:
            self._stream_queue = next(iter(self._stream_subscribers))
        else:
            self._stream_queue = None
        logger.info(f"[STREAM] Client unsubscribed — remaining subscribers: {len(self._stream_subscribers)}")

    def _open_stream_queue(self) -> asyncio.Queue:
        return self.subscribe_stream()

    def _close_stream_queue(self, q: asyncio.Queue | None = None) -> None:
        self.unsubscribe_stream(q)

    async def get_stream_frame(self, timeout: float = 1.0) -> bytes | None:
        if self._latest_jpeg is not None:
            return self._latest_jpeg
        if self._stream_queue is not None:
            try:
                return await asyncio.wait_for(self._stream_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
        if self._latest_bgr is not None:
            try:
                frame_copy = self._latest_bgr.copy()
                loop = asyncio.get_running_loop()
                def _encode():
                    ret, buf = cv2.imencode(".jpg", frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    return buf.tobytes() if ret else None
                return await loop.run_in_executor(None, _encode)
            except Exception:
                pass
        return None

    # ==================== Frame Access ====================

    def get_bgr_frame(self) -> np.ndarray | None:
        """Latest BGR frame for inference / recording."""
        frame = self._latest_bgr
        return frame.copy() if frame is not None else None

    # ==================== Auto-calibration Monitor ====================

    def _safe_calibrate(self) -> None:
        try:
            logger.info("[MONITOR] Starting automatic calibration")
            self.calibrate_auto_controls()
        except Exception as e:
            logger.error(f"[MONITOR] Auto-calibration failed: {e}")

    def _run_monitor(self, frame: np.ndarray) -> None:
        # If live inference is actively running, never trigger auto-calibration shifts
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return

        now = time.time()
        if now - self._monitor_last_calib_time < 5.0 or now - self._monitor_last_check < 0.5:
            return
        
        self._monitor_last_check = now

        # crop center for speed and relevance (50% of center)
        h, w = frame.shape[:2]
        crop = frame[h//4:3*h//4, w//4:3*w//4]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
        brightness = gray.mean()

        if self._monitor_baseline_sharpness is None:
            self._monitor_baseline_sharpness = sharpness
            self._monitor_baseline_brightness = brightness
            self._monitor_bad_start_time = None
            logger.info(f"[MONITOR] Baseline set - Sharpness: {sharpness:.1f}, Brightness: {brightness:.1f}")
            return

        # Check thresholds
        sharpness_drop = sharpness < (self._monitor_baseline_sharpness * 0.5)
        # 30.0 shift in brightness out of 255 (about 12%)
        brightness_shift = abs(brightness - self._monitor_baseline_brightness) > 30.0

        if sharpness_drop or brightness_shift:
            if self._monitor_bad_start_time is None:
                self._monitor_bad_start_time = now
            elif now - self._monitor_bad_start_time > 1.0:
                logger.info(f"[MONITOR] Triggering auto-calibration. Sharpness: {sharpness:.1f} (baseline: {self._monitor_baseline_sharpness:.1f}), Brightness: {brightness:.1f} (baseline: {self._monitor_baseline_brightness:.1f})")
                self._monitor_last_calib_time = now
                self._monitor_baseline_sharpness = None
                self._monitor_bad_start_time = None
                threading.Thread(target=self._safe_calibrate, name="oak-monitor-calib", daemon=True).start()
        else:
            self._monitor_bad_start_time = None

    # ==================== Camera Controls ====================

    def reset_controls(self) -> dict:
        """Reapply the saved default preset from database."""
        self._load_saved_db_settings()
        self.auto_exposure_enabled = self.auto_focus_enabled = False
        self._monitor_baseline_sharpness = None
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        if self.current_focus is not None:
            ctrl.setManualFocus(self.current_focus)
        ctrl.setManualExposure(self.current_exposure_us, self.current_gain)
        hw_b = max(-10, min(10, int(round(self.current_brightness / 5.0))))
        hw_c = max(-10, min(10, int(round((self.current_contrast - 50) / 5.0))))
        ctrl.setBrightness(hw_b)
        ctrl.setContrast(hw_c)
        try:
            if self._control_queue is not None and self._is_running and self.is_connected:
                self._control_queue.send(ctrl)
        except Exception as e:
            logger.warning(f"[CONTROL] Preset apply failed: {e}")
            return {"status": "error", "message": str(e)}
        return {"status": "ok", **self._camera_settings()}

    def update_controls(
        self, exposure: int | None = None, gain: int | None = None,
        focus: int | None = None, brightness: int | None = None,
        contrast: int | None = None, auto_focus: bool | None = None,
        auto_exposure: bool | None = None, control_mode: str | None = None,
        reset: bool | None = None,
    ) -> None:
        """Apply fixed user or saved preset values. Legacy AE/AF flags are ignored."""
        if reset:
            self.reset_controls()
            return
        requested_mode = (control_mode or "").lower()
        if requested_mode in {"auto", "manual"}:
            self.control_mode = requested_mode
        if requested_mode == "manual" or (not requested_mode and any(v is not None for v in (exposure, gain, focus))):
            self.control_mode = "manual"
        if brightness is not None:
            self.current_brightness = max(-50, min(50, int(brightness)))
        if contrast is not None:
            self.current_contrast = max(0, min(100, int(contrast)))
        if gain is not None:
            self.current_gain = max(100, min(1600, int(gain)))
        if focus is not None:
            self.current_focus = max(0, min(255, int(focus)))
            self.focus_available = True
        if exposure is not None:
            value = int(exposure)
            exp_us = value * 1000 if value <= 1000 else value
            max_us = max(1000, int(1_000_000 / max(self._configured_fps, 1)))
            self.current_exposure_us = max(100, min(max_us, exp_us))
        self.auto_exposure_enabled = self.auto_focus_enabled = False
        self._monitor_baseline_sharpness = None
        if not self._is_running or not self.is_connected or self._control_queue is None:
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        if self.current_focus is not None:
            ctrl.setManualFocus(self.current_focus)
        ctrl.setManualExposure(self.current_exposure_us, self.current_gain)
        ctrl.setBrightness(max(-10, min(10, int(round(self.current_brightness / 5.0)))))
        ctrl.setContrast(max(-10, min(10, int(round((self.current_contrast - 50) / 5.0)))))
        try:
            self._control_queue.send(ctrl)
        except Exception as ctrl_err:
            logger.warning(f"[CONTROL] Control queue send failed: {ctrl_err}")
        if self._active_recording is not None:
            self._active_recording.add_settings_event(self._recording_settings_snapshot())

    def _camera_settings(self) -> dict:
        return {
            "control_mode": self.control_mode,
            "exposure": round(self.current_exposure_us / 1000, 3),
            "gain": self.current_gain,
            "focus": self.current_focus,
            "focus_available": self.focus_available,
            "brightness": self.current_brightness,
            "contrast": self.current_contrast,
            "auto_exposure": False,
            "auto_focus": False,
            "calibrating": self._calibrating,
        }

    def calibrate_auto_controls(self) -> dict:
        """Use a short AE/one-shot AF phase, then freeze the measured metadata."""
        if not self._is_running or not self.is_connected or self._control_queue is None:
            raise RuntimeError("Start the OAK camera before calibrating")
        if self._active_recording is not None:
            raise RuntimeError("Stop recording before calibrating")
        if not self._camera_calibration_lock.acquire(blocking=False):
            raise RuntimeError("Camera calibration is already running")
        
        self._calibrating = True
        locked = False
        prev_exposure = self.current_exposure_us
        prev_gain = self.current_gain
        prev_focus = self.current_focus
        
        try:
            settle = dai.CameraControl()
            settle.setAutoExposureEnable()
            settle.setAutoExposureLimit(10000)

            # Calibrate LIGHTING ONLY (Auto-Exposure and ISO Gain).
            # Industrial cameras looking at conveyors/tables should NEVER autofocus-hunt
            # because specular glare and smooth surfaces cause DepthAI autofocus to blur the lens.
            # We strictly keep the lens focus fixed and untouched.
            settle.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            if self.current_focus is not None:
                settle.setManualFocus(self.current_focus)
            self._control_queue.send(settle)
            
            # Wait 1.5s for frames to arrive and settle auto-exposure while background threads keep running
            time.sleep(1.5)
            
            latest = self._latest_metadata
            if latest is None:
                raise RuntimeError("No exposure and gain metadata arrived during calibration")
            exp_us, iso, lens = latest
            
            lock = dai.CameraControl()
            lock.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            if self.current_focus is not None:
                lock.setManualFocus(self.current_focus)
            lock.setManualExposure(exp_us, iso)
            self._control_queue.send(lock)
            locked = True
            
            self.current_exposure_us = exp_us
            self.current_gain = max(100, min(1600, iso))
            # Keep current focus intact - never overwrite with lens position
            self.control_mode = "auto"
            self.current_brightness, self.current_contrast = 0, 50
            self.auto_exposure_enabled = self.auto_focus_enabled = False
            self._save_calibrated_preset()
            return {"status": "ok", **self._camera_settings()}
        except Exception as e:
            logger.error(f"[CALIBRATION] Calibration failed: {e}")
            try:
                restore = dai.CameraControl()
                restore.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                if prev_focus is not None:
                    restore.setManualFocus(prev_focus)
                restore.setManualExposure(prev_exposure, prev_gain)
                self._control_queue.send(restore)
            except Exception as restore_error:
                logger.error(f"[CALIBRATION] Failed to restore fixed controls after calibration error: {restore_error}")
            raise RuntimeError(str(e)) from e
        finally:
            self._calibrating = False
            self._camera_calibration_lock.release()

    def _save_calibrated_preset(self) -> None:
        from app.core.database import SessionLocal
        from app.models.camera_model import Camera
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == self._active_camera_id).first() if self._active_camera_id else None
            camera = camera or db.query(Camera).first()
            if camera:
                camera.control_mode = "auto"
                camera.exposure = round(self.current_exposure_us / 1000, 3)
                camera.gain, camera.focus = self.current_gain, self.current_focus
                camera.brightness, camera.contrast = self.current_brightness, self.current_contrast
                camera.auto_exposure = camera.auto_focus = False
                db.commit()
                logger.info(f"[CALIBRATION] Saved calibrated preset to DB: exp={camera.exposure}ms, gain={camera.gain}, focus={camera.focus}")
        finally:
            db.close()

    def save_current_controls_to_db(self, control_mode: str = "manual") -> dict:
        """Persist current runtime controls (exposure, gain, focus, etc.) directly to vision_ai.db."""
        from app.core.database import SessionLocal
        from app.models.camera_model import Camera
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == self._active_camera_id).first() if self._active_camera_id else None
            camera = camera or db.query(Camera).first()
            if camera:
                camera.control_mode = control_mode
                camera.exposure = round(self.current_exposure_us / 1000, 3)
                camera.gain = float(self.current_gain)
                camera.focus = self.current_focus
                camera.brightness = self.current_brightness
                camera.contrast = self.current_contrast
                camera.auto_exposure = False
                camera.auto_focus = False
                db.commit()
                logger.info(
                    f"[CONTROL] Saved {control_mode} controls to DB: "
                    f"exp={camera.exposure}ms, gain={camera.gain}, focus={camera.focus}, "
                    f"brightness={camera.brightness}, contrast={camera.contrast}"
                )
                return {"status": "ok", "message": "Settings saved to database", **self._camera_settings()}
            return {"status": "error", "message": "No camera found in database"}
        except Exception as e:
            logger.error(f"[CONTROL] Failed to save controls to DB: {e}")
            raise
        finally:
            db.close()

    def get_hardware_calibration(self, width: int | None = None, height: int | None = None) -> dict:
        """Read 100% genuine hardware calibration and intrinsics from OAK-D camera EEPROM.
        
        Automatically detects the camera resolution from the active running stream or
        native hardware sensor without requiring any manual user input.
        """
        calib = None
        device_mxid = None

        # 1. If device is already connected and running in oak_camera_service
        if self.device is not None and self.is_connected:
            try:
                calib = self.device.readCalibration()
                device_mxid = self.device.getMxId()
            except Exception as e:
                logger.warning(f"[CALIBRATION] Failed reading calibration from active device: {e}")

        # 2. If not running, probe USB device directly under the thread lock
        if calib is None:
            with _depthai_device_enum_lock:
                devices = dai.Device.getAllAvailableDevices()
                if not devices:
                    return {
                        "status": "error",
                        "message": "No OAK-D camera detected via USB. Please connect the camera."
                    }
                device_info = devices[0]
                device_mxid = device_info.getMxId()
                try:
                    with dai.Device(device_info) as dev:
                        calib = dev.readCalibration()
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Failed to connect to OAK-D device ({device_mxid}): {e}"
                    }

        if not calib.hasCalibrationData():
            return {
                "status": "error",
                "message": f"Connected device ({device_mxid}) has no factory calibration data in EEPROM."
            }

        socket = dai.CameraBoardSocket.CAM_A

        # 3. Read REAL native sensor resolution and genuine intrinsics from EEPROM
        try:
            raw_matrix, native_calib_w, native_calib_h = calib.getDefaultIntrinsics(socket)
        except Exception:
            raw_matrix = calib.getCameraIntrinsics(socket)
            native_calib_w = 0
            native_calib_h = 0

        try:
            source_w = calib.getSourceWidth(socket)
            source_h = calib.getSourceHeight(socket)
        except Exception:
            source_w, source_h = native_calib_w, native_calib_h

        # 4. Auto-detect active resolution:
        # If camera is actively running, use its active pipeline resolution.
        # Otherwise, use native calibration resolution from hardware.
        if width is not None and height is not None:
            actual_w, actual_h = int(width), int(height)
        elif getattr(self, "_pipeline_width", None) and getattr(self, "_pipeline_height", None):
            actual_w, actual_h = int(self._pipeline_width), int(self._pipeline_height)
        else:
            actual_w = native_calib_w or source_w
            actual_h = native_calib_h or source_h

        try:
            k = np.array(calib.getCameraIntrinsics(socket, actual_w, actual_h))
        except Exception:
            k = np.array(raw_matrix)

        try:
            dist_coeffs = [float(c) for c in calib.getDistortionCoefficients(socket)]
        except Exception:
            dist_coeffs = []

        try:
            dist_model = str(calib.getDistortionModel(socket))
        except Exception:
            dist_model = None

        try:
            baseline_cm = calib.getBaselineDistance(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
            baseline_mm = float(baseline_cm * 10.0)
        except Exception:
            baseline_mm = None

        try:
            fov_deg = float(calib.getFov(socket))
        except Exception:
            fov_deg = None

        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])

        # Format exact YAML output with the genuine hardware dimensions
        yaml_lines = [
            "CAMERA_MATRIX:",
            f"  - [{k[0, 0]:.6f}, {k[0, 1]:.6f}, {k[0, 2]:.6f}]",
            f"  - [{k[1, 0]:.6f}, {k[1, 1]:.6f}, {k[1, 2]:.6f}]",
            f"  - [{k[2, 0]:.6f}, {k[2, 1]:.6f}, {k[2, 2]:.6f}]",
            f"CALIBRATION_WIDTH: {actual_w}",
            f"CALIBRATION_HEIGHT: {actual_h}",
        ]
        if dist_coeffs:
            formatted_coeffs = ", ".join(f"{c:.6f}" for c in dist_coeffs)
            yaml_lines.append(f"DISTORTION_COEFFICIENTS: [{formatted_coeffs}]")
        yaml_string = "\n".join(yaml_lines)

        return {
            "status": "ok",
            "device_mxid": device_mxid,
            "calibration_width": actual_w,
            "calibration_height": actual_h,
            "sensor_source_width": source_w,
            "sensor_source_height": source_h,
            "camera_matrix": k.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion_coefficients": dist_coeffs,
            "distortion_model": dist_model,
            "baseline_distance_mm": baseline_mm,
            "fov_degrees": fov_deg,
            "yaml_string": yaml_string,
        }

    # ==================== Inference ====================

    # ─────────────────────────────────────────────────────────────────────────────
    # Tube Analyzer Worker
    # This worker runs continuous TubeAnalyzer inference.
    # ─────────────────────────────────────────────────────────────────────────────

    def _inference_worker(self, session_id: str) -> None:
        from app.ml.camera_inference_service import analyze_camera_frame

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

        last_seen_seq = -1
        _live_frame_count = 0

        try:
            while not self._inference_stop.is_set():

                # ── Frame acquisition ──────────────────────────────────────────
                depth_frame = None
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
                            from app.ml.camera_inference_service import close_camera_session
                            close_camera_session(session_id)
                            logger.info("[INFERENCE] Video boundary — session reset, cycle counter carried forward")
                            _got_boundary = True
                            break
                        else:
                            if isinstance(item, tuple):
                                frame, depth_frame = item
                            else:
                                frame, depth_frame = item, None
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
                                    "frames_processed": self._offline_total_frames,
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
                    frame_data = self._latest_frame_data
                    if frame_data is None:
                        time.sleep(0.002)
                        continue

                    if len(frame_data) >= 5:
                        frame = frame_data[0]
                        depth_frame = frame_data[1]
                        frame_seq = frame_data[2]
                        frame_ts = frame_data[3]
                        hw_ts = frame_data[4]
                    else:
                        frame = frame_data[0]
                        depth_frame = None
                        frame_seq = frame_data[1]
                        frame_ts = frame_data[2]
                        hw_ts = frame_data[3] if len(frame_data) > 3 else frame_ts

                    if frame_seq == last_seen_seq:
                        time.sleep(0.002)
                        continue

                    last_seen_seq = frame_seq
                    _live_frame_count += 1

                    if depth_frame is None and self._depth_enabled:
                        depth_frame = self.get_synchronized_depth(hw_ts)

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
                infer_res = analyze_camera_frame(
                    frame if not self._inference_offline else frame.copy(),
                    session_id,
                    depth_frame=depth_frame,
                    video_name=self._current_video_name,
                    wait_for_model=self._inference_offline,
                    frame_seq=self._offline_frame_index if self._inference_offline else _live_frame_count,
                )
                latency_ms = (time.perf_counter() - t_start) * 1000

                if infer_res is None:
                    logger.warning("❌ [INFERENCE] analyze_camera_frame returned None")
                    continue

                if isinstance(infer_res, tuple):
                    res_dict, annotated_frame = infer_res
                    result = dict(res_dict)
                else:
                    result, annotated_frame = dict(infer_res), frame

                if not isinstance(result, dict):
                    logger.warning("❌ [INFERENCE] analyze_camera_frame result is not a dict")
                    continue

                # ── FPS calculation (consistent with video inference) ─────────
                infer_fps = float(result.get("fps") or 0.0)
                if infer_fps <= 0.0 and latency_ms > 0:
                    infer_fps = round(1000.0 / latency_ms, 1)

                fps_counter += 1
                now = time.time()
                elapsed = now - fps_window_start
                if elapsed >= 0.5:
                    current_fps      = fps_counter / elapsed
                    fps_counter      = 0
                    fps_window_start = now

                # Display the model's true GPU inference throughput (7-8 FPS), matching video inference
                display_fps = round(infer_fps, 1) if infer_fps > 0 else round(current_fps, 1)

                # ── GPU usage (sampled every result, cheap if nvidia-smi absent) ─
                gpu_pct = self._gpu_usage

                result["frame_seq"] = _live_frame_count if not self._inference_offline else self._offline_frame_index
                result["camera_packet_seq"] = frame_seq if not self._inference_offline else self._offline_frame_index
                _fp = result.get("frames_processed")
                result["frames_processed"] = _live_frame_count if not self._inference_offline else (_fp if _fp is not None else self._offline_frame_index)
                result["frame"] = result["frames_processed"]
                result["capture_age_ms"] = round((time.time() - frame_ts) * 1000, 1) if (not self._inference_offline and frame_ts) else round(latency_ms, 1)
                result["latency_ms"] = round(latency_ms, 1)
                result["frame_time_ms"] = round(latency_ms, 1)
                result["fps"] = display_fps
                result["metrics"] = {
                    "fps":            display_fps,
                    "gpu_pct":        round(gpu_pct, 1),
                    "latency_ms":     round(latency_ms, 1),
                    "frame_time_ms":  round(latency_ms, 1),
                    "capture_age_ms": result["capture_age_ms"],
                }

                logger.debug(
                    f"[INFERENCE] status={result.get('status')} "
                    f"record={result.get('record')} "
                    f"fps={display_fps:.1f} gpu={gpu_pct:.0f}% "
                    f"latency={latency_ms:.1f}ms"
                )

                # ── Frame progress (offline) ──────────────────────────────────
                if self._inference_offline:
                    result["frame_index"]   = self._offline_frame_index
                    result["total_frames"]  = self._offline_total_frames
                    result["video_index"]   = self._current_video_idx
                    result["total_videos"]  = self._total_videos
                    result["video_name"]    = self._current_video_name

                # Synchronized frame: attach the analyzed frame so frontend renders the exact frame
                # in 100% lockstep with its detection bounding boxes (no latency mismatch or trail)
                result["_raw_frame"] = frame

                # ── Push result to WebSocket ──────────────────────────────────
                if self._inference_result_queue is not None and self._inference_loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self._async_inference_push(result),
                        self._inference_loop,
                    )

        except Exception as e:
            logger.error(f"[INFERENCE] Thread error: {e}", exc_info=True)
        finally:
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

                # Optional aligned depth video pair (<video>_depth.mkv)
                depth_reader = None
                try:
                    from app.ml.camera_inference_service import open_offline_depth_reader
                    depth_reader = open_offline_depth_reader(video_path)
                    if depth_reader is not None:
                        logger.info(f"[OFFLINE] Found paired depth video for: {video_path}")
                except Exception as d_err:
                    logger.debug(f"[OFFLINE] Paired depth video check: {d_err}")

                frames_read = 0
                try:
                    while not self._offline_stop.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            logger.info(f"[OFFLINE] EOF — frames_read={frames_read}")
                            break

                        depth = depth_reader.read() if depth_reader else None

                        frames_read += 1
                        self._offline_frame_index = frames_read

                        # Block here if inference is behind — no frames are skipped
                        queue_item = (frame, depth) if depth is not None else frame
                        while not self._offline_stop.is_set():
                            try:
                                self._offline_frame_queue.put(queue_item, timeout=0.1)
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
                    if depth_reader:
                        depth_reader.close()

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

    def subscribe_inference(self) -> asyncio.Queue:
        # maxsize=1 ensures clients always receive the freshest frame with zero backlog/drift
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        if self._latest_inference_result is not None:
            try:
                q.put_nowait(self._latest_inference_result)
            except Exception:
                pass
        self._inference_subscribers.add(q)
        logger.info(f"[INFERENCE] Client subscribed — active subscribers: {len(self._inference_subscribers)}")
        return q

    def unsubscribe_inference(self, q: asyncio.Queue | None = None) -> None:
        if q is not None:
            self._inference_subscribers.discard(q)
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        else:
            for sub in list(self._inference_subscribers):
                while not sub.empty():
                    try:
                        sub.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._inference_subscribers.clear()
        logger.info(f"[INFERENCE] Client unsubscribed — remaining subscribers: {len(self._inference_subscribers)}")

    async def _async_inference_push(self, result: dict) -> None:
        """Drop-oldest push into all client inference queues — always keeps freshest frame for zero lag."""
        self._latest_inference_result = result
        for q in list(self._inference_subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                pass

        if self._inference_result_queue is not None and self._inference_result_queue not in self._inference_subscribers:
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
            logger.info(f"[INFERENCE] Stopping previous inference thread before starting session {session_id}...")
            self.stop_inference()
            if self._inference_thread and self._inference_thread.is_alive():
                self._inference_thread.join(timeout=1.0)

        logger.info(f"[INFERENCE] Mode: {'OFFLINE' if offline else 'ONLINE'} — session: {session_id}")

        # Set up queue and loop FIRST so the WebSocket can connect even if validation fails.
        # Validation errors are pushed as a result so the WS receives them before closing.
        self._inference_session_id = session_id
        self._inference_loop = loop
        self._inference_offline = offline
        self._inference_result_queue = asyncio.Queue(maxsize=1)

        def _push_error(msg: str) -> None:
            asyncio.run_coroutine_threadsafe(
                self._async_inference_push({"status": "error", "error": msg, "reasons": [msg], "done": True}),
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
                self._enforce_fixed_controls()
            self._is_streaming = True

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
        from app.ml.camera_inference_service import close_camera_session

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
                        if self._inference_session_id != session_id:
                            close_camera_session(session_id)
                    except Exception as e:
                        logger.error(f"[INFERENCE] close_camera_session during delayed release failed: {e}")

                threading.Thread(
                    target=_wait_and_release,
                    name="oak-inference-delayed-release",
                    daemon=True,
                ).start()

        # ── Clean up inference session (only if the thread is actually gone) ──
        if thread_confirmed_dead and self._inference_session_id:
            logger.info(f"[INFERENCE] Clearing session: {self._inference_session_id}")
            close_camera_session(self._inference_session_id)

        # Only forget the thread reference once it's confirmed dead -- otherwise
        # start_inference()'s already_running guard must keep seeing it so a
        # second worker can't start on top of the still-running one.
        if thread_confirmed_dead:
            self._inference_thread = None
        self._inference_watchdog_thread = None
        self.unsubscribe_inference()
        self._inference_result_queue = None
        self._inference_session_id = None
        self._inference_offline = False
        self._latest_inference_result = None

        # Keep capture threads running while camera pipeline is active
        # to ensure DepthAI hardware XLink queues are continuously drained without overflow.
        logger.info("[INFERENCE] Stopped (capture threads continue draining hardware queues)")

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
            res = await asyncio.wait_for(self._inference_result_queue.get(), timeout=timeout)
            while not self._inference_result_queue.empty():
                try:
                    res = self._inference_result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            return res
        except asyncio.TimeoutError:
            return None

    # ==================== Recording ====================

    def _resolve_recording_path(self) -> str | None:
        try:
            from app.services.recording_service import _resolve_recording_path as _res
            return _res()
        except Exception:
            return None

    def _resolve_recording_format(self, override: str | None = None) -> str:
        if override and isinstance(override, str) and override.strip():
            return override.strip().upper()
        try:
            from app.core.database import SessionLocal
            from app.services.camera_service import get_camera_config
            db = SessionLocal()
            try:
                cam = get_camera_config(db)
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
        camera_settings: dict | None = None,
    ) -> str:
        if not self.is_connected or not self._is_running or self.device is None:
            raise RuntimeError("Cannot start recording: OAK camera is not connected.")

        from app.services.recording_service import start_recording as _start, active_recordings

        root_path = self._resolve_recording_path()
        rec_fmt = self._resolve_recording_format(recording_format)

        # Ensure recording resolution matches the actual camera pipeline to prevent PyAV rescaling
        if hasattr(self, "_pipeline_width") and self._pipeline_width and hasattr(self, "_pipeline_height") and self._pipeline_height:
            if (width, height) != (self._pipeline_width, self._pipeline_height):
                logger.warning(
                    f"[RECORD] Requested resolution {width}x{height} differs from camera pipeline "
                    f"{self._pipeline_width}x{self._pipeline_height} — overriding to match camera."
                )
                width = self._pipeline_width
                height = self._pipeline_height

        if self._latest_bgr is not None:
            actual_h, actual_w = self._latest_bgr.shape[:2]
            if (width, height) != (actual_w, actual_h):
                logger.warning(f"[RECORD] Resolution adjusted to actual frame size: {actual_w}x{actual_h}")
                width, height = actual_w, actual_h

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
            self._enforce_fixed_controls()
            logger.info("[RECORD] Step 2 — fixed camera controls enforced")
        else:
            logger.info("[RECORD] Capture threads already running (AE limit already applied at stream start)")
        self._is_streaming = True

        path = _start(session_id, width, height, fps, root_path, rec_fmt, camera_settings)
        self._active_recording = active_recordings.get(session_id)

        if self._active_recording is None:
            logger.error("[RECORD] Failed to get recording session from active_recordings!")
        else:
            logger.info(f"[RECORD] Session active — frames will be fed from _convert_loop → path: {path}")

        return path

    def _recording_settings_snapshot(self) -> dict:
        return {
            "resolution": f"{self._pipeline_width}x{self._pipeline_height}",
            "fps": self._configured_fps,
            "exposure_us": self.current_exposure_us,
            "gain": self.current_gain,
            "focus": self.current_focus,
            "brightness": self.current_brightness,
            "contrast": self.current_contrast,
            "auto_exposure": self.auto_exposure_enabled,
            "auto_focus": self.auto_focus_enabled,
            "control_mode": self.control_mode,
        }

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

        # Stop capture threads only if streaming and inference are also not active
        streaming_active = bool(self._is_streaming or self._stream_subscribers or (self._stream_queue is not None))
        inference_active = bool(self._inference_thread and self._inference_thread.is_alive())
        if not streaming_active and not inference_active:
            logger.info("[RECORD] No active streaming or inference — stopping capture threads")
            self._stop_capture_threads()
        else:
            logger.info(f"[RECORD] Streaming or inference active (streaming={streaming_active}, inference={inference_active}) — keeping capture threads running")

        return result

    # ==================== App Lifecycle (connect / disconnect) ====================

    async def start(self, config) -> dict:
        """App startup — connect device and build pipeline only. Does NOT start capture threads."""
        async with self._lifecycle_lock:
            if self._is_running:
                logger.info("[START] Pipeline already running")
                return {"status": "already_running"}

            try:
                self._active_camera_id = getattr(config, "id", None)
                if not self.is_connected:
                    if not await self.connect(config.ip_address):
                        # If device is already in use by another process, do NOT attempt USB fallback
                        # as it will only repeat the exact same failure and spam logs.
                        if "already used" in (self._last_connection_error or "").lower() or "in use" in (self._last_connection_error or "").lower():
                            return {
                                "status": "error",
                                "message": self._last_connection_error or "Device is already used by another application/process. Make sure to close all applications/processes using the device before starting a new one.",
                                "error_code": "DEVICE_IN_USE",
                            }

                        # If no OAK device was found at all, USB fallback won't help
                        if "no oak device found" in (self._last_connection_error or "").lower():
                            return {
                                "status": "error",
                                "message": self._last_connection_error or "No OAK device found over USB or network",
                                "error_code": "NO_DEVICE",
                            }

                        # Older databases may select a locked/unavailable
                        # network camera as the newest row. Prefer a local USB
                        # OAK when one is available so startup still works.
                        selected = str(config.ip_address or "").strip().lower()
                        if selected not in {"", "usb", "auto", "127.0.0.1", "localhost"}:
                            logger.warning(
                                f"[START] Could not connect to {config.ip_address}; trying USB OAK"
                            )
                            if not await self.connect("usb"):
                                return {"status": "error", "message": self._last_connection_error or "Device connection failed"}
                        else:
                            return {"status": "error", "message": self._last_connection_error or "Device connection failed"}

                self.current_brightness = int(config.brightness or 0)
                self.current_contrast = int(config.contrast if config.contrast is not None and config.contrast != 0 else 50)

                depth_attempted = self._depth_enabled
                build_ok = False
                try:
                    with self._pipeline_lock:
                        self._pipeline = dai.Pipeline(self.device)
                    build_ok = self._build_pipeline(self._pipeline, config, enable_depth=depth_attempted)
                except Exception as build_err:
                    logger.warning(f"[START] Combined pipeline build failed ({build_err})")
                    build_ok = False

                if not build_ok and depth_attempted:
                    logger.warning("[START] Rebuilding clean RGB-only pipeline fallback...")
                    self._depth_dai_queue = None
                    with self._pipeline_lock:
                        self._pipeline = dai.Pipeline(self.device)
                    build_ok = self._build_pipeline(self._pipeline, config, enable_depth=False)

                if not build_ok:
                    return {"status": "error", "message": "Pipeline build failed"}

                try:
                    self._pipeline.start()
                except Exception as start_err:
                    if depth_attempted and self._depth_dai_queue is not None:
                        logger.warning(f"[START] Depth pipeline start failed ({start_err}), retrying in RGB-only mode...")
                        self._cleanup_pipeline()
                        with self._pipeline_lock:
                            self._pipeline = dai.Pipeline(self.device)
                        self._depth_dai_queue = None
                        if self._build_pipeline(self._pipeline, config, enable_depth=False):
                            self._pipeline.start()
                        else:
                            raise start_err
                    else:
                        raise start_err

                self._is_running = True

                # Immediately launch capture threads so DepthAI hardware queues are continuously drained
                self._start_capture_threads()

                # Read factory camera intrinsics for active resolution and inject into ML analyzer
                try:
                    calib = self.device.readCalibration()
                    w_cal = int(self._pipeline_width or 1920)
                    h_cal = int(self._pipeline_height or 1080)
                    eeprom_k = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w_cal, h_cal)
                    k_mat = [
                        [float(eeprom_k[0][0]), float(eeprom_k[0][1]), float(eeprom_k[0][2])],
                        [float(eeprom_k[1][0]), float(eeprom_k[1][1]), float(eeprom_k[1][2])],
                        [float(eeprom_k[2][0]), float(eeprom_k[2][1]), float(eeprom_k[2][2])],
                    ]
                    try:
                        dist_coeffs = [float(c) for c in calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)]
                    except Exception:
                        dist_coeffs = None
                    from app.ml.camera_inference_service import update_camera_calibration
                    update_camera_calibration(k_mat, w_cal, h_cal, dist_coeffs)
                    logger.info(f"[CALIB] Factory CAM_A intrinsics injected for {w_cal}x{h_cal}: fx={eeprom_k[0][0]:.1f}, fy={eeprom_k[1][1]:.1f}")
                except Exception as cal_err:
                    logger.warning(f"[CALIB] Could not read factory calibration: {cal_err}")

                if self._legacy_preset_migration:
                    self._save_calibrated_preset()
                    self._legacy_preset_migration = False

                # Re-apply the fixed controls after pipeline startup so they take
                # effect before capture or recording begins.
                self._enforce_fixed_controls()

                logger.info("[START] Pipeline ready — waiting for stream/start")
                realtime_log_service.add_log(
                    "system",
                    "SYSTEM",
                    "Application initialized successfully",
                    "success"
                )

                # Pre-warm CUDA in background as soon as camera is ready
                try:
                    from app.ml.camera_inference_service import prewarm_camera_model
                    threading.Thread(target=prewarm_camera_model, daemon=True, name="oak-prewarm-model").start()
                except Exception as _e:
                    logger.warning(f"[PREWARM] Failed to start prewarm thread: {_e}")

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
                await asyncio.to_thread(self.stop_inference)
                
                # If a recording is active, we should attempt to stop it gracefully
                if self._active_recording and getattr(self._active_recording, "session_id", None):
                    await asyncio.to_thread(self.stop_recording, self._active_recording.session_id)
                elif self._inference_session_id:
                    await asyncio.to_thread(self.stop_recording, self._inference_session_id)
                    
                self._close_stream_queue()
                self._latest_jpeg = None
                await asyncio.to_thread(self._stop_capture_threads)
                await asyncio.to_thread(self._cleanup_pipeline)
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

    # ==================== Fixed-control Runtime Enforcement ====================

    def _enforce_fixed_controls(self) -> None:
        """Re-send fixed focus, exposure, brightness and contrast through the runtime control queue."""
        if not self._is_running or not self.is_connected or self._control_queue is None:
            return
        try:
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            if self.current_focus is not None:
                ctrl.setManualFocus(self.current_focus)
            ctrl.setManualExposure(self.current_exposure_us, self.current_gain)
            hw_b = max(-10, min(10, int(round(self.current_brightness / 5.0))))
            hw_c = max(-10, min(10, int(round((self.current_contrast - 50) / 5.0))))
            ctrl.setBrightness(hw_b)
            ctrl.setContrast(hw_c)
            self._control_queue.send(ctrl)
            logger.info(f"[CONTROL] Fixed controls sent after pipeline startup — brightness={hw_b} contrast={hw_c}")
        except Exception as e:
            if "closed" in str(e).lower() or "xlink" in str(e).lower():
                self._on_device_lost("Control queue closed during fixed-control enforcement")
            else:
                logger.warning(f"[CONTROL] Fixed-control enforcement failed: {e}")

    # ==================== Stream Lifecycle (capture threads) ====================

    async def start_streaming(self) -> dict:
        """User clicks Start Streaming — subscribe streaming clients and ensure capture threads are active."""
        async with self._streaming_lock:
            if not self._is_running or not self.is_connected or self.device is None:
                # Self-healing: if pipeline is not active, attempt auto-starting with active database config
                logger.info("[STREAMING] Device not connected when start_streaming requested — attempting auto-connect")
                try:
                    from app.core.database import SessionLocal
                    from app.services import camera_service as cam_db
                    with SessionLocal() as db:
                        cfg = cam_db.get_camera_config(db)
                    if cfg:
                        start_res = await self.start(cfg)
                        if start_res.get("status") == "error":
                            return start_res
                except Exception as auto_err:
                    logger.warning(f"[STREAMING] Auto-connect attempt failed: {auto_err}")

            if not self._is_running or not self.is_connected or self.device is None:
                return {"status": "error", "message": "Cannot stream: OAK camera is not connected — call /oak/start first"}

            if not self._start_capture_threads():
                return {"status": "error", "message": "Capture threads failed to start"}

            # Re-apply fixed controls now that the runtime queue is available.
            logger.info("[STREAMING] Step 1 — capture threads started / confirmed")
            self._enforce_fixed_controls()
            logger.info("[STREAMING] Step 2 — fixed camera controls enforced")

            self._is_streaming = True
            logger.info("[STREAMING] Started")
            realtime_log_service.add_log(
                "stream",
                "STREAM",
                "Video stream initialized successfully",
                "success"
            )

            # Pre-warm CUDA in background when stream starts so Start Inference has 0.0s delay
            try:
                from app.ml.camera_inference_service import prewarm_camera_model
                threading.Thread(target=prewarm_camera_model, daemon=True, name="oak-prewarm-model").start()
            except Exception as _e:
                logger.warning(f"[PREWARM] Failed to start prewarm thread: {_e}")

            return {"status": "streaming"}

    async def stop_streaming(self) -> dict:
        """User clicks Stop Streaming — close client stream queues without interrupting hardware capture."""
        async with self._streaming_lock:
            self._is_streaming = False
            self._latest_jpeg = None
            for q in list(self._stream_subscribers):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            self._stream_subscribers.clear()
            self._stream_queue = None

            # CRITICAL STABILITY FIX:
            # We intentionally DO NOT stop capture threads here.
            # The DepthAI camera continuously pushes frames into hardware XLink queues.
            # Leaving capture threads running ensures DepthAI queues never overflow,
            # prevents camera disconnection/crash on USB 2.0/3.0, and makes re-starting
            # the stream instantaneous without thread destruction races.
            logger.info("[STREAMING] Stopped (subscribers disconnected; capture threads keep draining camera queues)")
            return {"status": "stopped"}

    def _refresh_device_detection_async(self):
        """Asynchronously probe DepthAI devices without blocking health polling."""
        if getattr(self, "_is_scanning_devices", False):
            return
        self._is_scanning_devices = True

        def _scan():
            try:
                with _depthai_device_enum_lock:
                    devs = dai.Device.getAllAvailableDevices()
                count = len(devs)
                self._cached_devices_count = count
                self._cached_device_detected = count > 0
            except Exception as e:
                logger.debug(f"[OAK PROBE] Failed checking devices: {e}")
                self._cached_devices_count = 0
                self._cached_device_detected = False
            finally:
                self._last_device_scan_time = time.time()
                self._is_scanning_devices = False

        t = threading.Thread(target=_scan, daemon=True, name="oak-device-probe")
        t.start()

    def is_device_detected(self) -> bool:
        """Check if any physical OAK / DepthAI device is present."""
        if self.is_connected and self.device is not None:
            return True
        now = time.time()
        if now - getattr(self, "_last_device_scan_time", 0.0) > 2.5 and not getattr(self, "_is_scanning_devices", False):
            self._refresh_device_detection_async()
        return getattr(self, "_cached_device_detected", False)

    # ==================== Health ====================

    def health(self) -> dict:
        is_hardware_connected = bool(self.is_connected and self.device is not None)
        device_detected = self.is_device_detected()
        return {
            "running": bool(self._is_running and is_hardware_connected),
            "connected": is_hardware_connected,
            "device_detected": device_detected,
            "devices_count": 1 if is_hardware_connected else getattr(self, "_cached_devices_count", 0),
            "camera_running": bool(self._is_running and is_hardware_connected),
            "is_running": bool(self._is_running and is_hardware_connected),
            "is_connected": is_hardware_connected,
            "mjpeg_thread": bool(self._mjpeg_thread and self._mjpeg_thread.is_alive() and is_hardware_connected),
            "hires_thread": bool(self._hires_thread and self._hires_thread.is_alive() and is_hardware_connected),
            "convert_thread": bool(self._convert_thread and self._convert_thread.is_alive() and is_hardware_connected),
            "inference_thread": bool(self._inference_thread and self._inference_thread.is_alive() and is_hardware_connected),
            "inference_watchdog_thread": bool(self._inference_watchdog_thread and self._inference_watchdog_thread.is_alive() and is_hardware_connected),
            "depth_available": bool(self._depth_dai_queue is not None and is_hardware_connected),
            "depth_thread": bool(self._depth_thread and self._depth_thread.is_alive() and is_hardware_connected),
            "streaming": bool(self._is_streaming and is_hardware_connected),
            "fps": round(self.current_fps, 1) if is_hardware_connected else 0.0,
            "camera_settings": self._camera_settings() if (self._is_running and is_hardware_connected) else None,
            "last_error": getattr(self, "_last_connection_error", None),
        }


oak_camera_service = OakCameraService()
