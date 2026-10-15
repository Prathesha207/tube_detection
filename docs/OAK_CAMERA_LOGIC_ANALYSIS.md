# 🎥 OAK Camera Connection Logic - Complete Analysis

## 📋 Executive Summary

Your DHTX system has **two separate but related operational modes**:
1. **ONLINE MODE**: Camera connected via IP → Live streaming, recording, and inference
2. **OFFLINE MODE**: Video file uploaded → Inference only (no camera needed)

The backend manages all camera I/O through threading (MJPEG capture, raw frame conversion, inference) while the frontend controls the process lifecycle via API calls and WebSocket for real-time updates.

---

## 🏗️ System Architecture

### Backend Components

| Component | Path | Responsibility |
|-----------|------|-----------------|
| **oak_camera_service.py** | `app/services/` | Core service managing device connection, pipeline, capture threads, streaming, recording, inference |
| **oak_camera_router.py** | `app/routers/` | HTTP/WebSocket endpoints (start, stop, stream, inference) |
| **recording_service.py** | `app/services/` | PyAV-based video encoding with per-frame timestamps |
| **tube_inference_service.py** | `app/ml/` | YOLO-based AI inference (object detection) |
| **camera_model.py** | `app/models/` | Database persistence for camera config (IP, resolution, FPS, controls) |

### Frontend Components

| Component | Path | Responsibility |
|-----------|------|-----------------|
| **visionProcessStore.ts** | `src/store/` | Zustand store orchestrating process lifecycle (start/stop streaming, recording, inference) |
| **DashboardDock.tsx** | `src/components/` | UI buttons for Stream/Record/Inference with enable/disable logic |
| **useInferenceSocket.ts** | `src/hooks/` | WebSocket hook receiving inference results and updating UI |
| **cameraService.ts** | `src/services/` | API client for camera operations |

---

## 🌐 Camera Connection States

### State 1: Device Not Connected
```
Status: ❌ OFFLINE
├─ Camera IP not configured OR device not reachable
├─ Pipeline: NOT built
├─ Streaming: ❌ Disabled
├─ Recording: ❌ Disabled (requires camera)
└─ Inference: ✅ Enabled (video files only)
```

### State 2: Device Connected
```
Status: ✅ ONLINE
├─ Camera connected via IP address
├─ Pipeline: ✅ Built (MJPEG encoder + raw NV12 outputs)
├─ Streaming: ✅ Available (start button enabled)
├─ Recording: ✅ Available (after streaming starts)
└─ Inference: ✅ Available (camera or video files)
```

### State 3: Streaming Active
```
Status: 🟢 STREAMING
├─ 3 capture threads running:
│  ├─ MJPEG thread: drains encoder queue → pushes JPEG to HTTP stream
│  ├─ Hi-res thread: drains raw NV12 → queues for converter
│  └─ Converter thread: converts NV12 → BGR → stores in _latest_bgr
├─ HTTP MJPEG stream: GET /oak/oak/stream (multipart/x-mixed-replace)
├─ Recording: ✅ Can start (frames feed to recorder)
└─ Inference: ✅ Can start (frames feed to inference worker)
```

---

## 🔄 Complete System Flows

### FLOW 1: Application Startup → Device Connection

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: App Initialization                                   │
│ (User opens DHTX, backend starts)                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Frontend calls POST /oak/oak/start                   │
│ (Automatic or user-triggered from Dashboard)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: oak_camera_service.start()                         │
│                                                              │
│ 1. Read camera config from database:                        │
│    - IP address (e.g., "192.168.1.100")                     │
│    - Resolution (e.g., "1920x1080")                         │
│    - FPS (e.g., 30)                                         │
│    - Control mode ("auto" or "manual")                      │
│                                                              │
│ 2. Try to connect to OAK camera:                            │
│    - dai.Device(DeviceInfo(ip_address))                     │
│    - If successful: is_connected = True                     │
│    - If failed: is_connected = False → Return error         │
│                                                              │
│ 3. Build DAI pipeline (if connected):                       │
│    ├─ Camera sensor → MJPEG encoder                         │
│    │  └─ Creates queue: _mjpeg_dai_queue (for HTTP stream)  │
│    │                                                         │
│    └─ Camera sensor → Raw NV12 output                       │
│       └─ Creates queue: _raw_dai_queue (for hi-res frames)  │
│                                                              │
│ 4. Set camera exposure/focus settings:                      │
│    - Auto mode: setAutoExposureEnable()                     │
│    - Manual mode: setManualExposure(exp, gain)              │
│                                                              │
│ 5. Set _is_running = True                                  │
│                                                              │
│ 6. Log: "Camera connected (IP)" ✓                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: Receives { status: "ok" }                         │
│                                                              │
│ - CameraPanel shows "Camera Connected" badge                │
│ - Stream button becomes ENABLED ✅                          │
│ - Record button becomes AVAILABLE                           │
│ - Inference button becomes ENABLED ✅                       │
└─────────────────────────────────────────────────────────────┘
```

**Error Case:**
```
If camera connection fails:
  - Logs: "Connection failed: [error]"
  - Sends: { status: "error", message: "Camera not configured" }
  - UI shows: "Offline" mode
  - Stream/Record: DISABLED ❌
  - Inference: Enabled for video files only ✅
```

---

### FLOW 2: User Clicks STREAM Button

```
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: User clicks "START STREAMING" button             │
│ (visible only if camera connected)                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: visionProcessStore.startProcess({                 │
│   type: "streaming",                                        │
│   source: "camera"                                          │
│ })                                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: POST /oak/oak/stream/start (HTTP)                │
│                                                              │
│ - Sets UI state: activeProcess = "streaming"                │
│ - Button shows STOP icon (Square)                           │
│ - Spinner animation starts                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: oak_camera_service.start_streaming()              │
│                                                              │
│ 1. Check if pipeline is running (_is_running == True)      │
│    - If not: return error                                   │
│                                                              │
│ 2. Start 3 capture threads:                                │
│                                                              │
│    ┌─ MJPEG THREAD (oak-mjpeg) ──────────────────────┐     │
│    │ Loop:                                            │     │
│    │  - pkt = _mjpeg_dai_queue.tryGet()              │     │
│    │  - if pkt: jpeg = pkt.getData()                 │     │
│    │  - push to asyncio stream_queue                 │     │
│    │  - log FPS every 5 seconds                       │     │
│    │                                                  │     │
│    │ Purpose: feeds JPEG frames to HTTP stream       │     │
│    └─────────────────────────────────────────────────┘     │
│                                                              │
│    ┌─ HI-RES THREAD (oak-hires) ──────────────────────┐    │
│    │ Loop:                                             │    │
│    │  - pkt = _raw_dai_queue.tryGet()                │    │
│    │  - if pkt: _hires_packet_queue.put(pkt)         │    │
│    │  - drop-oldest if queue full                     │    │
│    │                                                  │    │
│    │ Purpose: queues raw NV12 for converter           │    │
│    └──────────────────────────────────────────────────┘    │
│                                                              │
│    ┌─ CONVERTER THREAD (oak-convert) ────────────────┐     │
│    │ Loop:                                            │     │
│    │  - pkt = _hires_packet_queue.get(timeout=0.1)   │     │
│    │  - bgr = pkt.getCvFrame()  ← NV12 → OpenCV BGR  │     │
│    │  - self._latest_bgr = bgr                        │     │
│    │  - if recording: rec.add_frame(bgr)              │     │
│    │  - log FPS every 2 seconds                        │     │
│    │                                                  │     │
│    │ Purpose: converts frames, feeds to recording     │     │
│    └─────────────────────────────────────────────────┘     │
│                                                              │
│ 3. State change:                                            │
│    - _stream_queue = asyncio.Queue(maxsize=4)               │
│    - _server_loop = asyncio.get_running_loop()              │
│    - All thread stop events: .clear()                       │
│    - All threads: .start()                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: Connects to HTTP MJPEG stream                    │
│ GET /oak/oak/stream                                         │
│                                                              │
│ Response: multipart/x-mixed-replace content                │
│  - JPEG frame 1: --frame\r\nContent-Type: image/jpeg\r\n   │
│  - JPEG frame 2: --frame\r\nContent-Type: image/jpeg\r\n   │
│  - ... (continuous until client disconnects)               │
│                                                              │
│ Video canvas displays live preview in real-time             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ User sees LIVE PREVIEW on screen 🎥                        │
│                                                              │
│ Button states update:                                       │
│ - Stream: ACTIVE (Stop button visible)                      │
│ - Record: NOW ENABLED ✅ (can start recording)              │
│ - Inference: NOW LOCKED (can click after setting source)   │
└─────────────────────────────────────────────────────────────┘
```

**Frame Flow During Streaming:**
```
OAK Camera (30 FPS)
    ↓
DAI Pipeline (splits into 2 queues)
    ├─ _mjpeg_dai_queue (MJPEG encoded frames)
    │   └─ MJPEG Thread → asyncio stream_queue
    │       └─ HTTP GET /oak/oak/stream → Browser <canvas>
    │
    └─ _raw_dai_queue (Raw NV12 frames)
        └─ HI-RES Thread → _hires_packet_queue
            └─ CONVERTER Thread → _latest_bgr (always latest frame)
                ├─ Recording: feeds to recorder
                └─ Inference: available for read
```

---

### FLOW 3: User Clicks RECORD Button

```
⚠️  PREREQUISITE: Streaming must be active
    (Record button disabled if streaming not running)

┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: User clicks "START RECORDING" button             │
│ (visible only after streaming starts)                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: visionProcessStore.startProcess({                 │
│   type: "recording",                                        │
│   source: "camera"                                          │
│ })                                                           │
│                                                              │
│ Creates: recordingSessionId = "session_" + timestamp        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: POST /recording/start                            │
│ Body: { session_id: "session_1234567890" }                 │
│                                                              │
│ - Sets UI state: activeProcess = "recording"                │
│ - Button shows STOP icon (Square)                           │
│ - Spinner animation starts                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: recording_api.start_recording()                   │
│                                                              │
│ 1. Get camera config (resolution, FPS)                     │
│    └─ Parse resolution: "1920x1080" → width=1920, h=1080   │
│                                                              │
│ 2. Call oak_camera_service.start_recording(                │
│      session_id, width, height, fps                        │
│    )                                                         │
│                                                              │
│ 3. Create RecordingSession:                                │
│    ├─ Path: ~/Desktop/recordings/YYYY-MM-DD/              │
│    ├─ Filename: session_YYYY-MM-DD_HH-MM-SS.avi            │
│    ├─ Codec: MJPEG (.avi) or FFV1 (.mkv)                  │
│    ├─ Container: av.open(video_path, mode="w")             │
│    └─ Stream: add_stream(codec, rate=1000)                 │
│                                                              │
│ 4. Start worker thread:                                    │
│    - Waits for frames in frame_queue                        │
│    - Encodes and writes frames to container                │
│    - Per-frame wall-clock timestamps (no FPS stretching)   │
│                                                              │
│ 5. Store in self._active_recording                         │
│                                                              │
│ Return: { status: "recording", recording_path: "/path" }   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: CONVERTER THREAD → adds frames to recording       │
│                                                              │
│ Each frame captured from camera:                            │
│                                                              │
│  _convert_thread loop:                                      │
│    bgr = pkt.getCvFrame()                                  │
│    self._latest_bgr = bgr                                  │
│    if self._active_recording is not None:                  │
│       self._active_recording.add_frame(bgr)                │
│       ↓                                                     │
│       RecordingSession.frame_queue.put_nowait(bgr)          │
│       ↓                                                     │
│       RecordingSession._worker thread:                      │
│          rgb = cv2.cvtColor(bgr, BGR2RGB)                  │
│          frame = av.VideoFrame.from_ndarray(rgb)            │
│          container.encode(frame)                            │
│          ← writes to .avi/.mkv file                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ User sees: RECORDING indicator on UI                       │
│ - "Recording..." status text                                │
│ - Red record circle visible                                 │
│ - Video frames being written to disk in real-time           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: User clicks "STOP RECORDING" button              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: POST /recording/stop                             │
│ Body: { session_id: "session_1234567890" }                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: oak_camera_service.stop_recording()               │
│                                                              │
│ 1. Set self._active_recording = None                        │
│    └─ Converter thread stops feeding frames                 │
│                                                              │
│ 2. RecordingSession.is_running = False                      │
│    └─ Worker thread drains queue and exits                  │
│                                                              │
│ 3. Container closes:                                        │
│    container.close()  ← finalizes file                      │
│                                                              │
│ 4. Video saved with per-frame timestamps:                   │
│    ~/Desktop/recordings/YYYY-MM-DD/session_HH-MM-SS.avi    │
│                                                              │
│ Return: { status: "done" }                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: Updates UI                                       │
│ - Recording button back to "Start" state                    │
│ - "Recording..." status cleared                             │
│ - User can start new recording or stop streaming            │
└─────────────────────────────────────────────────────────────┘
```

**Key Recording Features:**
- **PyAV Backend**: Each frame gets actual wall-clock timestamp
  - If camera delivers 3.2 FPS instead of 10 FPS → video is still correct duration
  - Not affected by camera's auto-exposure delays
- **Codec Choices**: MJPEG (lossy, smaller) or FFV1 (lossless, larger)
- **Folder Structure**: Organized by date `YYYY-MM-DD/`
- **Max Quality**: Uses JPEG quality 95 for inference-grade recordings

---

### FLOW 4: User Clicks INFERENCE Button (ONLINE - Camera)

```
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: User clicks "START INFERENCE" button             │
│ (visible if camera connected OR video uploaded)             │
│                                                              │
│ Source selection: "camera" (online mode)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: visionProcessStore.startProcess({                 │
│   type: "inference",                                        │
│   source: "camera"  ← KEY DIFFERENCE FROM OFFLINE          │
│ })                                                           │
│                                                              │
│ Creates: recordingSessionId = "inf_" + timestamp            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: AUTO-START STREAMING (if not already running)    │
│                                                              │
│ if (source === "camera")                                    │
│   POST /oak/oak/stream/start  ← automatic                   │
│                                                              │
│ (Video mode skips this - no HTTP stream for offline)        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: POST /oak/oak/inference/start/{sessionId}        │
│ Body: {                                                      │
│   offline: false,  ← online mode                            │
│   videoPath: null,                                          │
│   folderPath: null                                          │
│ }                                                            │
│                                                              │
│ - Sets UI state: activeProcess = "inference"                │
│ - Button shows STOP icon (Square)                           │
│ - Spinner animation starts                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: oak_camera_service.start_inference(                │
│   session_id,                                               │
│   loop=asyncio.get_running_loop(),                          │
│   offline=False  ← KEY FLAG                                 │
│ )                                                            │
│                                                              │
│ 1. Set state:                                               │
│    - _inference_offline = False                             │
│    - _inference_session_id = session_id                     │
│    - _inference_stop.clear()                                │
│                                                              │
│ 2. Create asyncio queue for results:                        │
│    - _inference_result_queue = asyncio.Queue()              │
│    - _inference_loop = loop (for thread-safe pushes)        │
│                                                              │
│ 3. Start GPU sampler thread:                               │
│    - Samples GPU usage every 2 seconds                      │
│    - Non-blocking (never interferes with inference)         │
│                                                              │
│ 4. Start inference worker thread:                          │
│    - _inference_thread = Thread(_inference_worker)          │
│    - _inference_thread.start()                              │
│                                                              │
│ 5. Start watchdog thread (online mode only):               │
│    - _inference_watchdog_thread = Thread(_inference_wd)     │
│    - If worker dies: auto-restart (max 5 times)             │
│                                                              │
│ Return: { status: "inference_started" }                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: _inference_worker loop (online mode)              │
│                                                              │
│ while not _inference_stop.is_set():                         │
│                                                              │
│   1. Get latest camera frame:                              │
│      frame = self._latest_bgr  ← captured by converter     │
│      if frame is None: sleep(0.01) and continue             │
│                                                              │
│   2. Normalize frame (MJPEG codec compatibility):           │
│      if config.inference_format == "MJPEG":                 │
│         frame = encode-decode roundtrip JPEG               │
│         ← matches what inference video will see             │
│                                                              │
│   3. Run AI inference (YOLO):                              │
│      result = run_inference(frame.copy(), session_id)       │
│      └─ Outputs: {                                          │
│           status: "WAITING_MODEL1" | "MODEL1_PASS" | ...   │
│           record: True/False                                │
│           model1_detected: True/False                       │
│           detections: [{bbox, confidence, contour, ...}]   │
│           detections_model2: [...]                          │
│           mask_polygons: [...]                              │
│           lenA1, lenA2: measurements                        │
│           metrics: { ... }                                  │
│         }                                                    │
│                                                              │
│   4. Calculate metrics:                                     │
│      - FPS: frames per second (rolling 0.5s window)         │
│      - GPU: nvidia-smi utilization %                        │
│      - Latency: inference time in milliseconds              │
│                                                              │
│   5. Attach metrics to result:                             │
│      result["metrics"] = {                                  │
│        fps: 28.5,                                           │
│        gpu_pct: 65.2,                                       │
│        latency_ms: 45.3                                     │
│      }                                                       │
│                                                              │
│   6. Push to asyncio queue:                                │
│      asyncio.run_coroutine_threadsafe(                      │
│        _async_inference_push(result),                       │
│        _inference_loop                                      │
│      )                                                       │
│      └─ WebSocket endpoint reads this queue                 │
│                                                              │
│   7. Auto-feed to inference recorder (if enabled):         │
│      if result["record"] and config.recorded_video:         │
│         recorder.write(frame, result)                       │
│                                                              │
│   Loop repeats: grab next frame, run inference ...          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: Connect to WebSocket                             │
│ new WebSocket(ws://localhost:8000/oak/oak/inference/ws/{id})│
│                                                              │
│ Loop: receive JSON results every inference iteration        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: WebSocket inference endpoint                      │
│ @router.websocket("/inference/ws/{session_id}")             │
│                                                              │
│ while True:                                                 │
│   result = await oak_camera_service.get_inference_result()  │
│   ← waits for result from worker thread                     │
│                                                              │
│   if result is None:                                        │
│      if _inference_result_queue is None:                    │
│         inference stopped → close WebSocket                 │
│         break                                               │
│      continue  ← wait for next result                       │
│                                                              │
│   if result.get("done"):                                    │
│      send final result, close WebSocket                     │
│      break                                                  │
│                                                              │
│   Encode raw frame if present:                             │
│   if "_raw_frame" in result:                                │
│      bgr_to_jpeg_base64()                                   │
│      result["frame"] = "data:image/jpeg;base64,..."         │
│      ← for offline mode (online doesn't need it)            │
│                                                              │
│   Send JSON to frontend:                                    │
│   await websocket.send_json(result)                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: useInferenceSocket receives JSON                 │
│                                                              │
│ onmessage = (event) => {                                    │
│   const result = JSON.parse(event.data);                    │
│   ├─ detections = result.detections                         │
│   ├─ model1_detected = result.model1_detected               │
│   ├─ lenA1/lenA2 = measurements                             │
│   ├─ metrics = { fps, gpu_pct, latency_ms }                │
│   └─ Update Zustand store state                             │
│ }                                                            │
│                                                              │
│ Canvas renders:                                             │
│  - Green contours (Model 1 detections)                      │
│  - Blue bounding boxes (Model 2 detections)                 │
│  - Colored polygons (Model 3 mask regions)                  │
│  - Overlay text: FPS, GPU%, Latency, Status                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ User sees: LIVE INFERENCE RESULTS on stream 🚀             │
│                                                              │
│ Canvas shows:                                               │
│ ├─ Live camera stream (MJPEG HTTP)                          │
│ └─ Inference overlays (detections, measurements)            │
│                                                              │
│ Sidebar shows:                                              │
│ ├─ Model 1 Detected: True/False                             │
│ ├─ FPS: 25.3                                                │
│ ├─ GPU: 68.2%                                               │
│ ├─ Latency: 48.2 ms                                         │
│ └─ Status: "MODEL1_PASS", "WAITING_MODEL2", etc.            │
└─────────────────────────────────────────────────────────────┘
```

---

### FLOW 5: User Clicks INFERENCE Button (OFFLINE - Video File)

```
⚠️  KEY DIFFERENCE: No camera connected, video file uploaded

┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: User uploads video file                          │
│ (or selects folder of videos)                              │
│                                                              │
│ Sets: selectedVideoPath = "/path/to/video.mp4"              │
│ Or: selectedFolderPath = "/path/to/videos/"                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: User clicks "START INFERENCE" button             │
│ (visible - inference works with video OR camera)            │
│                                                              │
│ Source: "video" (offline mode)                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: visionProcessStore.startProcess({                 │
│   type: "inference",                                        │
│   source: "video"  ← OFFLINE MODE                           │
│ })                                                           │
│                                                              │
│ Creates: recordingSessionId = "inf_" + videoId              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: DO NOT call POST /oak/oak/stream/start            │
│ ⚠️  Only camera (online) mode starts streaming              │
│                                                              │
│ if (source === "camera")                                    │
│   POST /oak/oak/stream/start  ← NOT CALLED                  │
│                                                              │
│ (Video mode relies on base64 frames in WebSocket)           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: POST /oak/oak/inference/start/{sessionId}        │
│ Body: {                                                      │
│   offline: true,  ← OFFLINE FLAG                            │
│   videoPath: "/path/to/video.mp4"  ← SINGLE VIDEO           │
│   folderPath: null                                          │
│ }                                                            │
│                                                              │
│ (OR for folder mode:)                                       │
│ Body: {                                                      │
│   offline: true,                                            │
│   videoPath: null,                                          │
│   folderPath: "/path/to/videos/"  ← FOLDER                  │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: oak_camera_service.start_inference(                │
│   session_id,                                               │
│   loop,                                                     │
│   offline=True,  ← KEY FLAG                                │
│   video_path="/path/to/video.mp4"                           │
│ )                                                            │
│                                                              │
│ 1. Set state:                                               │
│    - _inference_offline = True  ← FLAG                      │
│    - _inference_session_id = session_id                     │
│    - _inference_stop.clear()                                │
│                                                              │
│ 2. Create asyncio queue for results:                        │
│    - _inference_result_queue = asyncio.Queue()              │
│    - _inference_loop = loop                                 │
│                                                              │
│ 3. Load video frames into _offline_frame_queue:            │
│                                                              │
│    SINGLE VIDEO MODE:                                       │
│      cap = cv2.VideoCapture(video_path)                    │
│      total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)       │
│      self._offline_total_frames = int(total_frames)         │
│      self._offline_frame_index = 0                          │
│                                                              │
│      while cap.isOpened():                                  │
│        ret, frame = cap.read()                              │
│        if not ret: break                                    │
│        self._offline_frame_queue.put(frame)  ← queue frame  │
│        self._offline_frame_index += 1                       │
│                                                              │
│      cap.release()                                          │
│      self._offline_frame_queue.put(None)  ← EOF marker      │
│                                                              │
│    FOLDER MODE:                                             │
│      videos = sorted(glob.glob(folder + "/*.mp4"))          │
│      self._total_videos = len(videos)                       │
│      self._current_video_idx = 0                            │
│                                                              │
│      for video in videos:                                   │
│        cap = cv2.VideoCapture(video)                       │
│        while cap.isOpened():                                │
│          ret, frame = cap.read()                            │
│          if not ret: break                                  │
│          self._offline_frame_queue.put(frame)               │
│          self._offline_frame_index += 1                     │
│        cap.release()                                        │
│        self._offline_frame_queue.put(_VIDEO_BOUNDARY)       │
│        ↑ marker between videos                              │
│        self._current_video_idx += 1                         │
│                                                              │
│      self._offline_frame_queue.put(None)  ← final EOF       │
│                                                              │
│ 4. Start inference worker thread (no watchdog for offline): │
│    - _inference_thread = Thread(_inference_worker)          │
│    - daemon=True, but will exit on EOF                      │
│                                                              │
│ Return: { status: "inference_started" }                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: _inference_worker loop (offline mode)              │
│                                                              │
│ while not _inference_stop.is_set():                         │
│                                                              │
│   1. Get frame from offline queue (BLOCKING):              │
│      frame = self._offline_frame_queue.get(timeout=0.1)     │
│                                                              │
│      if frame is None:  ← EOF reached                       │
│         send done_result with "INFERENCE_COMPLETE"          │
│         _inference_stop.set()                               │
│         break                                               │
│                                                              │
│      if frame is _VIDEO_BOUNDARY:  ← folder mode             │
│         reset_session_for_next_video(session_id)            │
│         continue to next video                              │
│                                                              │
│   2. Run AI inference:                                     │
│      result = run_inference(frame.copy(), session_id)       │
│                                                              │
│   3. Attach metadata:                                       │
│      result["frame_index"]   = current frame number         │
│      result["total_frames"]  = total frames in video        │
│      result["video_index"]   = current video (folder mode)  │
│      result["total_videos"]  = total videos (folder mode)   │
│      result["video_name"]    = "video_001.mp4"              │
│                                                              │
│   4. ENCODE FRAME TO BASE64 (offline-specific):             │
│      ⚠️  Online mode doesn't need this (has HTTP stream)    │
│      result["_raw_frame"] = bgr  ← lightweight reference    │
│      ← Will be encoded to base64 at router level             │
│                                                              │
│   5. Push to asyncio queue:                                │
│      asyncio.run_coroutine_threadsafe(                      │
│        _async_inference_push(result),                       │
│        _inference_loop                                      │
│      )                                                       │
│                                                              │
│   Loop continues until None received (EOF)                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: WebSocket inference endpoint                      │
│                                                              │
│ (Same as online mode, but different encoding:)              │
│                                                              │
│ while True:                                                 │
│   result = await oak_camera_service.get_inference_result()  │
│                                                              │
│   if result is None or result.get("done"):                  │
│      send and close                                         │
│      break                                                  │
│                                                              │
│   🔄 FRAME ENCODING (offline-specific):                     │
│   if "_raw_frame" in result:                                │
│      # Online mode doesn't have this key                    │
│      # Offline mode MUST encode because no HTTP stream      │
│                                                              │
│      bgr = result.pop("_raw_frame")                         │
│      loop = asyncio.get_running_loop()                      │
│      def _encode():                                         │
│        _, buf = cv2.imencode(".jpg", bgr,                  │
│                  [cv2.IMWRITE_JPEG_QUALITY, 60])            │
│        return "data:image/jpeg;base64," + base64.b64encode  │
│      result["frame"] = await loop.run_in_executor(_encode)  │
│      └─ Send as: result["frame"] = "data:image/jpeg;base   │
│         64,/9j/4AAQSkZJRgABA..."                            │
│                                                              │
│   Send to frontend:                                         │
│   await websocket.send_json(result)                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: useInferenceSocket receives JSON                 │
│                                                              │
│ onmessage = (event) => {                                    │
│   result = JSON.parse(event.data);                          │
│                                                              │
│   if (result.frame) {  ← Base64 JPEG string                │
│      // Offline mode: decode and display                    │
│      canvas.drawImage(new Image(src=result.frame))          │
│   }                                                          │
│   // Online mode: ignored (has separate HTTP stream)        │
│                                                              │
│   Update detections, metrics, status                        │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: Display results                                  │
│                                                              │
│ Sidebar progress:                                           │
│  Video: 1/1                                                 │
│  Frame: 120/1080  (or "2/5, 45/240" in folder mode)         │
│                                                              │
│ Overlays:                                                   │
│  - Detections drawn on base64 frame                         │
│  - Metrics: FPS, GPU%, Latency                              │
│  - Status: WAITING_MODEL1, MODEL1_PASS, etc.                │
│                                                              │
│ Speed:                                                      │
│  - Often FASTER than real-time (video file → no camera     │
│    latency, can process frames at GPU speed)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Inference completes (EOF reached)                          │
│                                                              │
│ WebSocket receives: { done: true, status: "INFERENCE_*" }   │
│ WebSocket closes automatically                              │
│                                                              │
│ Frontend:                                                   │
│  - Button returns to "Start" state                          │
│  - "Inference Complete" message shown                       │
│  - User can review results or start new inference           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎮 UI Button States & Lock Logic

### Button Hierarchy

```
┌─────────────────────────────────────────┐
│ DASHBOARD DOCK (DashboardDock.tsx)       │
│                                          │
│  [▶ STREAM]  [● REC]  [⚙ INFERENCE]    │
│                                          │
│ Each button has:                         │
│  - enabled/disabled state                │
│  - active indicator                      │
│  - loading spinner (while connecting)    │
│  - tooltip text                          │
└─────────────────────────────────────────┘
```

### STREAM Button Logic

```
ENABLED if:
  ✅ Camera connected (cameraEnabled)
  ✅ NOT already streaming/recording/inferencing
  ✅ NOT in video mode
  ✅ Not loading (isConnecting == false)

DISABLED if:
  ❌ Camera not connected
  ❌ Another process active
  ❌ Video mode (can only use camera for streaming)
  ❌ Loading/connecting

STATE:
  Inactive:  Play icon, click to start
  Active:    Stop Square icon, click to stop
  Loading:   Spinner animation
```

### RECORD Button Logic

```
ENABLED if:
  ✅ Camera connected
  ✅ Streaming is ACTIVE (requirement!)
  ✅ NOT already recording
  ✅ NOT in video mode
  ✅ Not loading

DISABLED if:
  ❌ Camera not connected
  ❌ Streaming not active (streaming must start first!)
  ❌ Another process active
  ❌ Video mode
  ❌ Loading

STATE:
  Inactive:  Circle icon (red dot)
  Active:    Stop Square icon
  Loading:   Spinner animation
```

### INFERENCE Button Logic

```
ENABLED if:
  ✅ (Camera connected) OR (Video uploaded/folder selected)
  ✅ NOT already inferencing
  ✅ Not loading

DISABLED if:
  ❌ Neither camera nor video available
  ❌ Another process active
  ❌ Loading

STATE:
  Inactive:  CPU icon
  Active:    Stop Square icon
  Loading:   Spinner animation

NOTE: Can work in BOTH online (camera) and offline (video) modes
```

### Lock Mechanism (useDashboardDock.ts)

```typescript
isLocked = (processType: string) => {
  return (
    isConnecting ||                    // UI is loading
    isRendering ||                     // previous process rendering
    activeProcess !== null &&          // process running
    activeProcess !== processType      // different process
  );
};

// Button click:
disabled = 
  isConnecting ||
  isLocked("streaming") ||
  !cameraEnabled ||
  isVideoMode;  // streaming requires camera
```

---

## 🔀 Switching Between Processes

### Scenario: User Switches from Streaming to Recording

```
Initial state:
  activeProcess: "streaming"
  Button state: Stream=ACTIVE, Record=ENABLED, Inference=DISABLED

User clicks RECORD:
  1. toggleProcess("recording") called
  2. Check isLocked("recording") → false (streaming is ok)
  3. activeProcess changes: "streaming" → "recording"
  4. globalStreamCleanup() called (clears old stream resources)
  5. New process starts
  6. Recording active, Stream still running, Inference locked

Button effects:
  - Stream: Changes to STOP icon (active but secondary)
  - Record: Changes to STOP icon (active and primary)
  - Inference: DISABLED (isLocked)
```

### Scenario: User Switches from Inference to Recording

```
Initial state:
  activeProcess: "inference"
  source: "camera"  (live inference)
  Button state: Stream=ACTIVE, Record=ENABLED, Inference=ACTIVE

User clicks RECORD:
  1. POST /oak/oak/inference/stop (stop inference)
  2. globalStreamCleanup() clears inference UI state
  3. activeProcess changes: "inference" → "recording"
  4. POST /oak/oak/stream/start (restart streaming if needed)
  5. POST /recording/start (begin recording)

Button effects:
  - Stream: ACTIVE ✓
  - Record: ACTIVE ✓
  - Inference: DISABLED (isLocked)
```

---

## 🌍 Online vs Offline Comparison

| Feature | Online (Camera) | Offline (Video) |
|---------|-----------------|-----------------|
| **Source** | OAK-D Camera (IP-connected) | Video file or folder |
| **Requires Camera** | ✅ Yes | ❌ No |
| **Stream Preview** | ✅ HTTP MJPEG at `/oak/oak/stream` | ❌ No, base64 in WebSocket |
| **Frame Rate** | 📊 Live (depends on auto-exposure) | 📊 File speed (often faster) |
| **Recording** | ✅ PyAV with wall-clock timestamps | ⚠️ Not directly supported |
| **Streaming Button** | ✅ Independent control | ❌ Disabled |
| **Recording Button** | ✅ Requires streaming active | ❌ Disabled |
| **Inference Button** | ✅ Auto-starts streaming | ✅ Processes frames sequentially |
| **Watchdog Restart** | ✅ Yes (worker crash recovery) | ❌ No (can't reposition video) |
| **Progress Tracking** | 📊 FPS/GPU metrics only | 📊 FPS + Frame/Video index |
| **Frame Encoding** | ❌ No (separate HTTP stream) | ✅ Yes (base64 in WebSocket) |
| **Multi-Video Support** | ❌ Single camera only | ✅ Folder mode supported |
| **Session Reset** | ❌ N/A | ✅ At video boundaries |

---

## ⚙️ Architecture: Thread Coordination

### Online Mode Thread Map

```
Main FastAPI Thread
  ├─ HTTP GET /oak/oak/stream
  │   └─ Reads from asyncio stream_queue
  │
  ├─ HTTP POST /oak/oak/stream/start
  │   └─ Signals _start_capture_threads()
  │
  └─ HTTP POST /oak/oak/inference/start
      └─ Signals _start_inference_worker()

Capture Threads (spawned by start_streaming):
  ├─ oak-mjpeg thread (captures MJPEG)
  │   ├─ Drains _mjpeg_dai_queue (blocking loop)
  │   └─ Pushes JPEG to asyncio stream_queue
  │
  ├─ oak-hires thread (captures raw NV12)
  │   ├─ Drains _raw_dai_queue (blocking loop)
  │   └─ Queues packets to _hires_packet_queue
  │
  └─ oak-convert thread (converts frames)
      ├─ Consumes from _hires_packet_queue
      ├─ getCvFrame() → BGR
      ├─ Stores in _latest_bgr
      └─ Feeds to recorder (if active)

Inference Thread (spawned by start_inference):
  ├─ _inference_worker thread
  │   ├─ Reads _latest_bgr (camera mode)
  │   ├─ Runs run_inference()
  │   └─ Pushes result to _inference_result_queue (asyncio)
  │
  └─ _inference_watchdog thread (online only)
      ├─ Monitors _inference_worker
      └─ Restarts if it dies (max 5 times)

GPU Sampler Thread:
  └─ Samples nvidia-smi every 2 seconds
      └─ Updates _gpu_usage (non-blocking)

WebSocket Handler (async):
  └─ Reads _inference_result_queue
      └─ Sends JSON to connected clients

Recording Thread:
  └─ RecordingSession._worker
      ├─ Reads from frame_queue
      ├─ Encodes frames (MJPEG/FFV1)
      └─ Writes to .avi/.mkv file
```

### Offline Mode Thread Map

```
Main FastAPI Thread
  ├─ HTTP POST /oak/oak/inference/start
  │   └─ Loads all video frames into _offline_frame_queue
  │
  └─ WebSocket /oak/oak/inference/ws/
      └─ Reads from _inference_result_queue

Offline Loader (in start_inference):
  ├─ Opens cv2.VideoCapture(video_path)
  ├─ Reads all frames sequentially
  ├─ Queues to _offline_frame_queue
  └─ Adds None (or _VIDEO_BOUNDARY) at end

Inference Thread:
  ├─ _inference_worker thread
  │   ├─ Reads from _offline_frame_queue (blocking, sequential)
  │   ├─ Runs run_inference()
  │   ├─ Encodes to base64 (if needed)
  │   └─ Pushes result to _inference_result_queue
  │
  └─ NO watchdog (can't reposition video safely)

GPU Sampler Thread:
  └─ Same as online

WebSocket Handler (async):
  ├─ Reads _inference_result_queue
  ├─ Encodes raw frame to base64 (if present)
  └─ Sends JSON to client
```

---

## 🚨 Error Handling & Recovery

### Connection Failure

```
POST /oak/oak/start
  └─ dai.Device(ip) raises exception
     └─ Caught: logger.error("[DEVICE] Connection failed: {e}")
        └─ Logs to realtime_log: "CAMERA Connection failed ({ip})"
           └─ is_connected = False
           └─ Returns: { status: "error", message: "Camera not configured" }
              └─ Frontend shows Offline mode
                 └─ Stream/Record disabled, Inference for video only
```

### Streaming Thread Death

```
_mjpeg_thread or _hires_thread crashes:
  └─ Exception caught in thread target
     └─ Exception logged
     └─ Thread exits
     └─ HTTP clients still connected but receive no frames
        └─ Eventually timeout

User can:
  - Click stop then start (restart threads)
  - Check logs for root cause
```

### Inference Worker Death (Online Only)

```
_inference_worker crashes:
  └─ Exception logged
  └─ _inference_watchdog detects thread not alive
     └─ If offline: skip restart (can't reposition video)
     └─ If online:
        └─ Restart limit check (max 5)
           ├─ If < 5: spawn new worker thread
           │  └─ WebSocket clients get no interruption ✓
           │  └─ Inference resumes from next frame
           └─ If >= 5: log critical error, give up
              └─ WebSocket clients timeout waiting for results
```

### SQLite Busy Lock

```
_inference_worker tries to load config:
  db.session() → database locked
     └─ Exception caught
     └─ Falls back to last_known_config
     └─ If no fallback: worker thread dies
        └─ Watchdog restarts it (online mode)
```

### WebSocket Client Disconnect

```
Frontend closes connection or network fails:
  └─ asyncio detects disconnect
     └─ Exception: WebSocketDisconnect caught
     └─ Logs client disconnected message
     └─ Inference worker continues running (will push to closed queue)
        └─ Eventually times out

Backend continues inference until:
  - User clicks STOP
  - Inference reaches EOF (offline)
```

---

## 📊 Metrics & Monitoring

### Real-Time Metrics (Logged Every Frame)

```
FPS (Frames Per Second):
  - Calculated: frame_count / elapsed_time (0.5s rolling window)
  - For online: based on inference iteration rate
  - For offline: based on file read + inference rate
  - Normal: 20-30 FPS (depends on camera/video + inference)

GPU Usage (%):
  - Source: nvidia-smi --query-gpu=utilization.gpu
  - Sampled: every 2 seconds (non-blocking)
  - Range: 0-100%, or -1 if GPU unavailable
  - Typical: 60-85% during heavy inference

Latency (milliseconds):
  - Measured: time.perf_counter() around run_inference()
  - Time for single inference pass on single frame
  - Typical: 40-80 ms depending on model complexity
  - NOT including queue wait time

Capture FPS (Converter thread only):
  - Monitors camera frame delivery rate
  - Logged every 2 seconds
  - Flags "LOW" if below 80% of configured FPS
  - Indicates if auto-exposure or congestion
```

### Logging

```
[DEVICE] Connection to {ip} ...
[PIPELINE] Building...
[PIPELINE] {width}x{height} @ {fps}fps
[CAPTURE] MJPEG thread started
[CAPTURE] Hi-res thread started
[CAPTURE] Converter thread started
[STREAM] Queue opened
[STREAM] Frames sent: N | avg FPS: X.X
[MJPEG] FPS: 29.5 | pushed: 100 | dropped: 2
[HIRES] Thread running
[CONVERT] FPS: 28.3 [OK] | recording: True | record_frames_fed: 50
[RECORD] Started: /path/to/video.avi
[RECORD] Frame N encoded | PTS: 12345 ms
[RECORD] Stopped — frames written: 150
[INFERENCE] Thread running — session: inf_123456
[INFERENCE] status=MODEL1_PASS | record=True | fps=25.2 | gpu=72.1% | latency=45.3ms
[INFERENCE] Worker died unexpectedly — restarting (attempt 1/5)
[INFERENCE WS] Client connected — session: inf_123456
[INFERENCE WS] Terminal result received — closing
```

---

## 🔑 Key Implementation Details

### Why PyAV for Recording?

```
cv2.VideoWriter Problem:
  - Stamps every frame at (frame_number / declared_fps)
  - If camera delivers 3.2 FPS instead of 10 FPS:
    → 60 seconds of recording becomes only 19 seconds!

PyAV Solution:
  - Uses wall-clock time for each frame: container.encode(frame)
  - Per-frame PTS (presentation timestamp) based on actual arrival time
  - Time base 1/1000 = 1 ms precision
  - Result: always correct duration regardless of camera FPS
```

### Why Frame Normalization in Inference?

```
Problem:
  - Recorded video encodes MJPEG (lossy JPEG)
  - Offline inference reads back the .avi file
  - JPEG decode introduces artifacts
  - Live inference sees raw camera frames (no artifacts)
  - Results differ between online and offline!

Solution:
  - During online inference:
    - Encode frame to JPEG (quality 95)
    - Decode it back
    - Run inference on decoded (artifact-containing) frame
  - Result matches what offline inference will see
    - Ensures consistency between live and recorded results
```

### Offline Frame Queue: Blocking vs Non-Blocking

```
_offline_frame_queue = Queue(maxsize=32):
  - Producer (loader thread): puts frames sequentially
  - Consumer (inference worker): BLOCKING get(timeout=0.1)
    
Advantages:
  ✅ Processes EVERY frame exactly once (no drops)
  ✅ Sequential order guaranteed
  ✅ Worker waits if frame not ready (no busy-loop)
  ✅ Supports multi-video folder mode with _VIDEO_BOUNDARY sentinel

Difference from online:
  Online: reads _latest_bgr (most recent, may skip frames if slow)
  Offline: _offline_frame_queue (sequential, never skips)
```

### Why No Watchdog for Offline Inference?

```
Online mode watchdog can restart worker:
  - Camera streams continuously
  - Can just grab next available frame
  - If worker dies: start new one from next frame

Offline mode cannot restart safely:
  - File position is specific (frame 120/500)
  - If worker dies after processing frame 120:
    └─ Restart loads entire file again
    └─ Queue fills with frames 1-50 while consumer at frame 120
    └─ Result: incorrect frame order or gaps

Solution:
  - For offline: single worker, no restart
  - If it crashes: user must retry (rare)
  - Avoids frame ordering corruption
```

---

## 🎯 Summary Table: Button Behavior

| Button | Source | Enabled When | Disabled When | Auto-Starts | Locks |
|--------|--------|--------------|---------------|-------------|-------|
| **Stream** | Camera | Camera connected | Camera offline / already streaming / video mode | - | Recording/Inference |
| **Record** | Camera | Streaming active | No camera / no stream / video mode | - | Streaming/Inference |
| **Inference** | Camera OR Video | Camera OR video | Another process active | Stream (camera only) | Recording/Streaming |

---

## 🔗 API Endpoint Reference

### Camera Connection
- `POST /oak/oak/start` - Connect device & build pipeline
- `POST /oak/oak/stop` - Disconnect device
- `GET /oak/oak/health` - Check connection status

### Streaming
- `POST /oak/oak/stream/start` - Start capture threads
- `POST /oak/oak/stream/stop` - Stop capture threads
- `GET /oak/oak/stream` - MJPEG HTTP stream (SSE multipart)
- `GET /oak/oak/snapshot` - Single JPEG frame

### Recording
- `POST /recording/start` - Begin recording session
- `POST /recording/stop` - End recording session

### Inference
- `POST /oak/oak/inference/start/{session_id}` - Start inference
- `POST /oak/oak/inference/stop` - Stop inference
- `GET /oak/oak/inference/ws/{session_id}` - WebSocket for results

### Camera Controls (Runtime)
- `POST /oak/oak/controls` - Update exposure, gain, focus, brightness, contrast

---

## 📚 File Reference

### Backend
- [oak_camera_service.py](d:\DHTX\DHTX-BACKEND-AUTO-ANNOTATION-OLD\app\services\oak_camera_service.py) - Core service (1500+ lines)
- [oak_camera_router.py](d:\DHTX\DHTX-BACKEND-AUTO-ANNOTATION-OLD\app\routers\oak_camera_router.py) - API routes
- [recording_service.py](d:\DHTX\DHTX-BACKEND-AUTO-ANNOTATION-OLD\app\services\recording_service.py) - PyAV recorder
- [camera_model.py](d:\DHTX\DHTX-BACKEND-AUTO-ANNOTATION-OLD\app\models\camera_model.py) - DB model

### Frontend
- [visionProcessStore.ts](d:\DHTX\DHTX-FRONTEND-AUTO-ANNOTATION-OLD\src\store\visionProcessStore.ts) - Process lifecycle
- [DashboardDock.tsx](d:\DHTX\DHTX-FRONTEND-AUTO-ANNOTATION-OLD\src\components\dashboard\DashboardDock.tsx) - Button UI
- [useInferenceSocket.ts](d:\DHTX\DHTX-FRONTEND-AUTO-ANNOTATION-OLD\src\hooks\useInferenceSocket.ts) - WebSocket hook
- [cameraService.ts](d:\DHTX\DHTX-FRONTEND-AUTO-ANNOTATION-OLD\src\services\cameraService.ts) - API client

---

**Document Generated:** 2026-09-01  
**System Version:** DHTX Auto-Annotation  
**Analysis Scope:** Complete online/offline camera logic architecture
