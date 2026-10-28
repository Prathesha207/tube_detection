"""
main.py - Head/Tail ML Microservice
====================================
Pure ML microservice wrapping the installed wheel.
All paths are 100% relative to this directory.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
import time
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

# Load directly from the installed wheel
try:
    from head_tail_analyzer.main import AnalyzerError, HeadTailAnalyzer
except ImportError as err:
    raise ImportError(
        "Could not import HeadTailAnalyzer. Please install the wheel from the wheels/ directory:\n"
        "pip install wheels/head_tail_analyzer-*.whl"
    ) from err

# ----------------------------------------------------------------------
# 1. Base Paths (100% Relative to THIS file's folder)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tube-ml-api")

# ----------------------------------------------------------------------
# 2. Global State & Lock
# ----------------------------------------------------------------------
analyzer: Optional[HeadTailAnalyzer] = None
analyzer_lock = threading.Lock()
last_frame_time = 0.0
IDLE_RESET_SECONDS = 10.0


# ----------------------------------------------------------------------
# 3. Model Lifespan: Loads ONCE at Startup
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the ML model once on startup and keeps it in VRAM."""
    global analyzer
    logger.info("Initializing HeadTailAnalyzer from config: %s", CONFIG_PATH)
    
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")

    # HeadTailAnalyzer resolves relative model paths from config.yaml's folder
    analyzer = HeadTailAnalyzer(config=str(CONFIG_PATH))
    t0 = time.perf_counter()
    analyzer.load_model()
    logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    # Warm up GPU once with a dummy frame
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    analyzer.process_frame(dummy)
    analyzer.reset()
    logger.info("GPU warm-up completed. Service is ready!")

    yield

    logger.info("Shutting down ML service...")
    if analyzer is not None:
        try:
            analyzer.unload_model()
        except Exception:
            pass
        analyzer = None


# ----------------------------------------------------------------------
# 4. FastAPI App Definition & Exception Handling
# ----------------------------------------------------------------------
app = FastAPI(
    title="Tube Detection ML Microservice",
    description="Receives camera or video frames and returns pure detection JSON.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AnalyzerError)
async def ml_error(request: Request, exc: AnalyzerError):
    """Custom exception handler for wheel-specific errors."""
    return JSONResponse({"detail": f"ML error: {exc}"}, status_code=500)


# ----------------------------------------------------------------------
# 5. Worker Functions (Threadpool-Safe)
# ----------------------------------------------------------------------
def run_ml(frame: np.ndarray, new_video: bool, frame_index: Optional[int]):
    """Plain function running in a worker threadpool: new-video check + the wheel call."""
    global last_frame_time
    now = time.time()
    with analyzer_lock:
        if new_video or (now - last_frame_time > IDLE_RESET_SECONDS):
            analyzer.reset()
        last_frame_time = now
        t0 = time.perf_counter()
        result = analyzer.process_frame(frame, frame_index=frame_index)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return result, latency_ms


# ----------------------------------------------------------------------
# 6. Endpoints
# ----------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    """Health check for uptime monitors."""
    ready = analyzer is not None and getattr(analyzer, "is_loaded", True)
    return {
        "status": "ready" if ready else "not_ready",
        "model_loaded": ready,
        "config": str(CONFIG_PATH.name),
    }


@app.post("/infer", tags=["Inference"])
async def infer(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Image frame file (multipart upload)"),
    frame_index: Optional[int] = Query(None, description="Optional frame sequence number"),
    new_video: bool = Query(False, description="Set to true on the first frame of a new video"),
):
    """
    Primary Inference Endpoint:
    - Accepts frames from cameras or video files.
    - Converts input bytes into a NumPy BGR array.
    - Runs in worker threadpool (run_in_threadpool) so the event loop stays non-blocking.
    - Returns all detections, counts, and measurements as JSON.
    """
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Analyzer is not initialized.",
        )

    # 1. Read binary image bytes (supports multipart upload OR raw binary stream)
    if file is not None:
        data = await file.read()
    else:
        data = await request.body()

    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty frame received (0 bytes).")

    # 2. Decode bytes to NumPy BGR array
    np_buffer = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if frame is None or getattr(frame, "size", 0) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode image into a NumPy array. Unsupported or corrupted image format.",
        )

    # 3. New-video check and the ML call in a worker thread so the server stays responsive
    result, latency_ms = await run_in_threadpool(run_ml, frame, new_video, frame_index)

    return JSONResponse(content=result, headers={"X-Latency-Ms": str(latency_ms)})



@app.post("/reset", tags=["Inference"])
def reset():
    """Manual reset of tracker history."""
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Analyzer is not initialized.",
        )
    with analyzer_lock:
        analyzer.reset()
    logger.info("Tracker state reset.")
    return {"status": "ok", "message": "Tracker history cleared."}


if __name__ == "__main__":
    import argparse
    import socket
    import uvicorn

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    default_port = 8001 if is_port_in_use(8000) else 8000

    parser = argparse.ArgumentParser(description="Head/Tail Analyzer ML Microservice")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"\n" + "=" * 60)
    print(f"  👉 API Docs (Click to open): http://127.0.0.1:{args.port}/docs")
    print(f"  👉 Health Check:             http://127.0.0.1:{args.port}/health")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=args.host, port=args.port)
