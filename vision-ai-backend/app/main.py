from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import ResponseValidationError, RequestValidationError
import os
import sys
import uuid
import subprocess
from pathlib import Path

# --- PyInstaller PyTorch CUDA DLL Fix ---
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    torch_lib = os.path.join(bundle_dir, 'torch', 'lib')
    if os.path.exists(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
# ----------------------------------------

from app.api.router import router
from app.core.database import Base, engine
from app.utils.exceptions import AppException

from app.models.camera_model import Camera
from app.models.recording_model import Recording
from app.utils.resource_path import resource_path
import time
from app.core.logger import setup_logger
from app.services.realtime_log_service import realtime_log_service

logger = setup_logger("vision-ai")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Project root = parent of the /app folder
_ROOT = Path(__file__).parent.parent   # DHTX-BACKEND/


# ─────────────────────────────────────────
# BOOTSTRAP: .env
# ─────────────────────────────────────────
def _ensure_env_file():
    # Packaged applications keep configuration in the user's data directory;
    # their bundled resources (including an AppImage) are read-only.
    if getattr(sys, "frozen", False):
        return
    env_path = _ROOT / ".env"
    if not env_path.exists():
        env_path.write_text(
            "DATABASE_URL=sqlite:///./vision_ai.db\n"
            "APP_NAME=Vision AI Command Center\n"
            "DEBUG=True\n",
            encoding="utf-8",
        )

_ensure_env_file()


# ─────────────────────────────────────────
# BOOTSTRAP: DB tables + auto column migration
# ─────────────────────────────────────────
def _ensure_tables():
    try:
        from sqlalchemy import inspect, text
        from app.core.database import Base, engine

        # Create missing tables
        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)

        # =====================================================
        # cameras table migration
        # =====================================================
        if "cameras" in inspector.get_table_names():
            existing_camera_columns = [
                column["name"]
                for column in inspector.get_columns("cameras")
            ]

            camera_migrations = []

            if "recording_video_testing_path" not in existing_camera_columns:
                camera_migrations.append(
                    "ALTER TABLE cameras "
                    "ADD COLUMN recording_video_testing_path VARCHAR"
                )

            if "recording_format" not in existing_camera_columns:
                camera_migrations.append(
                    "ALTER TABLE cameras "
                    "ADD COLUMN recording_format VARCHAR(20) DEFAULT 'MP4'"
                )

            if camera_migrations:
                with engine.begin() as conn:
                    for sql in camera_migrations:
                        conn.execute(text(sql))

    except Exception as e:
        logger.warning(f"[BOOTSTRAP] Table migration check: {e}")

_ensure_tables()





# ---------------------------------------------------
#  FastAPI App — with startup lifespan
# ---------------------------------------------------
import threading
from contextlib import asynccontextmanager

# Set to True once the ML model + GPU are fully warmed up at startup.
# /health returns "starting" until this is True.
_backend_ready = False

def _preload_model_background():
    """Runs in a daemon thread at startup: loads TubeAnalyzer (and warms up the
    GPU if CUDA is available) so inference is instant when the user clicks Start.
    Sets _backend_ready=True when done so /health flips to 'ready'."""
    global _backend_ready
    try:
        from app.utils.resource_path import resource_path

        try:
            from app.ml.tube.inference.tube_analyzer import TubeAnalyzer
        except ImportError as e:
            logger.error(f"[STARTUP] Could not import TubeAnalyzer: {e}")
            TubeAnalyzer = None

        ml_tube_dir = os.path.join(os.path.dirname(__file__), "ml", "tube")
        model_candidates = [
            resource_path("app/ml/tube/model/best.pt"),
            os.path.join(ml_tube_dir, "model", "best.pt"),
            resource_path("app/ml/model/best.pt"),
            os.path.join(os.path.dirname(__file__), "ml", "model", "best.pt"),
        ]
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            model_candidates.insert(0, os.path.join(sys._MEIPASS, "app", "ml", "tube", "model", "best.pt"))
            model_candidates.insert(1, os.path.join(sys._MEIPASS, "app", "ml", "model", "best.pt"))

        resolved_model = next((c for c in model_candidates if c and os.path.exists(c)), None)

        if TubeAnalyzer is not None and resolved_model:
            logger.info(f"[STARTUP] Preloading TubeAnalyzer model from {resolved_model} into memory (GPU if available)...")

            # Determine device
            try:
                from app.ml.tube.inference.video_inference_service import is_cuda_operational
                device_val = "0" if is_cuda_operational() else "cpu"
            except Exception:
                device_val = "cpu"

            # Instantiating TubeAnalyzer loads YOLO weights + optionally CUDA
            _analyzer = TubeAnalyzer(model_path=resolved_model, device=device_val)

            # Warm up: run one dummy inference so CUDA kernels are compiled.
            try:
                import numpy as np
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _analyzer.process_frame(dummy)
                logger.info(f"[STARTUP] ML model warm-up complete — GPU kernels compiled (device={device_val}).")
            except Exception as warm_err:
                logger.warning(f"[STARTUP] Warm-up inference failed (non-fatal): {warm_err}")

            # Store it as the shared analyzer so VideoInferenceService reuses it
            try:
                from app.ml.tube.inference.video_inference_service import _set_shared_analyzer
                _set_shared_analyzer(_analyzer)
            except Exception:
                pass

            # Also inject into camera_inference_service so live camera sessions don't reload either
            try:
                from app.ml.tube.inference.camera_inference_service import _set_camera_analyzer
                _set_camera_analyzer(_analyzer)
            except Exception:
                pass
        else:
            logger.info("[STARTUP] TubeAnalyzer or model best.pt not found — skipping preload.")
    except Exception as e:
        logger.error(f"[STARTUP] Model preload failed: {e}", exc_info=True)
    finally:
        _backend_ready = True
        logger.info("[STARTUP] Backend ready.")


@asynccontextmanager
async def lifespan(app_instance):
    """FastAPI lifespan: kick off model preload in background, yield (app accepts
    requests), then clean up on shutdown."""
    thread = threading.Thread(target=_preload_model_background, daemon=True, name="model-preload")
    thread.start()
    yield
    # Shutdown: nothing special needed — daemon thread exits with the process


app = FastAPI(
    title="Vision AI Command Center",
    version="1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------
#  GLOBAL EXCEPTION HANDLERS
# ---------------------------------------------------
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    req_id = getattr(request.state, "request_id", "req_unknown")
    logger.warning(
        f"[APP EXCEPTION] [{req_id}] {request.method} {request.url.path} -> {exc.status_code}: {exc.message}"
    )
    realtime_log_service.add_log(
        "system",
        "WARN",
        f"{request.method} {request.url.path} -> {exc.message}",
        "warning"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": False, "message": exc.message, "data": None, "request_id": req_id}
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    req_id = getattr(request.state, "request_id", "req_unknown")
    logger.error(
        f"[RESPONSE VALIDATION ERROR] [{req_id}] {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    realtime_log_service.add_log(
        "system",
        "CRASH",
        f"Validation error on {request.method} {request.url.path}",
        "error"
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": False,
            "message": "Response validation error (endpoint returned invalid data)",
            "request_id": req_id,
            "detail": str(exc)
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "req_unknown")
    logger.error(
        f"[CRASH 500] [{req_id}] Unhandled exception in {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    realtime_log_service.add_log(
        "system",
        "CRASH",
        f"500 on {request.method} {request.url.path}: {exc}",
        "error"
    )
    return JSONResponse(
        status_code=500,
        content={"status": False, "message": "Internal Server Error", "request_id": req_id, "data": None}
    )


import json
from starlette.responses import Response, StreamingResponse
from app.core.logger import setup_logger, current_request_id

# Paths whose responses must NOT be buffered/read (streaming / binary / websocket-adjacent)
STREAMING_PATH_PREFIXES = (
    "/oak/stream",
    "/video/stream",
    "/oak/inference/ws",
    "/oak/snapshot",
    "/video/last_frame",
)

# ---------------------------------------------------
#  API REQUEST & CRASH LOGGING MIDDLEWARE
# ---------------------------------------------------
@app.middleware("http")
async def api_logging_middleware(request: Request, call_next):
    start_time = time.time()
    req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:8]}"
    request.state.request_id = req_id
    token = current_request_id.set(req_id)
    method = request.method
    path = request.url.path

    try:
        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 1)
            logger.error(
                f"[API CRASH] {method} {path} failed after {duration_ms}ms: {e}",
                exc_info=True,
            )
            realtime_log_service.add_log(
                "system",
                "CRASH",
                f"Unhandled error in {method} {path}: {e}",
                "error"
            )
            raise e

        duration_ms = round((time.time() - start_time) * 1000, 1)

        # Never buffer streaming or binary responses — just log status/timing
        is_streaming = (
            isinstance(response, StreamingResponse)
            or any(path.startswith(p) for p in STREAMING_PATH_PREFIXES)
        )

        if is_streaming:
            if response.status_code >= 400:
                logger.warning(f"[API WARN] {method} {path} -> {response.status_code} ({duration_ms}ms)")
            else:
                logger.info(f"[API] {method} {path} -> {response.status_code} ({duration_ms}ms)")
            response.headers["X-Request-ID"] = req_id
            return response

        if response.status_code >= 400:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    detail = parsed.get("detail") or parsed.get("message") or parsed
                else:
                    detail = parsed
            except Exception:
                detail = body.decode(errors="ignore")[:300]

            if response.status_code >= 500:
                logger.error(
                    f"[API ERROR] {method} {path} -> {response.status_code} ({duration_ms}ms) | {detail}"
                )
            else:
                logger.warning(
                    f"[API WARN] {method} {path} -> {response.status_code} ({duration_ms}ms) | {detail}"
                )

            # Rebuild response since body_iterator was consumed
            headers = dict(response.headers)
            headers["content-length"] = str(len(body))
            headers["X-Request-ID"] = req_id
            response = Response(
                content=body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )
            return response
        else:
            is_routine_poll = (
                path in ("/health", "/oak/health")
                or path.startswith("/video/status/")
                or path.startswith("/oak/inference/status/")
            )
            if not is_routine_poll:
                logger.info(f"[API] {method} {path} -> {response.status_code} ({duration_ms}ms)")

            response.headers["X-Request-ID"] = req_id
            return response
    finally:
        current_request_id.reset(token)


# ---------------------------------------------------
#  CORS
# ---------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1",
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "app://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
#  STATIC FILES (if storage directory exists)
# ---------------------------------------------------
storage_dir = resource_path("storage")
if os.path.exists(storage_dir):
    app.mount(
        "/storage",
        StaticFiles(directory=storage_dir),
        name="storage"
    )

# ---------------------------------------------------
#  ROOT HEALTH CHECK
# ---------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ready", "message": "Vision AI Backend Running", "data": None}


@app.get("/health")
async def get_health():
    """Health check endpoint for frontend readiness polling.
    Returns 'starting' while the ML model is loading, 'ready' once it is done.
    Electron's waitForBackend() polls this until it sees 'ready'."""
    if not _backend_ready:
        # FastAPI is up but model is still loading — tell Electron to keep waiting
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "Backend is loading ML model...", "data": None}
        )
    return {"status": "ready", "message": "Backend is healthy", "data": None}


@app.post("/api/system/shutdown")
@app.post("/shutdown")
def shutdown_system():
    """Endpoint for graceful desktop exit requested by Electron."""
    import threading
    import signal

    def _delayed_exit():
        time.sleep(0.5)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "shutting_down", "message": "Backend terminating cleanly"}


# ---------------------------------------------------
#  REGISTER ROUTES
# ---------------------------------------------------
app.include_router(router)

from app.routers.debug_router import router as debug_router
app.include_router(debug_router)

# ---------------------------------------------------
#  STARTUP EVENTS
# ---------------------------------------------------
# Auth/Role seeders have been removed.
