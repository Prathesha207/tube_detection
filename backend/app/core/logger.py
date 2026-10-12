import logging
import os
import contextvars
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Optional

# Max 5 MB per hourly log file, keep 3 rotated backups
_MAX_BYTES    = 5 * 1024 * 1024
_BACKUP_COUNT = 3

current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_request_id", default="")

_shared_handlers: List[logging.Handler] = []
_error_handlers: List[RotatingFileHandler] = []
_console_handler: Optional[logging.Handler] = None
_realtime_handler: Optional[logging.Handler] = None
_retention_cleaned = False


class RequestIdFilter(logging.Filter):
    """Injects current_request_id into log records so all logs within a request share the same correlation ID."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "req_id"):
            req_id = current_request_id.get()
            record.req_id = f"[{req_id}] " if req_id else ""
        return True


class HourlyFileHandler(logging.Handler):
    """Dynamically writes logs to base_dir/YYYY-MM-DD/HH/session_HH.log.
    Automatically rotates to a new hour/date folder as time advances."""
    def __init__(
        self,
        base_dir: str,
        max_bytes: int = _MAX_BYTES,
        backup_count: int = _BACKUP_COUNT,
        encoding: str = "utf-8",
    ):
        super().__init__()
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self._current_key: Optional[str] = None
        self._active_handler: Optional[RotatingFileHandler] = None

    def _get_active_handler(self) -> RotatingFileHandler:
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        hour_folder = now.strftime("%H")
        key = f"{date_folder}/{hour_folder}"

        if self._active_handler is None or key != self._current_key:
            if self._active_handler is not None:
                try:
                    self._active_handler.close()
                except Exception:
                    pass
            full_path = os.path.join(self.base_dir, date_folder, hour_folder)
            os.makedirs(full_path, exist_ok=True)
            log_file_path = os.path.join(full_path, f"session_{hour_folder}.log")
            self._active_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding=self.encoding,
            )
            if self.formatter:
                self._active_handler.setFormatter(self.formatter)
            for f in self.filters:
                self._active_handler.addFilter(f)
            self._current_key = key

        return self._active_handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._get_active_handler().emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._active_handler is not None:
            try:
                self._active_handler.close()
            except Exception:
                pass
        super().close()


class RealtimeLogHandler(logging.Handler):
    """Pushes WARNING and ERROR log records into realtime_log_service so the frontend sees them."""
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING:
            return
        try:
            from app.services.realtime_log_service import realtime_log_service
            name = (record.name or "").lower()
            if "record" in name:
                cat = "record"
            elif "stream" in name:
                cat = "stream"
            elif "infer" in name or "analyzer" in name:
                cat = "inference"
            elif "camera" in name or "oak" in name:
                cat = "camera"
            else:
                cat = "system"

            tag = "CRASH" if record.levelno >= logging.ERROR else "WARN"
            msg = record.getMessage()
            status = "error" if record.levelno >= logging.ERROR else "warning"
            realtime_log_service.add_log(cat, tag, msg, status)
        except Exception:
            pass


def clean_old_logs(base_log_dir: str, retention_days: int = 7) -> None:
    """Deletes dated log folders older than `retention_days` to prevent disk bloat."""
    try:
        if not os.path.exists(base_log_dir):
            return

        now = datetime.now()
        for entry in os.listdir(base_log_dir):
            entry_path = os.path.join(base_log_dir, entry)
            if os.path.isdir(entry_path):
                try:
                    folder_date = datetime.strptime(entry, "%Y-%m-%d")
                    age_days = (now - folder_date).days
                    if age_days > retention_days:
                        import shutil
                        shutil.rmtree(entry_path, ignore_errors=True)
                except ValueError:
                    # Not a YYYY-MM-DD folder (e.g. error.log or other subfolder)
                    pass
    except Exception:
        pass


def get_current_log_path() -> Optional[str]:
    for h in _shared_handlers:
        if isinstance(h, HourlyFileHandler) and h._active_handler:
            return h._active_handler.baseFilename
        elif hasattr(h, "baseFilename"):
            return h.baseFilename
    return None


def setup_logger(name: str = "vision-ai") -> logging.Logger:
    global _shared_handlers, _error_handlers, _console_handler, _realtime_handler, _retention_cleaned

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not _shared_handlers or not _error_handlers:
        import sys
        target_dirs: List[str] = []

        # 1. Workspace logs (so logs show in VS Code project explorer during dev)
        try:
            workspace_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "logs")
            )
            target_dirs.append(workspace_dir)
        except Exception:
            pass

        # 2. AppData logs (AppData\Local\Vision-Monitor\logs for Windows & production app)
        try:
            from app.core.app_paths import APP_DIR
            appdata_dir = os.path.abspath(os.path.join(str(APP_DIR), "logs"))
            if appdata_dir not in target_dirs:
                target_dirs.append(appdata_dir)
        except Exception:
            pass

        if not target_dirs:
            target_dirs.append(os.path.abspath("logs"))

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(req_id)s%(message)s"
        )
        req_filter = RequestIdFilter()

        # Console handler for clean terminal output
        if _console_handler is None:
            _console_handler = logging.StreamHandler(sys.stdout)
            _console_handler.setLevel(logging.INFO)
            _console_handler.addFilter(req_filter)
            _console_handler.setFormatter(formatter)

        # Realtime log handler for frontend live error streaming
        if _realtime_handler is None:
            _realtime_handler = RealtimeLogHandler()
            _realtime_handler.setLevel(logging.WARNING)

        for base_log_dir in target_dirs:
            try:
                os.makedirs(base_log_dir, exist_ok=True)

                if not _retention_cleaned:
                    clean_old_logs(base_log_dir, retention_days=7)

                # Dynamic hourly session log
                sh = HourlyFileHandler(
                    base_log_dir,
                    max_bytes=_MAX_BYTES,
                    backup_count=_BACKUP_COUNT,
                    encoding="utf-8",
                )
                sh.addFilter(req_filter)
                sh.setFormatter(formatter)
                _shared_handlers.append(sh)

                # Dedicated error-only log (stores all ERROR & CRITICAL from any module)
                error_file_path = os.path.join(base_log_dir, "error.log")
                eh = RotatingFileHandler(
                    error_file_path,
                    maxBytes=10 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
                eh.setLevel(logging.ERROR)
                eh.addFilter(req_filter)
                eh.setFormatter(formatter)
                _error_handlers.append(eh)
            except Exception:
                pass

        _retention_cleaned = True

        # Silence unwanted third-party noise
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("watchfiles").setLevel(logging.WARNING)

    # Attach all handlers to ROOT logger so every module (recording-service, oak-camera, etc.) logs properly
    for h in _shared_handlers:
        if h not in root_logger.handlers:
            root_logger.addHandler(h)
    for h in _error_handlers:
        if h not in root_logger.handlers:
            root_logger.addHandler(h)
    if _console_handler and _console_handler not in root_logger.handlers:
        root_logger.addHandler(_console_handler)
    if _realtime_handler and _realtime_handler not in root_logger.handlers:
        root_logger.addHandler(_realtime_handler)

    # Return configured logger with propagation enabled
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    return logger
