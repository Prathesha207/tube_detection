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

_shared_handlers: List[RotatingFileHandler] = []
_error_handlers: List[RotatingFileHandler] = []
_retention_cleaned = False


class RequestIdFilter(logging.Filter):
    """Injects current_request_id into log records so all logs within a request share the same correlation ID."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "req_id"):
            req_id = current_request_id.get()
            record.req_id = f"[{req_id}] " if req_id else ""
        return True


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
    if _shared_handlers:
        return _shared_handlers[0].baseFilename
    return None


def setup_logger(name: str = "vision-ai") -> logging.Logger:
    global _shared_handlers, _error_handlers, _retention_cleaned

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not _shared_handlers or not _error_handlers:
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

        date_folder = datetime.now().strftime("%Y-%m-%d")
        hour_folder = datetime.now().strftime("%H")

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(req_id)s%(message)s"
        )
        req_filter = RequestIdFilter()

        for base_log_dir in target_dirs:
            try:
                os.makedirs(base_log_dir, exist_ok=True)

                if not _retention_cleaned:
                    clean_old_logs(base_log_dir, retention_days=7)

                full_path = os.path.join(base_log_dir, date_folder, hour_folder)
                os.makedirs(full_path, exist_ok=True)

                # Hourly session log
                log_file_path = os.path.join(full_path, f"session_{hour_folder}.log")
                sh = RotatingFileHandler(
                    log_file_path,
                    maxBytes=_MAX_BYTES,
                    backupCount=_BACKUP_COUNT,
                    encoding="utf-8",
                )
                sh.addFilter(req_filter)
                sh.setFormatter(formatter)
                _shared_handlers.append(sh)

                # Dedicated error-only log
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

        # Silence unwanted uvicorn startup/shutdown lifecycle messages & httpx noise
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    for h in _shared_handlers:
        if h not in logger.handlers:
            logger.addHandler(h)
    for h in _error_handlers:
        if h not in logger.handlers:
            logger.addHandler(h)

    return logger
