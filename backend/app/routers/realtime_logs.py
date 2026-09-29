from fastapi import APIRouter
from app.services.realtime_log_service import realtime_log_service
import logging

logger = logging.getLogger("realtime-logs")
router = APIRouter(prefix="/logs", tags=["Realtime Logs"])


@router.get("")
@router.get("/")
@router.get("/{category}")
def get_logs(category: str = "system"):
    try:
        return {
            "status": "success",
            "category": category,
            "logs": realtime_log_service.get_logs(category)
        }
    except Exception as e:
        logger.error(f"Failed to fetch logs for category {category}: {e}", exc_info=True)
        return {"status": "error", "category": category, "logs": []}