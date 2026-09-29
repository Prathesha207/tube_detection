from collections import deque
from datetime import datetime
from threading import Lock


class RealtimeLogService:
    def __init__(self):
        self.logs = {
            "stream": deque(maxlen=100),
            "record": deque(maxlen=100),
            "inference": deque(maxlen=100),
            "camera": deque(maxlen=100),
            "system": deque(maxlen=100),
        }

        self.lock = Lock()

    def add_log(
        self,
        category: str,
        tag: str,
        message: str,
        status: str = "info"
    ):
        category = category.lower()

        if category not in self.logs:
            category = "system"

        log_data = {
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "tag": tag,
            "message": message,
            "status": status.lower()
        }

        with self.lock:
            self.logs[category].appendleft(log_data)

    def get_logs(self, category: str):
        category = category.lower()

        if category not in self.logs:
            category = "system"

        return list(self.logs[category])


realtime_log_service = RealtimeLogService()