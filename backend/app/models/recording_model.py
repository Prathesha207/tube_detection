from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from datetime import datetime
from app.core.database import Base


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)

    camera_id = Column(Integer, ForeignKey("cameras.id"))

    file_path = Column(String(255))

    start_time = Column(DateTime, default=datetime.utcnow)

    end_time = Column(DateTime, nullable=True)