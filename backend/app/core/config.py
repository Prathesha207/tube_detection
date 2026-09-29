import os

class Settings:
    TRAINING_STORAGE_PATH = os.getenv("TRAINING_STORAGE_PATH", "storage/training")

settings = Settings()