from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent
_ENV_PATH = _ROOT / ".env"


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./vision_ai.db"
    APP_NAME: str = "Vision AI Command Center"
    DEBUG: bool = True

    from pydantic import field_validator

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "debug", "dev", "development")
        return bool(v)

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
