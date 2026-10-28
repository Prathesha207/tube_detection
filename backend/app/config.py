import sys
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

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def resolve_database_url(cls, v: str) -> str:
        # 1. If a relative SQLite path is specified (e.g. sqlite:///./vision_ai.db),
        # anchor it directly to the backend directory.
        if v.startswith("sqlite:///./"):
            rel_name = v[len("sqlite:///./") :]
            abs_path = (_ROOT / rel_name).resolve().as_posix()
            return f"sqlite:///{abs_path}"

        # 2. If an absolute SQLite path was configured on another OS or drive that doesn't exist,
        # fallback safely to the backend/vision_ai.db in this environment.
        if v.startswith("sqlite:///"):
            clean_path = v[len("sqlite:///") :]
            # Strip extra leading slash on unix if present
            test_path = Path(clean_path if sys.platform == "win32" else "/" + clean_path.lstrip("/"))
            if not test_path.parent.exists():
                fallback = (_ROOT / "vision_ai.db").resolve().as_posix()
                return f"sqlite:///{fallback}"

        return v

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
