from pathlib import Path
from typing import ClassVar, Optional

from pydantic_settings import BaseSettings


class BaseAppSettings(BaseSettings):
    """Base settings class with common configuration."""

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class AuthSettings(BaseAppSettings):
    """Authentication and API keys settings."""

    HUGGINGFACE_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None


class Settings(BaseAppSettings):
    """Main settings class that combines all specialized settings."""

    auth: AuthSettings = AuthSettings()


# Create global settings instance
settings = Settings()
