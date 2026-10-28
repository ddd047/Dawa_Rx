"""Application configuration loaded from environment variables."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings are sourced from the .env file or environment."""

    # TrOCR model (HuggingFace — runs locally, no API key needed)
    trocr_model: str = "microsoft/trocr-base-handwritten"
    trocr_device: str = "cpu"  # set to "cuda" if GPU is available

    # API URL (used by frontend)
    next_public_api_url: str = "http://localhost:3000"

    # Matching thresholds
    brand_match_cutoff: int = 80
    generic_match_cutoff: int = 75

    # Upload limits
    max_upload_bytes: int = 8 * 1024 * 1024  # 8 MB
    allowed_mime_types: list[str] = ["image/jpeg", "image/png", "image/webp"]

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
