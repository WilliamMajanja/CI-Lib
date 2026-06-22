"""Application configuration via environment variables."""

from __future__ import annotations

import os


class Settings:
    """Application-level configuration loaded from environment variables.

    Environment variables:
        CI_HOST (str):          Host to bind to (default "0.0.0.0")
        CI_PORT (int):          Port to listen on (default 8000)
        CI_CORS_ORIGINS (str):  Comma-separated CORS origins (default "*")
        CI_LOG_LEVEL (str):     Logging level (default "info")
        CI_DEBUG (bool):        Enable debug/reload mode (default "false")
    """

    app_name: str = "CI-Lib API"
    version: str = "1.0.0"
    debug: bool = os.getenv("CI_DEBUG", "false").lower() == "true"
    host: str = os.getenv("CI_HOST", "0.0.0.0")
    port: int = int(os.getenv("CI_PORT", "8000"))
    allowed_origins: list[str] = os.getenv("CI_CORS_ORIGINS", "*").split(",")
    log_level: str = os.getenv("CI_LOG_LEVEL", "info").upper()


settings = Settings()
