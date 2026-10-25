"""Process-wide settings, read from the environment (and a local .env file).

Secrets live only here and in the environment. They are never written into prompts, logs
or traces; see ``observability.redaction``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TASK_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # Model gateway: a YAML file mapping roles (planner, worker, router, ...) to fallback chains.
    models_config: Path = Path("config/models.yaml")

    # Tracing. Langfuse is used as an OpenTelemetry backend, so any OTLP collector works too.
    tracing: Literal["auto", "langfuse", "otlp", "console", "off"] = "auto"
    otlp_endpoint: str | None = None  # used when tracing == "otlp"
    service_name: str = "task-agent"

    # Langfuse keeps its own conventional, unprefixed env names.
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")
    langfuse_public_key: SecretStr | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")

    # Whether prompts and completions are attached to traces. Turn off for sensitive data.
    trace_content: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
