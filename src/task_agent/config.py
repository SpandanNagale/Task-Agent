"""Process-wide settings, read from the environment (and a local .env file).

Secrets live only here and in the environment. They are never written into prompts, logs
or traces; see ``observability.redaction``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TASK_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # Model gateway: a YAML file mapping roles (planner, worker, router, ...) to fallback chains.
    models_config: Path = Path("config/models.yaml")
    # Approval rules per tool and side-effect level.
    policy_file: Path = Path("policies/default.yaml")

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

    # --- Tools. A tool is registered only when its key is present. -----------------------
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")
    firecrawl_api_key: SecretStr | None = Field(default=None, alias="FIRECRAWL_API_KEY")
    jina_api_key: SecretStr | None = Field(default=None, alias="JINA_API_KEY")  # optional
    e2b_api_key: SecretStr | None = Field(default=None, alias="E2B_API_KEY")
    composio_api_key: SecretStr | None = Field(default=None, alias="COMPOSIO_API_KEY")

    # Sandbox (E2B). Internet is off by default; code runs on what the image ships with.
    sandbox_internet: bool = False
    sandbox_lifetime_s: int = Field(default=600, ge=60)  # killed after this even if leaked
    sandbox_exec_timeout_s: float = Field(default=120.0, gt=0)

    # Composio: whose connected accounts tools act as, and which tools to load.
    composio_user_id: str = "default"
    composio_tools: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "GMAIL_FETCH_EMAILS",
            "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
            "GMAIL_CREATE_EMAIL_DRAFT",
            "GMAIL_SEND_EMAIL",
            "GMAIL_REPLY_TO_THREAD",
        ]
    )

    # Largest tool output the model sees, in characters. Longer outputs are truncated.
    tool_output_max_chars: int = Field(default=12_000, ge=500)

    @field_validator("composio_tools", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        # TASK_AGENT_COMPOSIO_TOOLS="GMAIL_SEND_EMAIL,GMAIL_FETCH_EMAILS"
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
