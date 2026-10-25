"""Scrub secrets out of anything headed for logs or traces."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

_REDACTED = "[REDACTED]"

# Common credential shapes. Deliberately conservative: false positives only hide text.
_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI / OpenRouter / Anthropic style
    re.compile(r"\b(?:pk|sk)-lf-[A-Za-z0-9\-]{8,}"),  # Langfuse
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),  # Groq
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),  # Google
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),  # GitHub
    re.compile(r"\btvly-[A-Za-z0-9\-]{16,}"),  # Tavily
    re.compile(r"\bfc-[A-Za-z0-9]{24,}"),  # Firecrawl
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
]

_SECRET_ENV_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


class Redactor:
    """Replaces known secret values and common credential patterns with a placeholder."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        # Longest first so a secret that contains another is replaced whole.
        self._secrets = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)

    @classmethod
    def from_environment(cls) -> Redactor:
        values = [
            v for k, v in os.environ.items() if any(h in k.upper() for h in _SECRET_ENV_HINTS)
        ]
        return cls(values)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, _REDACTED)
        for pattern in _PATTERNS:
            text = pattern.sub(_REDACTED, text)
        return text
