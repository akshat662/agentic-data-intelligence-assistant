"""LLM configuration: environment-variable based, no secrets in code.

Every agent that calls an LLM reads its configuration through `load_llm_settings` rather
than touching `os.environ` directly, so there is exactly one place that knows the
environment variable names and one place that validates them. No API key is ever
hardcoded, logged, or given a default — its absence is a typed error, not a silent `None`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

#: Used when `OPENAI_MODEL` is unset. A small, inexpensive model is the right default for a
#: structured-output classification task like feasibility assessment — this is not the model
#: choice for a future Synthesizer, which may reasonably want something larger.
DEFAULT_MODEL = "gpt-4o-mini"


class MissingApiKeyError(RuntimeError):
    """Raised when `OPENAI_API_KEY` is not set and an LLM call is about to be made."""


class LLMSettings(BaseModel):
    """Resolved LLM configuration for one run. Never constructed with a placeholder key."""

    model_config = ConfigDict(frozen=True)

    api_key: str = Field(..., min_length=1, repr=False, description="OpenAI API key.")
    model: str = Field(..., min_length=1, description="Chat model name, e.g. 'gpt-4o-mini'.")


def load_llm_settings(*, dotenv_path: str | Path | None = None) -> LLMSettings:
    """Load LLM settings from the environment (optionally via a `.env` file first).

    Args:
        dotenv_path: Path to a `.env` file to load before reading `os.environ`. Defaults to
            `python-dotenv`'s own discovery (nearest `.env` walking up from the cwd).
            Variables already set in the environment are never overridden by the file.

    Returns:
        The resolved settings.

    Raises:
        MissingApiKeyError: If `OPENAI_API_KEY` is not set anywhere.
    """
    load_dotenv(dotenv_path=dotenv_path, override=False)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise MissingApiKeyError(
            "OPENAI_API_KEY is not set. Set it in the environment or in a .env file "
            "(see .env.example)."
        )

    model = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
    return LLMSettings(api_key=api_key, model=model)
