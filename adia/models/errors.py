"""Typed error contract returned by tools instead of raising exceptions."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolErrorKind(StrEnum):
    """Closed set of ways a tool call can fail.

    The router branches on `kind`, not on exception messages, so this set must stay closed
    and exhaustive enough for the router to make retry/replan/escalate decisions from it alone.
    """

    VALIDATION = "validation"
    GUARD_REJECTED = "guard_rejected"
    NOT_FOUND = "not_found"
    INSUFFICIENT_DATA = "insufficient_data"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ToolError(BaseModel):
    """A typed, structured failure returned by a tool.

    Tools must never raise into the orchestration layer; they catch their own failures and
    return a `ToolError` inside a `ToolResult` so the router can branch on `kind` instead of
    parsing an exception message.
    """

    model_config = ConfigDict(frozen=True)

    kind: ToolErrorKind
    message: str = Field(..., description="Human-readable description of what went wrong.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured context for debugging (e.g. the offending column name).",
    )
    retryable: bool = Field(
        default=False, description="Whether retrying the same or a repaired call could succeed."
    )
