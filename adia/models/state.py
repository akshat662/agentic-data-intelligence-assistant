"""Shared orchestration state contract.

`AgentState` is the single source of truth intended to be threaded through the future
LangGraph run (Phase 2). It is defined here, independent of any graph or agent implementation,
so the contract can be fixed and unit-tested before any orchestration code exists.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from adia.models.catalog import DatasetCatalog
from adia.models.errors import ToolError
from adia.models.evidence import Evidence
from adia.models.plan import PlanStep


class FeasibilityVerdict(StrEnum):
    """Outcome of the Feasibility agent's check against the dataset catalog."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NEEDS_CLARIFICATION = "needs_clarification"


class FeasibilityResult(BaseModel):
    """Output contract of the Feasibility agent."""

    model_config = ConfigDict(frozen=True)

    verdict: FeasibilityVerdict
    reason: str
    missing_columns: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)


class ValidationSeverity(StrEnum):
    """Severity of a single static-validator finding."""

    WARN = "warn"
    FAIL = "fail"


class ValidationIssue(BaseModel):
    """A single finding raised by the static validator (numeral tracing, citation checks, ...)."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(
        ..., description="Machine-readable check identifier, e.g. 'untraced_numeral'."
    )
    message: str
    severity: ValidationSeverity


class ValidationResult(BaseModel):
    """Aggregate result of running the static validator over a rendered answer."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class CritiqueSeverity(StrEnum):
    """Outcome of the Critic agent's semantic-overreach review."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class Critique(BaseModel):
    """Output contract of the Critic agent.

    The Critic only judges semantic overreach (causal claims, confounders,
    over-generalization) — arithmetic and citation correctness are the static validator's job.
    """

    model_config = ConfigDict(frozen=True)

    severity: CritiqueSeverity
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None


class Budget(BaseModel):
    """Hard caps on a single run, enforced by the deterministic budget-guard node."""

    max_llm_calls: int = Field(..., gt=0)
    max_tool_calls: int = Field(..., gt=0)
    used_llm_calls: int = Field(default=0, ge=0)
    used_tool_calls: int = Field(default=0, ge=0)

    def exhausted(self) -> bool:
        """Whether either cap has been reached."""
        return (
            self.used_llm_calls >= self.max_llm_calls
            or self.used_tool_calls >= self.max_tool_calls
        )


class AgentState(BaseModel):
    """Single shared state object threaded through the (future) LangGraph run.

    Defined independently of any graph/agent code so its shape — including the three bounded
    loop counters and the append-only error log — is fixed and testable before Phase 2 exists.
    `evidence` must be merged, never overwritten, when parallel plan branches write to it; a
    plain assignment silently drops one branch's results.
    """

    model_config = ConfigDict(validate_assignment=True)

    run_id: str
    question: str
    dataset_id: str
    catalog: DatasetCatalog | None = None

    feasibility: FeasibilityResult | None = None
    plan: list[PlanStep] = Field(default_factory=list)
    evidence: dict[str, Evidence] = Field(
        default_factory=dict,
        description="Keyed by evidence ID. Must be merged, never overwritten, across branches.",
    )
    errors: list[ToolError] = Field(
        default_factory=list, description="Append-only log of tool errors encountered this run."
    )

    draft_answer: str | None = Field(
        default=None, description="Synthesizer output containing placeholders, pre-render."
    )
    rendered_answer: str | None = Field(
        default=None, description="Draft answer with placeholders substituted from evidence."
    )
    validation: ValidationResult | None = None
    critique: Critique | None = None
    final_answer: str | None = None
    refusal: FeasibilityResult | None = Field(
        default=None,
        description="Set when the run terminates via refusal/clarification, not an answer.",
    )

    repair_attempts: int = Field(
        default=0, ge=0, description="Tool repair loop counter, capped at 2 per step."
    )
    replan_count: int = Field(
        default=0, ge=0, description="Replan loop counter, capped at 1 per run."
    )
    validation_rounds: int = Field(
        default=0, ge=0, description="Validation repair loop counter, capped at 2 per run."
    )

    budget: Budget
