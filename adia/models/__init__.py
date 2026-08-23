"""Pydantic data contracts for ADIA.

This package defines every cross-layer contract (tool results, evidence, plan steps, shared
run state, dataset catalogs) with zero dependency on `adia.tools`, `adia.graph`, `adia.agents`,
or any other implementation layer. Implementation modules import from here; this package never
imports from them.
"""

from adia.models.catalog import ColumnProfile, DatasetCatalog, SemanticType
from adia.models.dataset import DatasetConfig
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.evidence import Evidence
from adia.models.plan import PlanStep
from adia.models.provenance import Provenance
from adia.models.state import (
    AgentState,
    Budget,
    Critique,
    CritiqueSeverity,
    FeasibilityResult,
    FeasibilityVerdict,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from adia.models.tool_result import ToolResult

__all__ = [
    "AgentState",
    "Budget",
    "ColumnProfile",
    "Critique",
    "CritiqueSeverity",
    "DatasetCatalog",
    "DatasetConfig",
    "Evidence",
    "FeasibilityResult",
    "FeasibilityVerdict",
    "PlanStep",
    "Provenance",
    "SemanticType",
    "ToolError",
    "ToolErrorKind",
    "ToolResult",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
]
