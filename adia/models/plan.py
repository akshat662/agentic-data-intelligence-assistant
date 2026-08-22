"""A single node in the Planner's step DAG."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanStep(BaseModel):
    """One step of a Planner-produced DAG.

    The Planner emits *intent*, not tool arguments — `tool_family` names which tool should
    handle this step, while the concrete arguments are filled in later, per-step, by ArgGen.
    Keeping the two separate keeps "which tool" and "which arguments" independently measurable.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        ..., description="Unique identifier of this step within its plan, e.g. 'step_1'."
    )
    intent: str = Field(
        ..., description="Natural-language description of what this step should do."
    )
    tool_family: str = Field(
        ..., description="Name of the tool (or tool family) expected for this step."
    )
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of steps that must complete before this one."
    )
    expected_output: str = Field(
        ..., description="Description of the expected shape/content of this step's result."
    )
    success_criteria: str = Field(
        ..., description="What must be true of the result for this step to count as successful."
    )

    @model_validator(mode="after")
    def _no_self_dependency(self) -> Self:
        if self.id in self.depends_on:
            raise ValueError(f"PlanStep '{self.id}' cannot depend on itself.")
        return self
