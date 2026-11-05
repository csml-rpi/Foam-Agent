from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field

from openfoam_target import ESI_V2006


class RunOut(BaseModel):
    job_id: Optional[str]
    status: str


class CaseImportError(ValueError):
    """Raised when an existing case cannot be materialised for the workflow."""


@dataclass
class CaseManifest:
    """Materialised existing case and the context discovered during import."""

    source: str
    case_root: str
    output_root: str
    platform: str
    version: Optional[str]
    application: Optional[str]
    allrun_provided: bool
    candidate_cases: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    target_mismatch: bool = False

    @property
    def supported(self) -> bool:
        return self.platform in {
            "foundation-v10",
            ESI_V2006,
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["supported"] = self.supported
        return result


class PlannedFileChange(BaseModel):
    """One file-level change requested by the planner or reviewer."""

    file: str = Field(description="Relative file path, e.g. system/fvSchemes or 0/U")
    changes: str = Field(description="Semicolon-separated concrete changes for this file")
    stage: Literal["mesh", "solver"]


class DeclaredStageInput(BaseModel):
    path: str = Field(description="Case-relative input file required before stage execution")
    evidence: str = Field(min_length=1, description="Why this operation needs the file")


class StageOperation(BaseModel):
    stage: Literal["mesh", "solver"] = Field(description="mesh builds or inspects geometry; solver prepares physical fields, sets up solver execution or solves")
    purpose: str = Field(min_length=1, description="Intended effect and why this operation belongs to its declared stage; do not classify by command or file name")
    command: str = Field(min_length=1, description="For script: literal executable Bash, with exact arguments and retained shell setup/content, never prose or action labels. For backend: describe the selected native backend operation.")
    executor: Literal["script", "backend"] = Field(description="script for all solver and standard mesh commands; backend only for native Gmsh generation or selected .msh conversion/checking")
    required_inputs: List[DeclaredStageInput]
    generated_inputs: List[str] = Field(description="Inputs produced by earlier operations, naming their producer; not preflight requirements")


class ExistingCasePlan(BaseModel):
    """Decisions for the existing-case graph routes."""

    status: Literal["ready", "failed"] = "ready"
    requires_meshing: bool = False
    requires_input_writer: bool = False
    target_files: List[PlannedFileChange] = Field(default_factory=list)
    reason: str = ""
    operations: List[StageOperation]
