"""Data contracts shared by the existing-case import workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


class CaseImportError(ValueError):
    """Raised when a case cannot safely enter import mode."""


@dataclass(frozen=True)
class ExecutionStep:
    """One validated OpenFOAM command in the controlled execution plan."""

    command: str
    args: tuple[str, ...] = ()
    parallel: bool = False
    origin: str = "user_allrun"

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": list(self.args),
            "parallel": self.parallel,
            "origin": self.origin,
        }


@dataclass
class CaseManifest:
    """Immutable description of a selected source case and its execution plan."""

    source: str
    case_root: str
    output_root: str
    platform: str
    version: Optional[str]
    application: str
    allrun_provided: bool
    mesh_state: str
    execution_plan: list[ExecutionStep] = field(default_factory=list)
    detected_libraries: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    original_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return not self.blocking_issues and self.platform in {
            "foundation-v10",
            "foundation-v10-compatible",
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["execution_plan"] = [step.to_dict() for step in self.execution_plan]
        result["supported"] = self.supported
        return result


@dataclass
class ImportRunResult:
    """Terminal result of importing and optionally executing a case."""

    status: str
    original_dir: str
    work_dir: str
    report_dir: str
    manifest: CaseManifest
    attempts: list[dict[str, Any]]
    errors: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "original_dir": self.original_dir,
            "work_dir": self.work_dir,
            "report_dir": self.report_dir,
            "manifest": self.manifest.to_dict(),
            "attempts": self.attempts,
            "errors": self.errors,
        }
