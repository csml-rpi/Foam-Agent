"""Keep editable case inputs separate from files expanded during execution."""

from pathlib import Path
import shutil
import tempfile

from .case_inputs import is_case_artifact
from .review_history import ReviewHistory


def prepare_case_workspace(
    case_dir: str, report_dir: str | None = None, source_dir: str | None = None,
) -> "CaseWorkspace | None":
    """Prepare full-workflow inputs; direct execution services stay independent."""
    if not case_dir or not str(case_dir).strip() or not Path(case_dir).is_dir():
        return None  # Let the Runner report its existing missing-case diagnostic.
    history = ReviewHistory(case_dir, report_dir)
    try:
        if source_dir:
            workspace = CaseWorkspace(case_dir, source_dir)
            workspace.restore_inputs()
            event = "inputs_restored"
        else:
            workspace = CaseWorkspace.create(case_dir, str(history.directory))
            event = "input_source_created"
    except (OSError, ValueError) as exc:
        history.append("input_snapshot_failed", source_dir=source_dir, error=str(exc))
        raise
    history.append(event, source_dir=str(workspace.source_dir))
    return workspace


class CaseWorkspace:
    """A workflow's canonical inputs and its existing execution directory.

    Snapshots include complete files, private assets and symbolic links, without
    applying the model context limit. Copying never follows symbolic links.
    """

    def __init__(self, case_dir: str, source_dir: str):
        if not str(case_dir).strip() or not str(source_dir).strip():
            raise ValueError("Case and canonical input paths must not be blank")
        self.case_dir = Path(case_dir).resolve()
        self.source_dir = Path(source_dir).resolve()
        if not self.case_dir.is_dir() or not self.source_dir.is_dir():
            raise FileNotFoundError("Both the case and its canonical input directory must exist")
        if self.source_dir == self.case_dir or self.case_dir.is_relative_to(self.source_dir):
            raise ValueError("Canonical inputs must not contain the execution directory")
        if self.source_dir.is_relative_to(self.case_dir) and self.source_dir.relative_to(self.case_dir).parts[0] != ".foamagent":
            raise ValueError("Canonical inputs inside the case must be protected by .foamagent metadata")

    @classmethod
    def create(cls, case_dir: str, report_dir: str) -> "CaseWorkspace":
        if not str(case_dir).strip() or not str(report_dir).strip():
            raise ValueError("Case and report paths must not be blank")
        root = Path(case_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Case directory does not exist: {root}")
        parent = Path(report_dir) / "inputs"
        parent.mkdir(parents=True, exist_ok=True)
        source = Path(tempfile.mkdtemp(prefix="workflow-", dir=parent))
        try:
            workspace = cls(str(root), str(source))
            workspace.capture_inputs()
        except Exception:
            shutil.rmtree(source)
            raise
        return workspace

    @staticmethod
    def _ignored(root: Path):
        def ignore(directory: str, names: list[str]) -> list[str]:
            parent = Path(directory)
            return [name for name in names if is_case_artifact(
                (parent / name).relative_to(root), directory=(parent / name).is_dir()
            )]
        return ignore

    def capture_inputs(self) -> None:
        """Commit a complete input capture only after its copy succeeds."""
        with tempfile.TemporaryDirectory(prefix=".capture-", dir=self.source_dir.parent) as temp:
            staged = Path(temp) / "inputs"
            previous = Path(temp) / "previous"
            shutil.copytree(self.case_dir, staged, symlinks=True, ignore=self._ignored(self.case_dir))
            self.source_dir.rename(previous)
            try:
                staged.rename(self.source_dir)
            except Exception:
                previous.rename(self.source_dir)
                raise

    def restore_inputs(self) -> None:
        """Prepare inputs afresh while keeping report files and run diagnostics.

        The Runner owns normal output/log cleanup. Removing input directories
        first also discards runtime-only meshes and expanded initial fields.
        """
        ignored = set(self._ignored(self.case_dir)(str(self.case_dir), [p.name for p in self.case_dir.iterdir()]))
        for entry in self.case_dir.iterdir():
            if entry.name in ignored:
                continue
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink()
            else:
                shutil.rmtree(entry)
        shutil.copytree(
            self.source_dir, self.case_dir, symlinks=True, dirs_exist_ok=True,
            ignore=self._ignored(self.source_dir),
        )
