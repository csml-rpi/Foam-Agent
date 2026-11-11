"""Keep editable case inputs separate from files expanded during execution."""

from pathlib import Path
import shutil
import tempfile

from .case_inputs import is_case_artifact, canonical_input_protections
from .review_history import ReviewHistory
from .case_paths import safe_case_relative_from_text, safe_case_path
from utils import resolve_case_file, case_file_key, _is_numeric_name, reject_symbolic_links, is_processor_dir
from .mesh import _missing_mesh_files


def prepare_case_workspace(
    case_dir: str, report_dir: str | None = None, source_dir: str | None = None,
    *, protected_inputs=(),
) -> "CaseWorkspace | None":
    """Prepare full-workflow inputs; direct execution services stay independent."""
    if not case_dir or not str(case_dir).strip() or not Path(case_dir).is_dir():
        return None  # Let the Runner report its existing missing-case diagnostic.
    history = ReviewHistory(case_dir, report_dir)
    try:
        if source_dir:
            workspace = CaseWorkspace(case_dir, source_dir)
            workspace.restore_inputs(protected_inputs=protected_inputs)
            event = "inputs_restored"
        else:
            workspace = CaseWorkspace.create(case_dir, str(history.directory), protected_inputs=protected_inputs)
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
        reject_symbolic_links(Path(case_dir).expanduser())
        reject_symbolic_links(Path(source_dir).expanduser())
        self.case_dir = Path(case_dir).expanduser().resolve()
        self.source_dir = Path(source_dir).expanduser().resolve()
        if not self.case_dir.is_dir() or not self.source_dir.is_dir():
            raise FileNotFoundError("Both the case and its canonical input directory must exist")
        if self.source_dir == self.case_dir or self.case_dir.is_relative_to(self.source_dir):
            raise ValueError("Canonical inputs must not contain the execution directory")
        if self.source_dir.is_relative_to(self.case_dir) and self.source_dir.relative_to(self.case_dir).parts[0] != ".foamagent":
            raise ValueError("Canonical inputs inside the case must be protected by .foamagent metadata")

    @classmethod
    def create(cls, case_dir: str, report_dir: str, *, protected_inputs=()) -> "CaseWorkspace":
        if not str(case_dir).strip() or not str(report_dir).strip():
            raise ValueError("Case and report paths must not be blank")
        reject_symbolic_links(Path(case_dir).expanduser())
        reject_symbolic_links(Path(report_dir).expanduser() / "inputs")
        root = Path(case_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Case directory does not exist: {root}")
        parent = Path(report_dir).expanduser() / "inputs"
        parent.mkdir(parents=True, exist_ok=True)
        source = Path(tempfile.mkdtemp(prefix="workflow-", dir=parent))
        try:
            workspace = cls(str(root), str(source))
            workspace.capture_inputs(protected_inputs=protected_inputs)
            workspace._capture_retry_mesh_sources()
        except Exception:
            shutil.rmtree(source)
            raise
        return workspace

    @staticmethod
    def _ignored(root: Path, protected_inputs=()):
        def ignore(directory: str, names: list[str]) -> list[str]:
            parent = Path(directory)
            return [name for name in names if is_case_artifact(
                (parent / name).relative_to(root), directory=(parent / name).is_dir(),
                protected_inputs=protected_inputs,
            ) or name == "polyMesh" and (
                _is_numeric_name((parent / name).relative_to(root).parts[0])
                or is_processor_dir((parent / name).relative_to(root).parts[0]))]
        return ignore

    def capture_inputs(self, *, protected_inputs=()) -> None:
        """Commit a complete input capture only after its copy succeeds."""
        reject_symbolic_links(self.case_dir)
        reject_symbolic_links(self.source_dir)
        self._check_diagnostic_inputs(protected_inputs)
        with tempfile.TemporaryDirectory(prefix=".capture-", dir=self.source_dir.parent) as temp:
            staged = Path(temp) / "inputs"
            previous = Path(temp) / "previous"
            shutil.copytree(self.case_dir, staged, symlinks=True,
                            ignore=self._ignored(self.case_dir, protected_inputs))
            self.source_dir.rename(previous)
            try:
                staged.rename(self.source_dir)
            except Exception:
                previous.rename(self.source_dir)
                raise

    def _check_diagnostic_inputs(self, protected_inputs) -> None:
        from .execution_logs import check_diagnostic_outputs
        errors = check_diagnostic_outputs(self.case_dir, protected_inputs)
        if errors:
            raise ValueError(' '.join(error['error_content'] for error in errors))

    def retain_startup_inputs(self, protected_inputs) -> None:
        """Retain newly declared assets without replacing canonical edited inputs."""
        self._check_diagnostic_inputs(protected_inputs)
        copies = []
        for name in protected_inputs:
            relative = safe_case_relative_from_text(name)
            target = safe_case_path(self.source_dir, str(relative.parent), relative.name)
            source = resolve_case_file(safe_case_path(self.case_dir, str(relative.parent), relative.name))
            if resolve_case_file(target).is_file():
                continue
            if source.is_file():
                actual = source.relative_to(self.case_dir)
                target = safe_case_path(self.source_dir, str(actual.parent), actual.name)
                copies.append((source, target))
        for source, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def restore_inputs(self, *, protected_inputs=()) -> None:
        """Prepare inputs afresh while keeping report files and run diagnostics.

        The Runner owns normal output/log cleanup. Removing input directories
        first also discards runtime-only meshes and expanded initial fields.
        """
        from .execution_logs import archive_native_logs
        reject_symbolic_links(self.case_dir)
        reject_symbolic_links(self.source_dir)
        retry = safe_case_path(self.source_dir, ".foamagent", "mesh-sources")
        handoff = safe_case_path(self.source_dir, ".foamagent", "mesh-handoff")
        retry_copies = []
        if retry.is_dir() and not handoff.exists():
            for source in retry.rglob("polyMesh"):
                relative = source.relative_to(retry)
                target = safe_case_path(self.case_dir, str(relative.parent), relative.name)
                retry_copies.append((source, target))
        self.retain_startup_inputs(protected_inputs)
        protected_inputs = canonical_input_protections(self.source_dir, protected_inputs)
        archive_native_logs(self.case_dir, protected_inputs=protected_inputs)
        ignored = set(self._ignored(self.case_dir, protected_inputs)(str(self.case_dir), [p.name for p in self.case_dir.iterdir()]))
        for entry in self.case_dir.iterdir():
            if entry.name in ignored:
                continue
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink()
            else:
                shutil.rmtree(entry)
        shutil.copytree(
            self.source_dir, self.case_dir, symlinks=True, dirs_exist_ok=True,
            ignore=self._ignored(self.source_dir, protected_inputs),
        )
        for source, target in retry_copies:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, symlinks=True)

    def _capture_retry_mesh_sources(self) -> None:
        """Retain imported time/partition meshes separately until the first handoff."""
        retry = safe_case_path(self.source_dir, ".foamagent", "mesh-sources")
        copies = []
        for source in self.case_dir.rglob("polyMesh"):
            relative = source.relative_to(self.case_dir)
            if (".foamagent" not in relative.parts and source.is_dir()
                    and (_is_numeric_name(relative.parts[0])
                         or is_processor_dir(relative.parts[0]))):
                target = safe_case_path(retry, str(relative.parent), relative.name)
                copies.append((source, target))
        for source, target in copies:
            shutil.copytree(source, target, symlinks=True)

    def commit_mesh(self, directories, *, declaration_inputs=()) -> None:
        """Stage complete active meshes and their declaration, then commit as a group."""
        relatives = [source.relative_to(self.case_dir) for source in directories.values()]
        declarations = [Path("constant/regionProperties"), Path("constant/regionProperties.gz")]
        for name in declaration_inputs:
            plain = Path(case_file_key(safe_case_relative_from_text(name)))
            declarations.extend((plain, Path(str(plain) + ".gz")))
        declarations = list(dict.fromkeys(path for path in declarations
                                         if not any(mesh in path.parents for mesh in relatives)))
        relatives.extend([*declarations, Path(".foamagent/mesh-handoff")])
        for relative in relatives:
            safe_case_path(self.source_dir, str(relative.parent), relative.name)
        with tempfile.TemporaryDirectory(prefix=".mesh-handoff-", dir=self.source_dir.parent) as temp:
            staging, backup = Path(temp) / "new", Path(temp) / "old"
            relatives = []
            for source in directories.values():
                relative = source.relative_to(self.case_dir)
                target = staging / relative
                shutil.copytree(source, target, symlinks=True)
                missing = _missing_mesh_files(target)
                if missing:
                    raise ValueError(f"Incomplete mesh handoff {relative}: missing {missing}")
                relatives.append(relative)
            for relative in declarations:
                source = self.case_dir / relative
                if source.is_file():
                    (staging / relative).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, staging / relative)
                relatives.append(relative)
            marker = Path(".foamagent/mesh-handoff")
            (staging / marker).parent.mkdir(parents=True, exist_ok=True)
            (staging / marker).write_text("complete\n")
            relatives.append(marker)
            moved, installed = [], []
            try:
                for relative in relatives:
                    target = self.source_dir / relative
                    previous = backup / relative
                    if target.exists() or target.is_symlink():
                        previous.parent.mkdir(parents=True, exist_ok=True)
                        target.rename(previous)
                        moved.append(relative)
                    replacement = staging / relative
                    if replacement.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        replacement.rename(target)
                        installed.append(relative)
            except Exception:
                for relative in reversed(installed):
                    target = self.source_dir / relative
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                for relative in reversed(moved):
                    (backup / relative).rename(self.source_dir / relative)
                raise

    def update_inputs(self, written_files) -> None:
        """Commit explicit input edits without capturing execution-expanded files."""
        retry = safe_case_path(self.source_dir, ".foamagent", "mesh-sources")
        handoff = safe_case_path(self.source_dir, ".foamagent", "mesh-handoff")
        copies = []
        for name in written_files:
            relative = safe_case_relative_from_text(name)
            source = safe_case_path(self.case_dir, str(relative.parent), relative.name)
            targets = [safe_case_path(self.source_dir, str(relative.parent), relative.name)]
            if not handoff.exists() and any(
                    parent.name == "polyMesh" and (retry / parent).is_dir()
                    for parent in relative.parents):
                targets.append(safe_case_path(retry, str(relative.parent), relative.name))
            for target in targets:
                previous = resolve_case_file(target)
                copies.append((source, target, previous))
        for source, target, previous in copies:
            if previous != target and previous.is_file():
                previous.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
