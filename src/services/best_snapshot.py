"""Keep an owned checkpoint of the furthest simulation attempt."""

from copy import deepcopy
import math
from pathlib import Path
import re
import shutil
import tempfile

from utils import read_case_foamfiles, scan_case_directory


_PRESERVED = {'.foamagent', '.foamagent-output', 'workflow.log', 'review.log'}


def _metadata_dir(case: Path) -> Path:
    metadata = case / '.foamagent'
    if metadata.is_symlink() or (metadata.exists() and not metadata.is_dir()):
        raise ValueError(f'Invalid checkpoint directory: {metadata}')
    metadata.mkdir(exist_ok=True)
    return metadata


def _owned_snapshot(case: Path, value) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_symlink() or path.parent.resolve() != (case / '.foamagent').resolve():
        raise ValueError(f'Checkpoint is outside the case metadata directory: {path}')
    if not path.name.startswith('best-'):
        raise ValueError(f'Invalid checkpoint name: {path}')
    return path


def _dataset_relative(state, case):
    value = getattr(state.get('config'), 'dataset_log_path', '')
    if value:
        try:
            return Path(value).expanduser().resolve().relative_to(case)
        except ValueError:
            pass
    return None


def compute_progress_score(case_dir: str, error_logs: list) -> float:
    """Use the highest finite logged timestep; this is not a physics score."""
    best = None
    for path in Path(case_dir).glob('log.*'):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with path.open(errors='replace') as handle:
                for line in handle:
                    match = re.match(r'^Time = ([\d.eE+\-]+)', line)
                    if match:
                        try:
                            value = float(match[1])
                        except ValueError:
                            continue
                        if math.isfinite(value) and (best is None or value > best):
                            best = value
        except OSError:
            continue
    return best if best is not None else float(-len(error_logs))


def save_best_snapshot(state):
    case = Path(state['case_dir']).resolve()
    if not case.is_dir():
        return {}
    errors = state.get('error_logs') or []
    score = compute_progress_score(str(case), errors)
    previous = state.get('best_progress_score')
    if previous is not None and score <= previous:
        return {}
    metadata = _metadata_dir(case)
    old_snapshot = _owned_snapshot(case, state.get('best_case_snapshot_dir'))
    dataset = _dataset_relative(state, case)

    def ignore(directory, names):
        relative = Path(directory).relative_to(case)
        ignored = set(_PRESERVED if relative == Path('.') else ())
        if dataset is not None and dataset.parent == relative:
            ignored.add(dataset.name)
        return ignored.intersection(names)

    snapshot = Path(tempfile.mkdtemp(prefix='best-', dir=metadata))
    try:
        shutil.copytree(case, snapshot, symlinks=True, dirs_exist_ok=True, ignore=ignore)
    except BaseException:
        shutil.rmtree(snapshot)
        raise
    if old_snapshot is not None and old_snapshot.exists():
        shutil.rmtree(old_snapshot)
    print(f'<snapshot>Saved progress={score:.4g} to {snapshot}</snapshot>')
    return {
        'best_case_snapshot_dir': str(snapshot),
        'best_progress_score': score,
        'best_snapshot_errors': deepcopy(errors),
    }


def restore_best_snapshot(state):
    case = Path(state['case_dir']).resolve()
    snapshot = _owned_snapshot(case, state.get('best_case_snapshot_dir'))
    if snapshot is None or not snapshot.is_dir():
        print('<restore_best>No checkpoint to restore.</restore_best>')
        return {}
    metadata = _metadata_dir(case)
    staging = Path(tempfile.mkdtemp(prefix='restore-', dir=metadata))
    backup = Path(tempfile.mkdtemp(prefix='previous-', dir=metadata))
    moved_old = []
    installed = []
    keep_backup = False
    try:
        shutil.copytree(snapshot, staging, symlinks=True, dirs_exist_ok=True)
        dataset = _dataset_relative(state, case)
        if dataset is not None and dataset.parts[0] not in _PRESERVED and (case / dataset).is_file():
            destination = staging / dataset
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(case / dataset, destination)
        try:
            # Keep the case root and active log handles in place. All files are
            # staged before the current attempt is touched; a failed swap rolls back.
            for child in case.iterdir():
                if child.name not in _PRESERVED:
                    child.rename(backup / child.name)
                    moved_old.append(child.name)
            for child in staging.iterdir():
                child.rename(case / child.name)
                installed.append(child.name)
        except BaseException:
            try:
                for name in installed:
                    (case / name).rename(staging / name)
                for name in moved_old:
                    (backup / name).rename(case / name)
            except BaseException:
                keep_backup = True
                raise
            raise
    finally:
        shutil.rmtree(staging)
        # A failed rollback must leave its recovery copy available.
        if not keep_backup:
            shutil.rmtree(backup)
    directory = scan_case_directory(str(case))
    foamfiles = read_case_foamfiles(str(case), directory)
    solver = state.get('case_solver', '')
    control = case / 'system' / 'controlDict'
    if control.is_file():
        text = re.sub(r'/\*.*?\*/|//[^\n]*', '', control.read_text(errors='replace'), flags=re.DOTALL)
        match = re.search(r'\bapplication\s+([^;\s]+)\s*;', text)
        if match:
            solver = match[1]
    print(f'<restore_best>Restored progress={state.get("best_progress_score")} from {snapshot}</restore_best>')
    return {
        'dir_structure': directory,
        'foamfiles': foamfiles,
        'case_solver': solver,
        'error_logs': deepcopy(state.get('best_snapshot_errors') or state.get('error_logs') or []),
        'restored_best_snapshot': True,
        'workflow_status': 'failed',
        'termination_reason': state.get('termination_reason') or 'max_review_loop_reached',
        'updated_files': [],
        'file_diffs': [],
    }
