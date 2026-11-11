"""Offline tests must supply model and scheduler doubles explicitly."""

import importlib
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def offline_services(monkeypatch):
    import utils

    def forbidden(*args, **kwargs):
        pytest.fail("Real model/network access is forbidden in offline tests")

    monkeypatch.setattr(utils.LLMService, "invoke", forbidden)
    monkeypatch.setattr(utils, "get_embedding_model", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    original_popen = subprocess.Popen

    class OfflinePopen(original_popen):
        def __init__(self, args, *positional, **kwargs):
            command = args if isinstance(args, str) else " ".join(map(str, args))
            import re
            if re.search(r"\b(sbatch|squeue|scancel|sacct)\b", command):
                pytest.fail("Real Slurm access is forbidden in offline tests")
            super().__init__(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", OfflinePopen)


@pytest.fixture
def modules(monkeypatch, tmp_path):
    """Load workflow services with model and native-tool calls isolated."""
    utils = importlib.import_module("utils")
    names = ("main", "nodes.solver_planner_node", "services.meshing", "services.mesh",
             "services.input_writer", "services.run_local", "services.plan", "services.preparation")
    result = SimpleNamespace(**{name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in names})
    result.read_patch_bounds = result.meshing.read_patch_bounds
    monkeypatch.setattr(result.meshing, "read_patch_bounds", lambda _: {"inlet": [0, 0, 0, 1, 0, 1]})
    from services import solver_execution
    from services.case_import import _application
    monkeypatch.setattr(solver_execution, "run_openfoam_utility", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0,
                            'application ' + (_application(Path(kw["working_dir"])) or '') + ';', ""))
    monkeypatch.setattr(result.preparation, "setup_logging", lambda *_: None)
    monkeypatch.setattr(result.meshing, "retrieve_mesh_reference", lambda state, input_context: {
        "case": None, "content": "", "script": "", "reason": "No tutorial required by this test"})
    from services import static_review
    real_review = static_review.review_case_inputs
    def passing_review(**kwargs):
        kwargs['llm_service'] = SimpleNamespace(invoke=lambda *a, **kw:
            static_review.StaticReviewResult(summary="Fixture inputs reviewed", findings=[]))
        return real_review(**kwargs)
    monkeypatch.setattr(static_review, "review_case_inputs", passing_review)
    yield result
    for path in tmp_path.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)



@pytest.fixture
def slurm_scheduler(monkeypatch):
    """Install a local Slurm double that executes real stage wrappers."""
    def install(scenario='failure'):
        from services import run_hpc
        real_run = subprocess.run
        calls = []
        monkeypatch.setenv('WM_PROJECT_VERSION', '10')
        monkeypatch.setenv('FOAM_CONTROLDICT', 'InfoSwitches {}')
        def run(argv, **kwargs):
            calls.append(argv[0])
            if argv[0] == 'sbatch':
                if scenario in {'rejected', 'submission_unknown'}:
                    return subprocess.CompletedProcess(argv, 1, '',
                        'sbatch: error: Invalid account' if scenario == 'rejected' else 'connection lost')
                if scenario != 'missing_marker':
                    real_run(['bash', argv[-1]], check=False, capture_output=True, text=True)
                return subprocess.CompletedProcess(argv, 0, 'Submitted batch job 123\n', '')
            if argv[0] == 'squeue':
                if scenario == 'runtime_unknown' or scenario == 'cancel_confirmed' and 'scancel' not in calls:
                    raise subprocess.CalledProcessError(1, argv, stderr='monitor unavailable')
                return subprocess.CompletedProcess(argv, 0, '', '')
            assert argv[0] == 'scancel'
            return subprocess.CompletedProcess(argv, 0, '', '')
        monkeypatch.setattr(run_hpc.subprocess, 'run', run)
        return calls

    return install
