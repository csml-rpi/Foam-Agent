"""Current native execution evidence, result fields, and bounded attempts."""

import ctypes
import gzip
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


def test_old_completion_cannot_prove_current_execution(tmp_path, monkeypatch):
    from services import solver_execution as service
    from services.execution_logs import snapshot_execution_logs
    (tmp_path / "log.solve").write_text("Exec : icoFoam\nEnd\n")
    before = snapshot_execution_logs(tmp_path)
    fields_before = service.snapshot_result_fields(tmp_path)
    (tmp_path / "1").mkdir()
    (tmp_path / "1/U").write_text("FoamFile {class volVectorField;}\n")
    monkeypatch.setattr(service, "run_openfoam_utility", lambda argv, **kw:
        subprocess.CompletedProcess(argv, 0, "application icoFoam;", ""))
    errors = service.check_solver_execution(str(tmp_path), fields_before=fields_before, logs_before=before)
    assert any("No normal completion" in e["error_content"] for e in errors)


@pytest.mark.parametrize('transcript,completed', [
    ('Exec : icoFoam\nExec : checkMesh -constant\nMesh OK.\nEnd\nEnd\n', True),
    ('Exec : icoFoam\nExec : checkMesh -constant\nMesh OK.\nEnd\n', False),
    ('Exec : icoFoam\nEnd\nExec : checkMesh -constant\nMesh OK.\nEnd\n', True),
    ('Exec : icoFoam\nExec : icoFoam\nExec : checkMesh -constant\nEnd\nEnd\n', False),
    ('Exec : icoFoam\nExec : checkMesh -constant\nFOAM FATAL ERROR: child failed\nEnd\nEnd\n', False),
])
def test_native_child_and_sequential_utility_footers_preserve_solver_evidence(
        tmp_path, monkeypatch, transcript, completed):
    from services import solver_execution

    (tmp_path / 'Allrun.out').write_text(transcript)
    (tmp_path / '1').mkdir()
    (tmp_path / '1/U').write_text('FoamFile {class volVectorField;}\n')
    monkeypatch.setattr(solver_execution, 'run_openfoam_utility', lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, 'application icoFoam;\n', ''))

    errors = solver_execution.check_solver_execution(str(tmp_path), fields_before={}, logs_before={})

    assert (not errors) is completed
    if not completed:
        assert any('No normal completion' in item['error_content'] or 'FOAM FATAL ERROR' in item['error_content']
                   for item in errors)


@pytest.mark.parametrize('native_log', ['log.icoFoam', 'logs/log.icoFoam', 'logs/icoFoam'])
def test_fresh_local_attempt_archives_native_log_and_keeps_startup_input(tmp_path, monkeypatch, native_log):
    from services import run_local, solver_execution
    old = "Exec : icoFoam\nEnd\n"
    native = tmp_path / native_log
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text(old)
    (tmp_path / "log.input").write_text(old)
    (tmp_path / "Allrun").write_text(
        f"set -e\ntest ! -e {native_log}\ntest -f log.input\nmkdir -p 1\n"
        "printf 'FoamFile {class volVectorField;}\\n' > 1/U\n"
        f"printf 'Exec : icoFoam\\nEnd\\n' > {native_log}\n")

    def execute(script, out, err, case, *a, **kw):
        result = subprocess.run(["bash", script], cwd=case, capture_output=True, text=True)
        Path(out).write_text(result.stdout)
        Path(err).write_text(result.stderr)
        return {"returncode": result.returncode, "timed_out": False}

    monkeypatch.setattr(run_local, "run_command", execute)
    monkeypatch.setattr(solver_execution, "run_openfoam_utility", lambda argv, **kw:
        subprocess.CompletedProcess(argv, 0, "Using #codeStream with nativeLibrary\napplication icoFoam;\n", ""))
    assert not run_local.run_allrun_and_collect_errors(str(tmp_path), protected_inputs=["log.input"])
    assert (tmp_path / "log.input").read_text() == old
    assert len(list((tmp_path / ".foamagent/log-history").rglob(native.name))) == 1


def forbidden(*args, **kwargs):
    pytest.fail('Conflicting execution outputs reached cleanup, native execution or model setup')


@pytest.mark.parametrize('startup', ['Allrun.out', 'Allrun.err', 'Allrun.out.gz'])
def test_direct_local_execution_reports_output_conflict_before_cleanup(tmp_path, monkeypatch, startup):
    from services import run_local

    (tmp_path / 'Allrun').write_text('true\n')
    path = tmp_path / startup
    original = gzip.compress(b'Supplied startup input') if startup.endswith('.gz') else b'Supplied startup input'
    path.write_bytes(original)
    monkeypatch.setattr(run_local, '_cleanup_run_artifacts', forbidden)
    monkeypatch.setattr(run_local, 'run_command', forbidden)
    errors = run_local.run_allrun_and_collect_errors(str(tmp_path), protected_inputs=[startup])
    assert len(errors) == 1 and errors[0]['file'] == startup
    assert 'conflicts with execution output' in errors[0]['error_content']
    assert path.read_bytes() == original


def test_completed_solver_with_corrupt_gzip_result_reports_verification_failure(tmp_path, monkeypatch):
    from services import solver_execution

    (tmp_path / 'system').mkdir()
    (tmp_path / 'system/controlDict').write_text('application icoFoam;\n')
    (tmp_path / 'log.solve').write_text('Exec : icoFoam\nEnd\n')
    (tmp_path / '1').mkdir()
    field = tmp_path / '1/U.gz'
    encoded = gzip.compress(b'FoamFile {class volVectorField;}\n')
    corrupt = encoded[:10] + b'\x06' + encoded[11:]
    field.write_bytes(corrupt)
    monkeypatch.setattr(solver_execution, 'run_openfoam_utility', lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, 'application icoFoam;', ''))

    errors = solver_execution.check_solver_execution(
        str(tmp_path), fields_before={}, logs_before={})

    assert len(errors) == 1
    assert errors[0]['error_content'].startswith("Cannot verify this attempt's result fields:")
    assert 'invalid block type' in errors[0]['error_content']
    assert field.read_bytes() == corrupt


def case_state(tmp_path, startup='startup.dat'):
    from config import Config
    case = tmp_path / 'case'
    case.mkdir()
    script = '#!/bin/bash\ncat ' + startup + ' >&2\nexit 1\n'
    (case / 'Allrun').write_text(script)
    (case / startup).parent.mkdir(parents=True, exist_ok=True)
    (case / startup).write_text('A')
    cluster = dict(cluster_name='cluster', account_number='account', partition='test', nodes=1,
                   ntasks_per_node=1, time_limit=1, memory=1)
    def invoke(user, system):
        if 'Extract cluster information' in user:
            return json.dumps(cluster)
        return '#!/bin/bash\nbash ' + shlex.quote(re.search(r'using the (.+?) script', user)[1]) + '\n'
    return dict(case_dir=str(case), config=Config(max_loop=1), user_requirement='Run on cluster',
        llm_service=SimpleNamespace(invoke=invoke), loop_count=0, solver_review_history=[],
        solver_operations=[dict(command=script, stage='solver', purpose='Run supplied program', executor='script',
            required_inputs=[dict(path=startup, evidence='program reads it')], generated_inputs=[])])


@pytest.mark.parametrize('runner,startup', [
    ('local', 'startup.dat'),
    ('hpc', 'postProcessing/inlet-profile.dat'),
])
def test_a_b_a_stops_before_third_execution_and_retains_original_errors(tmp_path, monkeypatch, runner, startup, slurm_scheduler):
    from nodes.local_runner_node import local_runner_node
    from nodes.hpc_runner_node import hpc_runner_node
    from services import run_local, solver_execution
    state = case_state(tmp_path, startup)
    calls = []
    real_run = subprocess.run
    def execute(script, out, err, case, *args, **kwargs):
        calls.append('local')
        result = real_run(['bash', script], cwd=case, capture_output=True, text=True)
        Path(out).write_text(result.stdout)
        Path(err).write_text(result.stderr)
        return {'returncode': result.returncode, 'timed_out': False}
    monkeypatch.setattr(run_local, 'run_command', execute)
    monkeypatch.setattr(solver_execution, 'run_openfoam_utility', lambda argv, **kw:
        subprocess.CompletedProcess(argv, 0, 'application icoFoam;', ''))
    if runner == 'hpc':
        calls = slurm_scheduler()
    node = local_runner_node if runner == 'local' else hpc_runner_node
    first_errors = None
    for value in ('A', 'B', 'A'):
        directory = Path(state.get('case_input_source_dir') or state['case_dir'])
        (directory / startup).write_text(value)
        state.update(node(state))
        if first_errors is None:
            first_errors = state['error_logs']
    assert calls.count('local' if runner == 'local' else 'sbatch') == 2
    assert state['termination_reason'] == 'repair_made_no_progress'
    assert state['error_logs'] == first_errors
    assert len(state['failed_run_inputs']) == 2


@pytest.fixture
def child_pid_file(tmp_path):
    # Adopt the script's orphaned helper so this test can also reap it. Product
    # code only needs to stop its own process group; it does not change prctl.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    path = tmp_path / 'child.pid'
    try:
        yield path
    finally:
        if path.exists():
            pid = int(path.read_text())
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def helper_script(body):
    child = ('from pathlib import Path; import time; time.sleep(2.5); '
             'Path("geometry.msh").write_text("late mesh"); time.sleep(30)')
    # The helper inherits the program's output handles, just as a background
    # shell helper can; successful completion must not wait for it to exit.
    return ('import subprocess, sys, time\nfrom pathlib import Path\n'
            f'child = subprocess.Popen([sys.executable, "-c", {child!r}])\n'
            'Path("child.pid").write_text(str(child.pid))\n' + body)


def assert_helper_stopped(path):
    pid = int(path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            status = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        except FileNotFoundError:
            return
        if status == 'Z':
            return
        time.sleep(0.01)
    pytest.fail(f'Attempt returned while its helper {pid} was still running')


@pytest.mark.skipif(sys.platform != 'linux', reason='Child adoption checks use Linux prctl')
@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout'])
def test_local_stage_stops_owned_helper_before_return_and_retains_streams(
        tmp_path, child_pid_file, monkeypatch, outcome):
    from utils import run_command

    runtime = tmp_path / 'runtime'
    (runtime / 'etc').mkdir(parents=True)
    (runtime / 'etc/bashrc').write_text('export WM_PROJECT_VERSION=10\n')
    monkeypatch.setenv('WM_PROJECT_DIR', str(runtime))
    body = ('print("current stdout", flush=True)\n'
            'print("current stderr", file=sys.stderr, flush=True)\n'
            + ('time.sleep(30)\n' if outcome == 'timeout'
               else 'sys.exit(7)\n' if outcome == 'failure' else ''))
    script = tmp_path / 'Allrun'
    script.write_text(shlex.join([sys.executable, '-c', helper_script(body)]) + '\n')
    stdout, stderr = tmp_path / 'Allrun.out', tmp_path / 'Allrun.err'
    result = run_command(str(script), str(stdout), str(stderr), str(tmp_path),
        1 if outcome == 'timeout' else 5, openfoam_target='foundation-v10', fail_fast=True)

    assert_helper_stopped(child_pid_file)
    assert not (tmp_path / 'geometry.msh').exists()
    if outcome == 'timeout':
        assert result['timed_out'] and result['returncode'] == -signal.SIGKILL
        message = 'OpenFOAM execution exceeded the 1 second timeout.\n'
        assert stdout.read_text() == 'current stdout\n' + message
        assert stderr.read_text() == 'current stderr\n' + message
    else:
        assert not result['timed_out']
        assert result['returncode'] == (7 if outcome == 'failure' else 0)
        assert stdout.read_text() == 'current stdout\n'
        assert stderr.read_text() == 'current stderr\n'
