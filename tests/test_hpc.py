"""Slurm submission identity, monitoring, cancellation, and batch contracts."""

import json
import re
import shlex
import subprocess
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('reply,output,identity', [
    ('nonzero', 'Submitted batch job 123\n', '123'),
    ('timeout', '123\n', '123'),
    ('exception', 'Submitted batch job 123 on cluster clusterA\n', '123;clusterA'),
    ('nonzero', '123;clusterA\n', '123;clusterA'),
])
def test_reliable_returned_job_id_always_enters_monitoring(monkeypatch, reply, output, identity):
    from services import run_hpc
    def submit(argv, **kwargs):
        if reply == 'timeout':
            raise subprocess.TimeoutExpired(argv, 7, output=output.encode())
        if reply == 'exception':
            raise subprocess.CalledProcessError(1, argv, output=output, stderr='transport closed')
        return subprocess.CompletedProcess(argv, 1, output, 'transport closed')
    monkeypatch.setattr(run_hpc.subprocess, 'run', submit)
    assert run_hpc.submit_slurm_job('submit.slurm') == (identity, True, '')


@pytest.mark.parametrize('reply', ['timeout', 'communication', 'unknown', 'ambiguous'])
def test_unconfirmed_submission_is_not_a_retryable_rejection(monkeypatch, reply):
    from services import run_hpc
    def submit(argv, **kwargs):
        if reply == 'timeout':
            raise subprocess.TimeoutExpired(argv, 7)
        if reply == 'ambiguous':
            return subprocess.CompletedProcess(argv, 0, 'ambiguous submission response', '')
        return subprocess.CompletedProcess(argv, 1, '', 'Socket timed out' if reply == 'communication' else 'unknown failure')
    monkeypatch.setattr(run_hpc.subprocess, 'run', submit)
    with pytest.raises(run_hpc.HpcWorkflowStop) as stop:
        run_hpc.submit_slurm_job('submit.slurm')
    assert stop.value.reason == 'hpc_submission_unconfirmed'


def test_submission_wait_environment_removed_and_timeout_bounded(monkeypatch):
    from services import run_hpc
    monkeypatch.setenv('SBATCH_WAIT', '1')
    def submit(argv, **kwargs):
        assert 'SBATCH_WAIT' not in kwargs['env']
        assert kwargs['timeout'] == 7
        return subprocess.CompletedProcess(argv, 0, 'Submitted batch job 123', '')
    monkeypatch.setattr(run_hpc.subprocess, 'run', submit)
    assert run_hpc.submit_slurm_job('submit.slurm', timeout=7)[0] == '123'


@pytest.mark.parametrize('cluster_suffix,scenario', [
    ('', 'nonzero_id'),
    (';clusterA', 'parsable_id'),
    ('', 'unknown'),
    (';clusterA', 'cancel_confirmed'),
    (';clusterA', 'cancel_unconfirmed'),
    ('', 'rejected_zero'),
])
def test_stage_submission_monitoring_and_stop_confirmation(tmp_path, monkeypatch, scenario, cluster_suffix):
    from config import Config
    from services import run_hpc
    cluster = dict(cluster_name='cluster', account_number='account', partition='test', nodes=1,
                   ntasks_per_node=1, time_limit=1, memory=1)
    def invoke(user, system):
        if 'Extract cluster information' in user:
            return json.dumps(cluster)
        wrapper = re.search(r'using the (.+?) script', user)[1]
        return '#!/bin/bash\n' + ('#SBATCH --parsable\n' if scenario == 'parsable_id' else '') + 'bash ' + shlex.quote(wrapper) + '\n'
    state = dict(case_dir=str(tmp_path), config=Config(max_loop=0 if scenario == 'rejected_zero' else 25), user_requirement='Run on test cluster',
                 llm_service=SimpleNamespace(invoke=invoke), loop_count=0)
    script = tmp_path / 'Allrun'
    script.write_text('true\n')
    real_run = subprocess.run
    calls = []
    monkeypatch.setenv('WM_PROJECT_VERSION', '10')
    monkeypatch.setenv('FOAM_CONTROLDICT', 'InfoSwitches {}')

    def scheduler(argv, **kwargs):
        calls.append(argv[0])
        if argv[0] == 'sbatch':
            assert 0 < kwargs['timeout'] <= 7
            if scenario == 'unknown':
                return subprocess.CompletedProcess(argv, 1, '', 'communication lost')
            if scenario == 'rejected_zero':
                return subprocess.CompletedProcess(argv, 1, '', 'sbatch: error: Invalid account')
            if scenario == 'parsable_id':
                assert '#SBATCH --parsable' in (tmp_path / 'submit_job.slurm').read_text()
            real_run(['bash', argv[-1]], check=True, capture_output=True, text=True)
            return (subprocess.CompletedProcess(argv, 0, '123' + cluster_suffix + '\n', '') if scenario == 'parsable_id'
                    else subprocess.CompletedProcess(argv, 1, 'Submitted batch job 123'
                         + (' on cluster clusterA' if cluster_suffix else '') + '\n', 'connection closed'))
        assert ('--clusters' in argv) is bool(cluster_suffix)
        if cluster_suffix:
            assert argv[argv.index('--clusters') + 1] == 'clusterA'
        if argv[0] == 'squeue':
            assert argv[argv.index('-j') + 1] == '123'
            if scenario == 'cancel_unconfirmed' or scenario == 'cancel_confirmed' and 'scancel' not in calls:
                raise subprocess.CalledProcessError(1, argv, output='', stderr='monitor unavailable')
            return subprocess.CompletedProcess(argv, 0, '', '')
        assert argv[0] == 'scancel'
        assert argv[-1] == '123'
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(run_hpc.subprocess, 'run', scheduler)
    if scenario in {'unknown', 'cancel_unconfirmed', 'rejected_zero'}:
        with pytest.raises(run_hpc.HpcWorkflowStop) as stop:
            run_hpc.run_stage_hpc(state, str(script), 'solver', 7)
        assert stop.value.reason == {
            'unknown': 'hpc_submission_unconfirmed',
            'cancel_unconfirmed': 'hpc_job_state_unconfirmed',
            'rejected_zero': 'max_review_loop_reached',
        }[scenario]
    else:
        result = run_hpc.run_stage_hpc(state, str(script), 'solver', 7)
        assert result['job_id'] == '123' + cluster_suffix
        assert result['returncode'] == (0 if scenario in {'nonzero_id', 'parsable_id'} else 1)
        assert 'squeue' in calls
    assert calls.count('sbatch') == 1
    assert ('scancel' in calls) == scenario.startswith('cancel')


@pytest.mark.parametrize('response', ['not JSON', '[]', '{}'])
def test_invalid_cluster_configuration_stops_before_batch_generation(tmp_path, monkeypatch, response):
    from config import Config
    from nodes.hpc_runner_node import hpc_runner_node
    from services import run_hpc
    from services.stage_plan import render_stage_script

    operations = [dict(stage='solver', purpose='Run the approved solver', command='true',
                       executor='script', required_inputs=[], generated_inputs=[])]
    (tmp_path / 'Allrun').write_text(render_stage_script(operations))
    source = tmp_path / 'supplied-input'
    source.write_text('preserve source')
    prompts = []
    def invoke(user, system):
        prompts.append(user)
        assert 'actual_project' in user and 'partition short' in user
        return response
    monkeypatch.setattr(run_hpc, 'create_slurm_script', lambda *a, **kw: pytest.fail('Generated a batch after invalid configuration'))
    monkeypatch.setattr(run_hpc, 'run_simulation_hpc', lambda *a, **kw: pytest.fail('Submitted after invalid configuration'))
    result = hpc_runner_node(dict(case_dir=str(tmp_path), config=Config(execution_mode='fast'),
        user_requirement='Use account actual_project and partition short',
        llm_service=SimpleNamespace(invoke=invoke), loop_count=0,
        solver_operations=operations, solver_review_history=[]))
    assert result['workflow_status'] == 'failed'
    assert result['termination_reason'] == 'hpc_configuration_failed'
    assert 'Cannot determine cluster configuration' in result['error_logs'][0]['error_content']
    assert len(prompts) == 1 and result['loop_count'] == 0
    assert not result['failed_run_inputs'] and not result['solver_jobs']
    assert not (tmp_path / 'submit_job.slurm').exists()
    assert source.read_text() == 'preserve source'
