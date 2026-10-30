"""Regression coverage for case generation, repair, checkpoints, and call records."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import main
from dataset_logging import DatasetRecorder
from logger import close_logging, setup_logging
from services.best_snapshot import compute_progress_score, restore_best_snapshot, save_best_snapshot
from services.input_writer import build_allrun, initial_write
from services.review import error_fingerprint
from utils import LLMService, FoamPydantic, FoamfilePydantic, read_case_foamfiles, scan_case_directory


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, user_prompt, system_prompt, pydantic_obj=None, **kwargs):
        self.calls.append((user_prompt, system_prompt, pydantic_obj, kwargs))
        return next(self.responses)


def make_case(root):
    (root / 'system').mkdir(parents=True)
    (root / 'constant').mkdir()
    (root / '0').mkdir()
    (root / 'system/controlDict').write_text('application icoFoam; endTime 10;\n')
    (root / '0/U').write_text('initial velocity\n')
    (root / 'Allrun').write_text('#!/bin/sh\nicoFoam\n')
    return root


def state_for(case, **kwargs):
    return {
        'case_dir': str(case), 'case_origin': 'generated',
        'config': SimpleNamespace(max_loop=3, openfoam_target='foundation-v10', dataset_log_path=''),
        'error_logs': ['solver error'], 'user_requirement': 'Keep endTime 10',
        'loop_count': 0, 'requires_visualization': False,
        'foamfiles': read_case_foamfiles(str(case)),
        **kwargs,
    }


@pytest.mark.parametrize('log_context', [None, {'step': 'rewrite'}])
def test_dataset_basename_and_structured_response(tmp_path, monkeypatch, log_context):
    monkeypatch.chdir(tmp_path)
    client = LLMService.__new__(LLMService)
    client.dataset_recorder = DatasetRecorder('dataset.jsonl', 'case-A')
    client.model_provider = 'openai-codex'
    client.model_version = 'test-only'
    client.retrieval_top3_scores = [0.25]
    client.retrieval_top3_cases = ['reference']
    client.total_calls = client.total_prompt_tokens = client.total_completion_tokens = client.total_tokens = 0
    client.failed_calls = client.retry_count = 0
    response = FoamPydantic(list_foamfile=[])
    client.llm = SimpleNamespace(
        get_num_tokens=lambda text: len(text),
        with_structured_output=lambda schema: SimpleNamespace(invoke=lambda messages: response),
    )
    assert client.invoke('prompt', 'system', pydantic_obj=FoamPydantic,
                         log_context=log_context) == response
    record = json.loads(Path('dataset.jsonl').read_text())
    assert record['case_id'] == 'case-A'
    assert json.loads(record['response']) == {'list_foamfile': []}
    assert record['retrieval_top3_cases'] == ['reference']
    assert record['duration_seconds'] >= 0
    assert client.total_calls == 1


def test_dataset_parallel_writes_are_complete_json_lines(tmp_path):
    path = tmp_path / 'dataset.jsonl'
    recorder = DatasetRecorder(str(path), 'parallel')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: recorder.record(index=i, response='汉字' * 200), range(64)))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(record['index'] for record in records) == list(range(64))


def test_failed_planning_does_not_create_output(tmp_path):
    output = tmp_path / 'case'
    recorder = DatasetRecorder(str(output / 'dataset.jsonl'))
    with pytest.raises(ValueError):
        with recorder.defer():
            recorder.record(step='plan')
            raise ValueError('planning failed')
    assert not output.exists()


def test_planner_flushes_dataset_after_output_preparation(tmp_path, monkeypatch):
    planner = importlib.import_module('nodes.planner_node')
    output = tmp_path / 'case'
    recorder = DatasetRecorder(str(output / 'dataset.jsonl'), 'planning')
    def plan(**kwargs):
        recorder.record(step='plan')
        assert not output.exists()
        return {
            'case_name': 'case', 'case_domain': 'incompressible', 'case_category': 'laminar',
            'case_solver': 'icoFoam', 'case_dir': str(output), 'tutorial_reference': '',
            'case_path_reference': str(output / 'similar_case.txt'),
            'dir_structure_reference': '', 'allrun_reference': '', 'subtasks': [],
        }
    monkeypatch.setattr(planner, 'generate_simulation_plan', plan)
    monkeypatch.setattr(planner, 'llm_requires_custom_mesh', lambda state: 0)
    monkeypatch.setattr(planner, 'llm_requires_hpc', lambda state: False)
    monkeypatch.setattr(planner, 'llm_requires_visualization', lambda state: False)
    try:
        result = planner.planner_node({
            'config': SimpleNamespace(case_dir=str(output)), 'user_requirement': 'case',
            'case_stats': {}, 'llm_service': SimpleNamespace(dataset_recorder=recorder),
        })
        assert result['case_dir'] == str(output)
        assert json.loads((output / 'dataset.jsonl').read_text())['step'] == 'plan'
        assert (output / '.foamagent-output').exists()
    finally:
        close_logging()


@pytest.mark.parametrize('classification', ['reuse', 'modify', 'present_but_unused', 'write_fresh'])
def test_writer_consumes_per_file_advice(tmp_path, classification):
    client = FakeLLM(['application icoFoam;\n'])
    initial_write(str(tmp_path), [{'folder_name': 'system', 'file_name': 'controlDict'}],
                  'endTime 10', 'TUTORIAL_BODY', 'icoFoam', llm_service=client,
                  similar_case_advice={'per_file': [
                      {'folder_name': 'system', 'file_name': 'controlDict',
                       'classification': classification, 'delta': 'endTime must stay 10'},
                  ]})
    prompt = client.calls[0][0]
    assert classification in prompt and 'endTime must stay 10' in prompt
    assert ('TUTORIAL_BODY' in prompt) == (classification in {'reuse', 'modify'})


def test_pre_solver_steps_reach_both_allrun_prompts(tmp_path, monkeypatch):
    writer = importlib.import_module('services.input_writer')
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'openfoam_commands.txt').write_text('blockMesh\nsetFields\nicoFoam\n')
    monkeypatch.setattr(writer, 'retrieve_faiss', lambda *args, **kwargs: [{'full_content': 'command help'}])
    client = FakeLLM([SimpleNamespace(commands=['icoFoam']), '#!/bin/sh\nsetFields\nicoFoam\n'])
    build_allrun(str(tmp_path / 'case'), str(tmp_path), 2, {}, 'icoFoam', '',
                 'standard_mesh', [], llm_service=client, pre_solver_steps=['blockMesh', 'setFields'])
    assert all("['blockMesh', 'setFields']" in call[0] for call in client.calls)


@pytest.mark.parametrize('origin', ['generated', 'imported'])
@pytest.mark.parametrize('mesh', [False, True])
def test_reviewer_uses_direct_repair_for_all_case_origins(tmp_path, origin, mesh):
    node = importlib.import_module('nodes.reviewer_node')
    case = make_case(tmp_path / 'case')
    client = FakeLLM(['Repair the solver configuration'])
    state = state_for(case, case_origin=origin, repairing_mesh=mesh, llm_service=client,
                      rewrite_plan={'target_files': [{'file': 'system/controlDict', 'changes': 'Old plan'}]})
    result = node.reviewer_node(state)
    assert len(client.calls) == 1
    assert result['rewrite_plan'] is None
    assert result['loop_count'] == 1


@pytest.mark.parametrize('origin', ['generated', 'imported'])
def test_direct_mesh_repair_rewrites_before_remeshing(tmp_path, origin):
    from nodes.input_writer_node import input_writer_node
    from nodes.reviewer_node import reviewer_node
    from router_func import route_after_input_writer, route_after_reviewer

    case = make_case(tmp_path / 'case')
    client = FakeLLM(['Correct the mesh dictionary', FoamPydantic(list_foamfile=[
        FoamfilePydantic(folder_name='system', file_name='blockMeshDict', content='blocks ();\n'),
    ])])
    state = state_for(case, case_origin=origin, repairing_mesh=True, llm_service=client)
    state.update(reviewer_node(state))
    assert route_after_reviewer(state) == 'input_writer'
    state.update(input_writer_node(state))
    assert (case / 'system/blockMeshDict').read_text() == 'blocks ();\n'
    assert state['updated_files'] == ['system/blockMeshDict']
    assert route_after_input_writer(state) == 'meshing'
    assert len(client.calls) == 2


def test_snapshot_restore_preserves_logs_dataset_and_state(tmp_path):
    case = make_case(tmp_path / 'case')
    sibling = tmp_path / 'case_best'
    sibling.mkdir()
    (sibling / 'keep').write_text('unrelated')
    (case / 'log.icoFoam').write_text('Time = 8\nsolver failed\n')
    dataset = case / 'training' / 'dataset.jsonl'
    dataset.parent.mkdir()
    dataset.write_text('{"step":"before"}\n')
    state = state_for(case)
    state['config'].dataset_log_path = str(dataset)
    state.update(save_best_snapshot(state))
    (case / 'system/controlDict').write_text('application simpleFoam; endTime 99;\n')
    (case / 'log.icoFoam').write_text('Time = 2\nfailed again\n')
    (case / 'new-broken-file').write_text('remove me')
    dataset.write_text('{"step":"before"}\n{"step":"after"}\n')
    state['termination_reason'] = 'max_review_loop_reached'
    state['error_logs'] = ['later failure']
    try:
        setup_logging(str(case))
        print('complete audit history')
        result = restore_best_snapshot(state)
        print('logging still works after restoration')
    finally:
        close_logging()
    assert 'endTime 10' in (case / 'system/controlDict').read_text()
    assert 'Time = 8' in (case / 'log.icoFoam').read_text()
    assert not (case / 'new-broken-file').exists()
    assert 'after' in dataset.read_text()
    assert 'logging still works' in (case / 'workflow.log').read_text()
    assert (sibling / 'keep').read_text() == 'unrelated'
    assert result['workflow_status'] == 'failed'
    assert result['case_solver'] == 'icoFoam'
    assert result['error_logs'] == ['solver error']
    assert main.workflow_exit_code(result) != 0
    assert '.foamagent' not in result['dir_structure']
    assert not list((case / '.foamagent').glob('previous-*'))


def test_failed_restore_copy_keeps_current_case(tmp_path, monkeypatch):
    snapshot_service = importlib.import_module('services.best_snapshot')
    case = make_case(tmp_path / 'case')
    state = state_for(case)
    state.update(save_best_snapshot(state))
    current = case / 'system/controlDict'
    current.write_text('application simpleFoam;\n')
    monkeypatch.setattr(snapshot_service.shutil, 'copytree', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        restore_best_snapshot(state)
    assert current.read_text() == 'application simpleFoam;\n'


def test_restore_rolls_back_failed_directory_swap(tmp_path, monkeypatch):
    case = make_case(tmp_path / 'case')
    state = state_for(case)
    state.update(save_best_snapshot(state))
    control = case / 'system/controlDict'
    control.write_text('application simpleFoam;\n')
    original = Path.rename
    failed = False
    def rename(path, target):
        nonlocal failed
        if not failed and path.parent.name.startswith('restore-'):
            failed = True
            raise OSError('failed swap')
        return original(path, target)
    monkeypatch.setattr(Path, 'rename', rename)
    with pytest.raises(OSError, match='failed swap'):
        restore_best_snapshot(state)
    assert control.read_text() == 'application simpleFoam;\n'
    assert (case / 'Allrun').exists()


def test_checkpoint_and_dataset_do_not_break_no_progress_detection(tmp_path):
    case = make_case(tmp_path / 'case')
    dataset = case / 'training' / 'dataset.jsonl'
    dataset.parent.mkdir()
    dataset.write_text('first\n')
    state = state_for(case)
    state['config'].dataset_log_path = str(dataset)
    fingerprint = error_fingerprint(state)
    state.update(save_best_snapshot(state))
    dataset.write_text('first\nsecond\n')
    (case / '100').mkdir()
    (case / '100/U').write_text('runtime result')
    assert error_fingerprint(state) == fingerprint


def test_graph_restores_furthest_attempt_and_retains_failure(tmp_path, monkeypatch):
    case = make_case(tmp_path / 'case')
    client = FakeLLM(['repair first run', 'repair later run'])
    runs = 0
    def planner(state):
        return {'case_dir': str(case), 'mesh_type': 'standard_mesh', 'requires_hpc': False,
                'requires_visualization': False}
    def writer(state):
        if state.get('loop_count'):
            (case / 'system/controlDict').write_text('application simpleFoam; endTime 99;\n')
        return {'error_logs': []}
    def runner(state):
        nonlocal runs
        runs += 1
        (case / 'log.icoFoam').write_text(f'Time = {8 if runs == 1 else 2}\n')
        return {'error_logs': [f'error-{runs}']}
    monkeypatch.setattr(main, 'planner_node', planner)
    monkeypatch.setattr(main, 'input_writer_node', writer)
    monkeypatch.setattr(main, 'local_runner_node', runner)
    state = state_for(case, llm_service=client, error_logs=[])
    state['config'].max_loop = 2
    result = main.create_foam_agent_graph().compile().invoke(state)
    assert runs == 2
    assert result['restored_best_snapshot'] is True
    assert result['termination_reason'] == 'max_review_loop_reached'
    assert 'endTime 10' in (case / 'system/controlDict').read_text()
    assert result['error_logs'] == ['error-1']
    assert result['workflow_status'] == 'failed'


@pytest.mark.parametrize('target', ['foundation-v10', 'esi-v2006'])
def test_retrieval_keeps_domain_solver_identity_and_target(monkeypatch, target):
    plan = importlib.import_module('services.plan')
    selected = {'case_name': 'cavity', 'case_domain': 'incompressible',
                'case_category': 'laminar', 'case_solver': 'icoFoam',
                'score': 1.0, 'full_content': '<directory_structure>system/controlDict</directory_structure>'}
    wrong = {**selected, 'case_name': 'other', 'case_domain': 'compressible',
             'score': 0.0, 'full_content': 'WRONG_BODY'}
    wrong_solver = {**selected, 'case_name': 'different-solver',
                    'case_solver': 'simpleFoam', 'score': 0.1}
    correct = {**selected, 'dir_structure': 'system/controlDict', 'full_content': 'CORRECT_BODY'}
    seen = []
    config = SimpleNamespace(openfoam_target=target)
    def retrieve(database, query, **kwargs):
        seen.append(kwargs['config'])
        if database == 'openfoam_tutorials_structure':
            return [wrong, wrong_solver, selected]
        if database == 'openfoam_tutorials_details':
            return [wrong, correct]
        return []
    monkeypatch.setattr(plan, 'retrieve_faiss', retrieve)
    result = plan.retrieve_references('case', 'icoFoam', 'incompressible', 'laminar', config=config)
    assert result[0] == 'CORRECT_BODY'
    assert all(value is config for value in seen)


def test_openclaw_cache_is_supported_without_reading_real_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv('CODEX_HOME', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    cache = tmp_path / '.openclaw/agents/main/agent/auth-profiles.json'
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({'profiles': {'openai-codex:default':
                                             {'access': 'test-placeholder', 'accountId': 'test-account'}}}))
    assert LLMService.__new__(LLMService)._load_codex_oauth() == ('test-placeholder', 'test-account')


def test_progress_ignores_nonfinite_or_malformed_times(tmp_path):
    (tmp_path / 'log.solver').write_text('Time = 1e999\nTime = e\nTime = 0.3\n')
    assert compute_progress_score(str(tmp_path), ['error']) == 0.3


def test_cli_forwards_dataset_arguments_as_distinct_tokens(tmp_path, monkeypatch):
    import foambench_main
    captured = []
    path = str(tmp_path / "training data's.jsonl")
    case_id = "case; literal '$text'"
    monkeypatch.setattr(sys, 'argv', ['foambench_main.py', '--output', str(tmp_path / 'case'),
                                    '--dataset_log_path', path, '--case_id', case_id])
    monkeypatch.setattr(foambench_main, 'run_command', lambda command: captured.append(command))
    foambench_main.main()
    command = captured[0]
    assert command[command.index('--dataset_log_path') + 1] == path
    assert command[command.index('--case_id') + 1] == case_id


def test_mcp_generation_receives_file_advice_and_keeps_retrieval_api(tmp_path, monkeypatch):
    import asyncio
    from src.mcp import fastmcp_server as server
    from services.plan import SimilarCaseAdviceModel, PerFileAdvice
    output = tmp_path / 'mcp-case'
    seen = {}
    advice = SimilarCaseAdviceModel(per_file=[
        PerFileAdvice(folder_name='system', file_name='controlDict', classification='modify', delta='set endTime 10'),
    ])
    monkeypatch.setattr(server, 'require_target_corpus', lambda config: None)
    monkeypatch.setattr(server, 'resolve_case_dir', lambda **kwargs: str(output))
    def retrieve(**kwargs):
        seen['retrieval'] = kwargs
        return 'tutorial', '', '', 'Allrun reference', advice
    monkeypatch.setattr(server, 'retrieve_references', retrieve)
    def write(**kwargs):
        seen['writer'] = kwargs
        (output / 'system').mkdir()
        (output / 'system/controlDict').write_text('application icoFoam; endTime 10;\n')
        (output / 'Allrun').write_text('#!/bin/sh\nicoFoam\n')
        return {'foamfiles': read_case_foamfiles(str(output))}
    monkeypatch.setattr(server, 'initial_write', write)
    async def ignore(*args, **kwargs):
        return None
    context = SimpleNamespace(info=ignore, error=ignore, report_progress=ignore)
    request = server.GenerateFilesRequest(
        case_name='case', subtasks=[{'folder': 'system', 'file': 'controlDict'}],
        user_requirement='endTime 10', case_solver='icoFoam', case_domain='incompressible', case_category='laminar',
    )
    result = asyncio.run(server.input_writer.fn(request, context))
    assert result.case_dir == str(output)
    assert seen['retrieval']['subtasks'] == [{'folder_name': 'system', 'file_name': 'controlDict'}]
    assert seen['retrieval']['user_requirement'] == 'endTime 10'
    assert seen['writer']['similar_case_advice'] is advice


def test_no_progress_restores_without_another_model_call(tmp_path):
    node = importlib.import_module('nodes.reviewer_node')
    from router_func import route_after_reviewer
    case = make_case(tmp_path / 'case')
    client = FakeLLM(['first review'])
    state = state_for(case, llm_service=client)
    state.update(node.reviewer_node(state))
    state.update(node.reviewer_node(state))
    assert len(client.calls) == 1
    assert state['termination_reason'] == 'repair_made_no_progress'
    assert route_after_reviewer(state) == 'restore_best'
    state.update(restore_best_snapshot(state))
    assert state['termination_reason'] == 'repair_made_no_progress'
    assert main.workflow_exit_code(state) != 0


def test_numpy_retrieval_scores_are_normalized_for_dataset_json(monkeypatch):
    import numpy as np
    plan = importlib.import_module('services.plan')
    candidate = {'case_name': 'reference', 'case_domain': 'incompressible',
                 'case_category': 'laminar', 'case_solver': 'icoFoam',
                 'score': np.float32(0.25), 'dir_structure': 'system/controlDict'}
    monkeypatch.setattr(plan, 'retrieve_faiss', lambda index, *a, **k:
                        [candidate] if index == 'openfoam_tutorials_structure' else [])
    client = SimpleNamespace()
    plan.retrieve_references('case', 'icoFoam', 'incompressible', 'laminar', llm_service=client)
    assert type(client.retrieval_top3_scores[0]) is float
    assert json.loads(json.dumps(client.retrieval_top3_scores)) == [0.25]
