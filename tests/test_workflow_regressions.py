"""Regression coverage for reference-guided generation and direct case repair."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import main
from services.input_writer import build_allrun, initial_write
from services.review import error_fingerprint
from utils import FoamPydantic, FoamfilePydantic, read_case_foamfiles


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
        'config': SimpleNamespace(max_loop=3, openfoam_target='foundation-v10'),
        'error_logs': ['solver error'], 'user_requirement': 'Keep endTime 10',
        'loop_count': 0, 'requires_visualization': False,
        'foamfiles': read_case_foamfiles(str(case)),
        **kwargs,
    }


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


def test_runtime_artifacts_do_not_count_as_changed_inputs(tmp_path):
    case = make_case(tmp_path / 'case')
    state = state_for(case)
    fingerprint = error_fingerprint(state)
    (case / '100').mkdir()
    (case / '100/U').write_text('runtime result')
    (case / 'log.icoFoam').write_text('Time = 100\n')
    assert error_fingerprint(state) == fingerprint
    (case / 'system/controlDict').write_text('application icoFoam; endTime 20;\n')
    assert error_fingerprint(state) != fingerprint


def test_graph_keeps_last_attempt_and_failure_at_repair_limit(tmp_path, monkeypatch):
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
    assert result['termination_reason'] == 'max_review_loop_reached'
    assert 'endTime 99' in (case / 'system/controlDict').read_text()
    assert 'Time = 2' in (case / 'log.icoFoam').read_text()
    assert result['error_logs'] == ['error-2']
    assert result['workflow_status'] == 'failed'
    assert main.workflow_exit_code(result) != 0


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


def test_no_progress_stops_without_another_model_call(tmp_path):
    node = importlib.import_module('nodes.reviewer_node')
    from router_func import route_after_reviewer
    case = make_case(tmp_path / 'case')
    client = FakeLLM(['first review'])
    state = state_for(case, llm_service=client)
    state.update(node.reviewer_node(state))
    state.update(node.reviewer_node(state))
    assert len(client.calls) == 1
    assert state['termination_reason'] == 'repair_made_no_progress'
    from langgraph.graph import END
    assert route_after_reviewer(state) == END
    assert main.workflow_exit_code(state) != 0
