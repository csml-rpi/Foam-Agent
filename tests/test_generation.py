"""Planning context, approved file generation, and repair progress."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _operations(*required):
    return [dict(stage='solver', purpose='Consume the supplied solver inputs', command='true',
                 executor='script', generated_inputs=[],
                 required_inputs=[dict(path=path, evidence='Declared startup input') for path in required])]


def _solver_state(case, invoke, operations, *, generation='sequential_dependency', **updates):
    from config import Config
    from main import initialize_state
    from services.case_workspace import CaseWorkspace
    from services.stage_plan import render_stage_script

    (case / 'Allrun').write_text(render_stage_script(operations))
    workspace = CaseWorkspace.create(str(case), str(case / '.foamagent'))
    state = initialize_state('Preserve the supplied inputs and make the approved correction',
        Config(execution_mode='fast', input_writer_generation_mode=generation))
    state.update(case_dir=str(case), case_input_source_dir=str(workspace.source_dir),
        llm_service=SimpleNamespace(invoke=invoke), requires_hpc=False,
        solver_operations=operations, tutorial_reference='', case_solver='icoFoam', **updates)
    return state, workspace


@pytest.mark.parametrize('existing,requested,changed', [
    (None, 'A', True),
    (('A', 'created A'), 'A', False),
])
def test_mixed_create_and_unchanged_rewrite_preserve_actual_progress(tmp_path, existing, requested, changed):
    from nodes.input_writer_node import input_writer_node
    from router_func import route_after_input_writer
    from utils import FoamPydantic, read_case_text, save_case_text

    (tmp_path / 'system').mkdir()
    (tmp_path / 'system/B').write_text('already corrected B')
    if existing:
        save_case_text(tmp_path / 'system' / existing[0], existing[1])
    calls = []

    def invoke(user, system, pydantic_obj=None):
        calls.append(pydantic_obj)
        return (FoamPydantic(list_foamfile=[dict(folder_name='system', file_name='B',
                    content='already corrected B')]) if pydantic_obj is FoamPydantic else 'created A')

    task = dict(folder_name='system', file_name=requested, stage='solver', action='create',
                changes='Create the approved A input')
    state, workspace = _solver_state(tmp_path, invoke, _operations('system/A', 'system/B'),
        error_logs=[dict(file='system/A', error_content='Previous native failure')],
        review_analysis='Supply A and retain the corrected B', input_writer_mode='initial',
        subtasks=[task], pending_solver_subtasks=[task],
        rewrite_plan=dict(target_files=[dict(file='system/B', stage='solver', changes='Ensure the correction')]))

    result = input_writer_node(state)

    actual = 'system/A.gz' if requested.endswith('.gz') or (existing and existing[0].endswith('.gz')) else 'system/A'
    assert calls == [None, FoamPydantic]
    assert result['written_files'] == [actual, 'system/B']
    assert result['updated_files'] == ([actual] if changed else [])
    assert result['last_repair']['updated_files'] == result['updated_files']
    assert result['subtasks'] == result['pending_solver_subtasks'] == []
    assert result['input_writer_mode'] == 'rewrite'
    assert read_case_text(workspace.source_dir / 'system/A') == 'created A'
    if changed:
        assert not result['error_logs']
        assert not result.get('termination_reason')
        assert route_after_input_writer({**state, **result}) == 'local_runner'
        assert read_case_text(tmp_path / 'system/A') == 'created A'
    else:
        assert result['termination_reason'] == 'repair_made_no_progress'
        assert result['error_logs'] == state['error_logs']
        assert route_after_input_writer({**state, **result}) == '__end__'
    history = [json.loads(line) for line in (tmp_path / '.foamagent/review_history.jsonl').read_text().splitlines()]
    repairs = [entry for entry in history if entry['event'] == 'file_repair']
    assert len(repairs) == 1 and repairs[0]['updated_files'] == result['updated_files']


@pytest.mark.parametrize('compressed,second_repair', [
    (False, 'changed'),
    (True, 'unchanged'),
])
def test_partial_solver_repairs_review_current_canonical_inputs(tmp_path, monkeypatch, compressed, second_repair):
    from nodes.input_writer_node import input_writer_node
    from nodes.local_runner_node import local_runner_node
    from nodes.solver_reviewer_node import solver_reviewer_node
    from router_func import route_after_input_writer, route_after_reviewer
    from services.case_inputs import case_input_fingerprint
    from services.review import error_fingerprint
    from utils import FoamPydantic, read_case_text, save_case_text

    (tmp_path / 'system').mkdir()
    (tmp_path / 'system/controlDict').write_text('application icoFoam;\n')
    first_name = 'fvSolution' + ('.gz' if compressed else '')
    original = 'solvers { p { tolerance 1e-06; relTol 0.1; } }\n'
    values = ['solvers { p { tolerance 1e-07; relTol 0.1; } }\n',
              'solvers { p { tolerance 1e-07; relTol 0.01; } }\n']
    save_case_text(tmp_path / 'system' / first_name, original)
    (tmp_path / 'system/fvSchemes').write_text('ddtSchemes { default Euler; }\n')
    writes, reviews, runs = [], [], []

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj is FoamPydantic:
            index = len(writes)
            writes.append(user)
            return FoamPydantic(list_foamfile=[dict(folder_name='system',
                file_name=first_name,
                content=values[index if second_repair == 'changed' else 0])])
        reviews.append(user)
        return json.dumps(dict(action='repair_solver', operations=None, operation_request=None,
            diagnosis='Complete the approved solver corrections', evidence=['Current inputs and missing output'],
            reason='Apply the next explicit correction and finish the pending task', target_files=[
                dict(file='system/' + first_name, stage='solver', changes='Apply the approved solver correction'),
                dict(file='system/fvSchemes', stage='solver', changes='Complete the pending scheme correction')]))

    state, workspace = _solver_state(tmp_path, invoke, _operations('system/fvSolution'),
        input_writer_mode='rewrite')
    errors = [dict(file='log.solve', error_content='Controlled failure of the supplied solver inputs')]
    monkeypatch.setattr('nodes.local_runner_node.run_allrun_and_collect_errors',
                        lambda *args, **kwargs: runs.append(args) or errors)
    state.update(local_runner_node(state))
    native_fingerprint = state['last_run_input_fingerprint']
    native_review_fingerprint = error_fingerprint(state)
    state.update(solver_reviewer_node(state))
    assert route_after_reviewer(state) == 'input_writer'
    identities = []

    for index in range(2):
        state.update(input_writer_node(state))
        outcome = state['solver_execution']
        assert outcome == dict(executed=False, error_logs=state['error_logs'])
        assert state['solver_review_history'][-1]['outcome'] == outcome
        assert read_case_text(tmp_path / 'system' / first_name) == original
        identities.append(case_input_fingerprint(workspace.source_dir))
        if index == 1 and second_repair == 'unchanged':
            assert state['termination_reason'] == 'repair_made_no_progress'
            assert not state['updated_files'] and route_after_input_writer(state) == '__end__'
            break
        assert state['updated_files'] == ['system/' + first_name]
        assert state['file_diffs'] and not state.get('termination_reason')
        assert route_after_input_writer(state) == 'solver_reviewer'
        state.update(solver_reviewer_node(state))

    assert len(runs) == 1 and len(writes) == 2 and len(state['failed_run_inputs']) == 1
    assert state['last_run_input_fingerprint'] == native_fingerprint
    # Actual native failures retain the attempted identity even after sources change.
    assert error_fingerprint({**state, 'solver_execution': dict(executed=True),
                              'error_logs': errors}) == native_review_fingerprint
    assert error_fingerprint({**state, 'solver_execution': dict(executed=True),
                              'last_run_input_fingerprint': None}) is None
    if second_repair == 'changed':
        assert identities[0] != identities[1]
        assert len(reviews) == len(state['solver_review_history']) == state['loop_count'] == 3
        assert not state.get('termination_reason') and route_after_reviewer(state) == 'input_writer'
    else:
        assert identities[0] == identities[1] and len(reviews) == 2
        assert state['termination_reason'] == 'repair_made_no_progress'
        assert route_after_reviewer(state) == '__end__'
    assert all(entry['outcome']['executed'] is False for entry in state['solver_review_history'][:2])


@pytest.mark.parametrize('target,indirect', [
    ('foundation-v10', 'include'),
    ('esi-v2006', 'quoted_override'),
])
def test_rewrite_context_uses_native_effective_application(tmp_path, monkeypatch, target, indirect):
    from config import Config
    from services import solver_execution
    from services.input_writer import rewrite_files
    from utils import FoamPydantic

    system = tmp_path / 'system'
    system.mkdir()
    content = {
        'include': 'application icoFoam;\n#include "solverSelection"\n',
        'macro': 'chosenSolver /opt/native/boundaryFoam;\napplication $chosenSolver;\n',
        'substitution': 'application icoFoam;\nselection { application /opt/native/boundaryFoam; }\n$selection;\n',
        'quoted_override': 'application icoFoam;\n"application" "/opt/native/boundaryFoam";\n',
        'quoted_before_literal': '"application" icoFoam;\napplication "/opt/native/boundaryFoam";\n',
        'pattern_after_literal': 'application "/opt/native/boundaryFoam";\n"app.*" icoFoam;\n',
        'pattern_before_literal': '"app.*" icoFoam;\napplication "/opt/native/boundaryFoam";\n',
    }[indirect]
    (system / 'controlDict').write_text(content)
    (system / 'solverSelection').write_text('application /opt/native/boundaryFoam;\n')
    native_calls, prompts = [], []

    def native(argv, **kwargs):
        native_calls.append((argv, kwargs))
        return SimpleNamespace(stdout='Using #codeStream with nativeLibrary\n'
                                     'application "/opt/native/boundaryFoam";\n')

    monkeypatch.setattr(solver_execution, 'run_openfoam_utility', native)
    result = rewrite_files(str(tmp_path), [], 'Apply the approved repair', None,
        'Preserve supplied physics', case_solver='cachedSolver', config=Config(openfoam_target=target),
        llm_service=SimpleNamespace(invoke=lambda *args, **kw: prompts.append(args) or FoamPydantic(list_foamfile=[])))

    assert len(native_calls) == len(prompts) == 1
    argv, kwargs = native_calls[0]
    assert ('-expand' in argv) == (target == 'foundation-v10')
    assert kwargs['openfoam_target'] == target
    assert '<case_solver>/opt/native/boundaryFoam</case_solver>' in prompts[0][0]
    assert 'The selected solver is /opt/native/boundaryFoam.' in prompts[0][1]
    assert not result['error_logs'] and (system / 'controlDict').read_text() == content


@pytest.mark.parametrize('link,writer', [
    ('file', 'initial'),
    ('parent', 'rewrite'),
    ('gzip', 'initial'),
])
def test_whole_batch_rejects_linked_targets_before_any_write(tmp_path, link, writer):
    from services.input_writer import initial_write, rewrite_files
    from utils import FoamPydantic

    (tmp_path / "0").mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "U").write_text("preserve target")
    (tmp_path / "safe").write_text("preserve first input")
    if link == "parent":
        (tmp_path / "0").rmdir()
        (tmp_path / "0").symlink_to(other, target_is_directory=True)
    else:
        (tmp_path / "0" / ("U.gz" if link == "gzip" else "U")).symlink_to(other / "U")
    paths = ["safe", "0/U"]
    response = FoamPydantic(list_foamfile=[dict(folder_name="", file_name="safe", content="changed"),
        dict(folder_name="0", file_name="U", content="changed")])
    client = SimpleNamespace(invoke=lambda *a, **kw: response if writer == "rewrite" else "changed")
    with pytest.raises(ValueError, match="symbolic link"):
        if writer == "initial":
            initial_write(str(tmp_path), [dict(folder_name=str(Path(p).parent),
                file_name=Path(p).name) for p in paths], "Preserve inputs", "", "icoFoam",
                llm_service=client)
        else:
            rewrite_files(str(tmp_path), [], "Apply approved edits",
                {"target_files": [dict(file=p, stage="solver", changes="approved edit") for p in paths]},
                "Preserve inputs", foamfiles=FoamPydantic(list_foamfile=[]), dir_structure={}, llm_service=client)
    assert (tmp_path / "safe").read_text() == "preserve first input"
    assert (other / "U").read_text() == "preserve target"


@pytest.mark.parametrize('helper,fenced', [
    (False, False),
    (True, True),
])
def test_initial_generation_preserves_complete_file_content(tmp_path, helper, fenced):
    from services.input_writer import initial_write

    content = (
        'from pathlib import Path\n'
        'header = "FoamFile { class dictionary; }"\n'
        'Path("generated").write_text(header)\n'
        if helper else
        '// Keep the complete native input, including its leading content.\n'
        'FoamFile { class dictionary; object controlDict; }\napplication icoFoam;\n'
    )
    filename = 'helper.py' if helper else 'controlDict'
    response = ('```' + ('python' if helper else 'foam') + '\n' + content + '```\n'
                if fenced else content)
    calls = []
    result = initial_write(str(tmp_path), [dict(folder_name='', file_name=filename)],
        'Generate this complete declared input', '', 'icoFoam',
        llm_service=SimpleNamespace(invoke=lambda *a: calls.append(a) or response))

    actual = (tmp_path / filename).read_text()
    assert actual == content
    assert result['written_files'] == [filename] and len(calls) == 1
    assert result['foamfiles'].list_foamfile[0].content == content
    if helper:
        compile(actual, filename, 'exec')
