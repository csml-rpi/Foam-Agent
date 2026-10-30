"""PR44 workflow and repair regressions, including the follow-up fixes.

Run on Linux: python -m pytest tests/test_pr44.py -q
The graph, services and shell wrappers are real; LLM, OpenFOAM and Slurm
responses are simulated. General import and translator tests remain separate.
"""

import ast
import asyncio
import gzip
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _poly_mesh(root):
    for name in ("points", "faces", "owner", "neighbour"):
        _put(root, f"constant/polyMesh/{name}", "0\n()\n")
    _put(root, "constant/polyMesh/boundary",
         "2\n(\ninlet { type patch; nFaces 1; startFace 0; }\n"
         "wall { type wall; nFaces 2; startFace 1; }\n)\n")


@pytest.fixture
def modules(monkeypatch, tmp_path):
    providers = {'langchain.chat_models': 'init_chat_model', 'langchain_community.vectorstores': 'FAISS', 'langchain_openai.embeddings': 'OpenAIEmbeddings', 'langchain_aws': 'ChatBedrockConverse', 'langchain_anthropic': 'ChatAnthropic', 'langchain_ollama': 'ChatOllama', 'botocore.exceptions': 'ClientError', 'tracking_aws': None, 'tiktoken': None}
    for name, symbol in providers.items():
        try:
            importlib.import_module(name)
        except ImportError:
            for size in range(1, len(name.split('.')) + 1):
                part = '.'.join(name.split('.')[:size])
                if part not in sys.modules:
                    module = ModuleType(part)
                    module.__path__ = []
                    if part == 'langchain':
                        module.debug = module.verbose = False
                        module.llm_cache = None
                    monkeypatch.setitem(sys.modules, part, module)
            if symbol:
                setattr(sys.modules[name], symbol, type(symbol, (Exception,) if symbol == 'ClientError' else (), {}))
    utils = importlib.import_module('utils')
    monkeypatch.setattr(utils, 'LLMService', lambda *_: None)
    names = ('main', 'nodes.solver_planner_node', 'nodes.local_runner_node', 'services.meshing', 'services.mesh', 'services.input_writer', 'services.run_local', 'services.plan', 'services.preparation')
    result = SimpleNamespace(**{name.rsplit('.', 1)[-1]: importlib.import_module(name) for name in names})
    monkeypatch.setattr(result.preparation, 'setup_logging', lambda *_: None)
    yield result
    for path in tmp_path.rglob('*'):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 448)

def _tool_fn(tool):
    """Access the function when FastMCP wraps a registered tool."""
    return getattr(tool, 'fn', tool)


def _state(case, **updates):
    from config import Config
    return {'case_dir': str(case), 'config': Config(openfoam_target='foundation-v10', max_loop=3), 'user_requirement': 'Run the case', 'llm_service': None, **updates}


def _shell(script, out, err, case, *args, **kwargs):
    result = subprocess.run(['bash', script], cwd=case, capture_output=True, text=True)
    Path(out).write_text(result.stdout)
    Path(err).write_text(result.stderr)
    return {'returncode': result.returncode}


def _solver_state(case, llm):
    from config import Config
    from utils import load_case_files
    state = {'case_dir': str(case), 'config': Config(openfoam_target='foundation-v10'), 'case_origin': 'imported', 'llm_service': llm, 'user_requirement': 'Repair the case', 'loop_count': 0, 'solver_planned': True, 'solver_allrun': '', 'input_writer_mode': 'initial', 'requires_input_writer': True, 'requires_hpc': False, 'review_analysis': 'Apply the explicit file changes', 'rewrite_plan': None, 'tutorial_reference': '', 'case_solver': 'icoFoam'}
    return {**state, **load_case_files(state)}


def _check_preparation_allocates_independent_default_runs(modules, tmp_path, monkeypatch, start, overwrite):
    from config import Config
    from services.output_safety import output_is_owned
    config = Config(run_directory=tmp_path / 'runs', run_times=start, overwrite_case_dir=overwrite)
    llm = object()
    monkeypatch.setattr(modules.preparation, 'load_agent_resources', lambda _: (llm, {}))
    state = {'config': config, 'workflow_mode': 'prompt', 'requires_hpc': False}
    first = modules.preparation.prepare_inputs({**state, 'user_requirement': 'Run cavity flow'})
    first_dir = config.run_directory / f'run_{start}'
    assert first['workflow_status'] == 'planning' and Path(first['case_dir']) == first_dir
    _put(first_dir, '0/U', 'existing initial field')
    second = modules.preparation.prepare_inputs({**state, 'user_requirement': 'Run cylinder flow'})
    second_dir = config.run_directory / f'run_{start + 1}'
    assert second['workflow_status'] == 'planning' and Path(second['case_dir']) == second_dir
    assert first['case_name'] == first_dir.name and second['case_name'] == second_dir.name
    assert (first_dir / '0/U').read_text() == 'existing initial field'
    assert output_is_owned(first_dir) and output_is_owned(second_dir)
    assert config.case_dir == '' and config.run_times == start


def _check_default_runs_skip_occupied_names(modules, tmp_path, monkeypatch):
    from config import Config
    runs = tmp_path / 'runs'
    _put(runs, 'run_7/Allrun', 'previous run')
    _put(runs, 'run_8', 'unrelated file')
    (runs / 'run_9').symlink_to(tmp_path / 'missing')
    config = Config(run_directory=runs, run_times=7, overwrite_case_dir=True)
    case = modules.preparation._prepare_generated_case_dir(config, None)
    assert case == runs / 'run_10'
    assert (runs / 'run_7/Allrun').read_text() == 'previous run'
    assert (runs / 'run_8').read_text() == 'unrelated file'
    assert (runs / 'run_9').is_symlink() and (not (tmp_path / 'missing').exists())


def _check_concurrent_default_runs_reserve_different_directories(modules, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from config import Config
    from services.output_safety import output_is_owned
    config = Config(run_directory=tmp_path / 'runs')
    barrier = Barrier(2)
    validate = modules.preparation.validate_output_path

    def validate_together(path, **kwargs):
        result = validate(path, **kwargs)
        if Path(path).name == 'run_1':
            barrier.wait(timeout=10)
        return result
    monkeypatch.setattr(modules.preparation, 'validate_output_path', validate_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        cases = list(pool.map(lambda _: modules.preparation._prepare_generated_case_dir(config, None), range(2)))
    assert set(cases) == {config.run_directory / 'run_1', config.run_directory / 'run_2'}
    assert all((output_is_owned(case) for case in cases))


def _check_explicit_output_keeps_existing_overwrite_rules(modules, tmp_path, monkeypatch):
    from config import Config
    from services.output_safety import OutputDirectorySafetyError, OUTPUT_MARKER_NAME
    output = tmp_path / 'chosen-output'
    config = Config(case_dir=str(output), run_directory=tmp_path / 'unused-runs')
    assert modules.preparation._prepare_generated_case_dir(config, None) == output
    _put(output, '0/U', 'existing field')
    with pytest.raises(FileExistsError):
        modules.preparation._prepare_generated_case_dir(config, None)
    assert (output / '0/U').read_text() == 'existing field'
    config.overwrite_case_dir = True
    assert modules.preparation._prepare_generated_case_dir(config, None) == output
    assert not (output / '0/U').exists()
    (output / OUTPUT_MARKER_NAME).unlink()
    _put(output, 'valuable-input', 'preserve me')
    with pytest.raises(OutputDirectorySafetyError):
        modules.preparation._prepare_generated_case_dir(config, None)
    assert (output / 'valuable-input').read_text() == 'preserve me'
    assert not config.run_directory.exists()


def _check_default_runs_retain_output_path_checks(modules, tmp_path, monkeypatch, unsafe):
    from config import Config
    source = None
    if unsafe == 'symlink_parent':
        target = tmp_path / 'target'
        target.mkdir()
        runs = tmp_path / 'runs'
        runs.symlink_to(target, target_is_directory=True)
    elif unsafe == 'whitespace':
        runs = tmp_path / 'runs with spaces'
    elif unsafe == 'source_directory':
        source = tmp_path / 'source'
        source.mkdir()
        runs = source / 'runs'
    else:
        runs = tmp_path / 'file'
        runs.write_text('preserve me')
    with pytest.raises((OSError, ValueError)):
        modules.preparation._prepare_generated_case_dir(Config(run_directory=runs), source)
    assert not (runs / 'run_1').exists()


def _check_mesh_planner_uses_current_sources_and_early_hpc_routing(modules, tmp_path, monkeypatch):
    _poly_mesh(tmp_path / 'processor0')
    _put(tmp_path, 'duct.msh', 'mesh')
    _put(tmp_path, 'constant/triSurface/duct.stl', 'solid duct')
    _put(tmp_path, '.foamagent/mesh-logs/old.msh', 'archive')
    reconstruction = {'sources': ['processor0/constant/polyMesh'], 'argv': ['reconstructParMesh', '-constant']}

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj is None:
            return 'hpc_run'
        files = ast.literal_eval(re.search('<case_files>(.*?)</case_files>', user, re.S)[1])
        assert 'constant/triSurface/duct.stl' in files
        assert not any(('.foamagent' in name for name in files))
        assert 'processor0/constant/polyMesh' in user
        return pydantic_obj(mesh_type='custom_mesh', mesh_source=next((name for name in files if name.endswith('.msh'))), final_reconstruction=reconstruction)
    llm = SimpleNamespace(invoke=invoke)
    monkeypatch.setattr(modules.preparation, 'prepare_imported_case', lambda _: {'case_dir': str(tmp_path), 'case_context': {'platform': 'foundation-v10'}, 'llm_service': llm})
    state = _state(tmp_path, workflow_mode='imported_case')
    state.update(modules.preparation.prepare_inputs(state))
    assert state['requires_hpc'] is True
    for name in ('duct.msh', 'replacement.msh'):
        plan = modules.meshing.plan_meshing(state)
        assert plan['mesh_source'] == str(tmp_path / name)
        assert plan['final_reconstruction']['sources'] == reconstruction['sources']
        if name == 'duct.msh':
            (tmp_path / name).rename(tmp_path / 'replacement.msh')


def _check_custom_mesh_source_retries_switches_and_same_file(modules, tmp_path, monkeypatch):
    a, b, case = tmp_path / "a.msh", tmp_path / "b.msh", tmp_path / "case"
    a.write_text("A")
    b.write_text("B")
    converted = []

    def convert(argv, **kwargs):
        assert argv == ["gmshToFoam", "geometry.msh"]
        converted.append((case / "geometry.msh").read_text())
        _poly_mesh(case)

    sources = [(a, False), (a, True), (b, True), (b, True), (case / "geometry.msh", False)]
    for index, (source, retry) in enumerate(sources):
        if index == 3:
            (case / "geometry.msh").unlink()
        result = modules.mesh.copy_custom_mesh(str(source), str(case), retry=retry, utility_runner=convert)
        assert not result["error_logs"]
        if index == 0:
            (case / "geometry.msh").write_text("repaired A")
    assert converted == ["A", "repaired A", "B", "B", "B"]


def _check_mesh_handoff_and_cleanup_preserve_required_inputs(modules, tmp_path, monkeypatch):
    _poly_mesh(tmp_path)
    _put(tmp_path, '0/U', 'initial field')
    validate = modules.meshing.validate_mesh_handoff
    for relative in ('0/polyMesh', '1/polyMesh', 'processor0/constant/polyMesh'):
        _put(tmp_path, relative + '/points', 'unresolved mesh')
        assert validate(tmp_path)
        shutil.rmtree(tmp_path / relative)
    _put(tmp_path, '0/polyMesh/cellMap', 'auxiliary')
    assert not validate(tmp_path)
    mesh = tmp_path / 'constant/polyMesh'
    (mesh / 'points').unlink()
    assert validate(tmp_path)
    _poly_mesh(tmp_path)
    for path in mesh.iterdir():
        Path(str(path) + '.gz').write_bytes(gzip.compress(path.read_bytes()))
        path.unlink()
    result = modules.meshing.execute_meshing(_state(tmp_path), lambda _: {'error_logs': []})
    assert not result['error_logs'] and 'inlet' in result['mesh_info']['boundary_text']
    shutil.copytree(mesh, tmp_path / '1/polyMesh')
    _put(tmp_path, '2/U', 'old solver output')
    modules.run_local._cleanup_run_artifacts(str(tmp_path), preserve_mesh=True)
    assert (tmp_path / '1/polyMesh/points.gz').exists()
    assert not (tmp_path / '2').exists()
    assert (tmp_path / '0/U').read_text() == 'initial field'


def _check_final_reconstruction_requires_complete_sources_and_new_output(modules, tmp_path, monkeypatch, outcome):
    _poly_mesh(tmp_path)
    source = 'processor0/constant/polyMesh'
    _poly_mesh(tmp_path / 'processor0')
    old = tmp_path / 'constant/polyMesh/points'
    old.write_text('old serial mesh')
    if outcome == 'missing_source':
        (tmp_path / source / 'points').unlink()

    def reconstruct(argv, **kwargs):
        assert argv == ['reconstructParMesh', '-constant'] and (not old.exists())
        if outcome == 'failed':
            raise subprocess.CalledProcessError(1, argv, stderr='reconstruction failed')
        _poly_mesh(tmp_path)
        if outcome == 'incomplete':
            old.unlink()
        return subprocess.CompletedProcess(argv, 0, 'reconstructed', '')
    monkeypatch.setattr(modules.meshing, 'run_openfoam_utility', reconstruct)
    state = _state(tmp_path, final_reconstruction={'sources': [source], 'argv': ['reconstructParMesh', '-constant']})
    result = modules.meshing.execute_meshing(state, lambda _: {'error_logs': []})
    assert bool(result['error_logs']) == (outcome != 'success')
    if outcome == 'missing_source':
        assert old.read_text() == 'old serial mesh'
    elif outcome == 'success':
        assert old.read_text() != 'old serial mesh' and (tmp_path / source).exists()
        modules.run_local._cleanup_run_artifacts(str(tmp_path))
        assert old.exists() and (not (tmp_path / 'processor0').exists())


def _check_explicit_allmesh_and_mixed_file_context(modules, tmp_path, monkeypatch, action, repair):
    from utils import load_case_files

    _poly_mesh(tmp_path)
    _put(tmp_path, "system/controlDict", "application simpleFoam; endTime 237;")
    script = "#!/bin/sh\nprintf executed > mesh-result\n"
    if action == "modify":
        _put(tmp_path, "Allmesh", "#!/bin/sh\nexit 9\n")

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj:
            assert "endTime 237" in user and "meshScale 0.0037" in user
            files = [dict(folder_name="system", file_name="controlDict",
                          content="application simpleFoam; endTime 237; writeInterval 2;")]
            if action == "modify":
                assert "exit 9" in user
                files.append(dict(folder_name="", file_name="Allmesh", content=script))
            return pydantic_obj(list_foamfile=files)
        return script if "<file_name>Allmesh</file_name>" in system else "meshScale 0.0037;"

    tasks = [dict(folder_name="system", file_name="meshSettings", changes="Set meshScale 0.0037")]
    changes = [dict(file="system/controlDict", stage="mesh", changes="Set writeInterval 2")]
    if action == "create":
        tasks.append(dict(folder_name="", file_name="Allmesh", changes="Execute the prepared mesh script"))
    else:
        changes.append(dict(file="Allmesh", stage="mesh", changes="Run the planned operation"))
    state = _state(tmp_path, llm_service=SimpleNamespace(invoke=invoke), mesh_type="standard_mesh",
                   mesh_commands=[] if repair else ["must not replace the explicit script"],
                   mesh_subtasks=tasks, repairing_mesh=repair,
                   error_logs=["previous mesh failure"] if repair else [],
                   rewrite_plan={"target_files": changes}, review_analysis="Preserve existing settings")
    state.update(load_case_files(state))
    monkeypatch.setattr(modules.meshing, "run_command", _shell)
    monkeypatch.setattr(modules.meshing, "build_allrun", lambda **_: pytest.fail("Explicit Allmesh replaced"))
    result = modules.meshing.execute_meshing(state, modules.meshing.standard_meshing)
    assert not result["error_logs"] and (tmp_path / "mesh-result").read_text() == "executed"
    assert (tmp_path / "Allmesh").read_text() == script
    assert "endTime 237; writeInterval 2" in (tmp_path / "system/controlDict").read_text()


def _check_mesh_retry_rotates_tool_logs_and_preserves_inputs(modules, tmp_path, monkeypatch, source_key):
    inputs = {'input/material.log': 'material', 'constant/triSurface/logo.stl': 'geometry', 'constant/triSurface/log.stl': 'nested geometry', 'input/log.msh': 'uploaded mesh', 'log.stl': 'root geometry', 'log.msh': 'root mesh', 'log.eMesh': 'feature edges', 'log.selected': 'selected mesh', 'workflow.log': 'ERROR: historical workflow', 'review.log': 'ERROR: historical review'}
    for name, content in inputs.items():
        _put(tmp_path, name, content)
    _put(tmp_path, 'log.tool', 'ERROR: old native failure')
    monkeypatch.setattr(modules.meshing, 'run_command', _shell)
    for message, failed in (('ERROR: current native failure', True), ('native success', False)):
        _put(tmp_path, 'Allmesh', f"set -e\ntest ! -e log.tool\nprintf ran > log.tool\nprintf '{message}' > native.out\n")
        state = _state(tmp_path, explicit_allmesh=True, **{source_key: str(tmp_path / 'log.selected')})
        result = modules.meshing._run_mesh_commands(state, [])
        assert result['mesh_execution']['command_result']['returncode'] == 0
        assert bool(result['error_logs']) is failed
        assert result['mesh_execution']['logs']['native.out'] == message
        assert 'workflow.log' not in result['mesh_execution']['logs']
        assert all(((tmp_path / name).read_text() == content for name, content in inputs.items()))
    assert len(list((tmp_path / '.foamagent/mesh-logs').glob('*/log.tool'))) == 2


def _check_explicit_solver_script_preserves_instructions_and_shell(modules, tmp_path, monkeypatch, action, mode):
    writer = importlib.import_module('nodes.input_writer_node')
    script = "#!/bin/sh\ncat <<'EOF'\n$(literal) `unchanged`\nEOF\nicoFoam\n"
    calls = []

    def invoke(user, system, pydantic_obj=None):
        calls.append(user)
        if pydantic_obj:
            return pydantic_obj(list_foamfile=[{'folder_name': '', 'file_name': 'Allrun', 'content': script}])
        return '```sh\n' + script + '```'
    monkeypatch.setattr(writer, '_build_solver_allrun', lambda _: pytest.fail('Explicit Allrun overwritten'))
    state = _solver_state(tmp_path, SimpleNamespace(invoke=invoke))
    state['config'].input_writer_generation_mode = mode
    task = {'folder_name': '', 'file_name': 'Allrun', 'stage': 'solver', 'action': action, 'changes': 'Retain the quoted heredoc before icoFoam'}
    if action == 'create':
        state.update(subtasks=[task], pending_solver_subtasks=[task])
    else:
        state.update(input_writer_mode='rewrite', rewrite_plan={'target_files': [{'file': 'Allrun', 'stage': 'solver', 'changes': task['changes']}]})
    result = writer.input_writer_node(state)
    assert len(calls) == 1 and task['changes'] in calls[0]
    assert (tmp_path / 'Allrun').read_text() == script
    assert not result.get('error_logs') and result['solver_allrun'] is None
    if action == 'create':
        assert result['pending_solver_subtasks'] == result['subtasks'] == []


def _check_pending_solver_tasks_survive_retries_once_with_distinct_edits(modules, tmp_path, monkeypatch):
    review = importlib.import_module('services.review')
    writer = importlib.import_module('nodes.input_writer_node')
    monkeypatch.setattr(review, 'log_review', lambda *args: None)
    monkeypatch.setattr(writer, '_build_solver_allrun', lambda _: pytest.fail('Explicit Allrun overwritten'))
    _poly_mesh(tmp_path)
    (tmp_path / 'system').mkdir()
    (tmp_path / 'system/fvSolution').write_text('original settings')
    edits = [{'file': 'system/fvSolution', 'stage': 'solver', 'changes': 'Correct the pressure tolerance'}, {'file': './system/fvSolution', 'stage': 'solver', 'changes': 'Preserve velocity relaxation'}]
    plans, creates, rewrites = ([], [], [])
    script = '#!/bin/sh\nicoFoam -noFunctionObjects\n'

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj is modules.meshing.MeshingPlan:
            plans.append(user)
            return pydantic_obj(mesh_type='custom_mesh', mesh_subtasks=[{'folder_name': '' if len(plans) == 1 else '.', 'file_name': 'Allrun', 'stage': 'solver', 'action': 'create', 'changes': f'Use solver plan {len(plans)}'}])
        if pydantic_obj is review.MeshRepairPlan:
            return pydantic_obj(target_files=state['rewrite_plan']['target_files'], repair_domain='mesh', action='continue')
        if pydantic_obj is modules.meshing.MeshRepairImpact:
            return pydantic_obj(replan_solver=False, reason='Preserve the physical plan')
        if pydantic_obj:
            rewrites.append(user)
            return pydantic_obj(list_foamfile=[{'folder_name': 'system', 'file_name': 'fvSolution', 'content': 'corrected settings'}])
        if '<file_task>' in user:
            creates.append(user)
            return script
        return 'Retry the mesh while preserving pending solver tasks'
    state = _solver_state(tmp_path, SimpleNamespace(invoke=invoke))
    state.update(repairing_mesh=True, rewrite_plan={'target_files': edits})
    for attempt in range(3):
        state.update(modules.meshing.plan_meshing(state))
        state.update(modules.meshing.execute_meshing(state, lambda _: {'error_logs': [{'file': 'Allmesh', 'error_content': 'failed mesh command'}] if attempt < 2 else []}))
        assert len(state['rewrite_plan']['target_files']) == 2
        if attempt < 2:
            state.update(review.review_case(state))
    assert [t['changes'] for t in state['rewrite_plan']['target_files']] == [t['changes'] for t in edits]
    state.update(writer.input_writer_node(state))
    assert len(creates) == len(rewrites) == 1 and 'Use solver plan 3' in creates[0]
    assert all((t['changes'] in rewrites[0] for t in edits))
    assert (tmp_path / 'Allrun').read_text() == script
    assert (tmp_path / 'system/fvSolution').read_text() == 'corrected settings'
    assert state['pending_solver_subtasks'] == [] and (not state['error_logs'])


def _check_mcp_plan_writer_preserves_root_paths_and_task_metadata(modules, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    server = importlib.import_module('src.mcp.fastmcp_server')
    database = tmp_path / 'database'
    (database / 'raw').mkdir(parents=True)
    (database / 'raw/openfoam_case_stats.json').write_text('{}')
    monkeypatch.setattr(server, 'require_target_corpus', lambda *_: None)
    monkeypatch.setattr(server, 'database_path_for_config', lambda *_: database)
    monkeypatch.setattr(server, 'global_config', server.Config(openfoam_target='foundation-v10'))
    monkeypatch.setattr(server, 'resolve_case_dir', lambda **_: str(tmp_path / 'case'))
    monkeypatch.setattr(server, 'retrieve_references', lambda **_: ('', '', '', '', None))
    tasks = [{'file_name': 'Allrun', 'folder_name': '', 'stage': 'solver', 'action': 'create', 'changes': 'Use icoFoam'}, {'file_name': 'blockMeshDict', 'folder_name': 'system', 'stage': 'mesh', 'action': 'modify', 'changes': 'Keep patches'}, {'file_name': 'U', 'folder_name': '0'}]
    metadata = dict(case_name='case', case_solver='icoFoam', case_domain='incompressible', case_category='laminar')
    monkeypatch.setattr(server, 'generate_simulation_plan', lambda **_: {**metadata, 'subtasks': tasks})
    captured = []

    def generate(**kwargs):
        captured.append(kwargs['subtasks'])
        return {'foamfiles': server.FoamPydantic(list_foamfile=[])}

    async def log(message):
        pass
    monkeypatch.setattr(server, 'initial_write', generate)
    context = SimpleNamespace(info=log, error=log)
    response = asyncio.run(_tool_fn(server.plan)(server.PlanRequest(user_requirement='Run the case'), context))
    assert response.subtasks[0] == {'file': 'Allrun', 'folder': '', 'stage': 'solver', 'action': 'create', 'changes': 'Use icoFoam'}
    for subtasks in (response.subtasks, tasks):
        asyncio.run(_tool_fn(server.input_writer)(server.GenerateFilesRequest(**metadata, user_requirement='Run the case', subtasks=subtasks, overwrite_output=True), context))
    assert captured == [tasks, tasks]


def _check_writer_preserves_explicit_allrun_and_automatic_fallback(modules, tmp_path, monkeypatch, mode, through_mcp, folder):
    writer = modules.input_writer
    case, database = (tmp_path / 'case', tmp_path / 'database')
    _put(database, 'raw/openfoam_commands.txt', 'icoFoam\n')
    explicit = '#!/bin/sh\nicoFoam -noFunctionObjects\n'
    automatic = '#!/bin/sh\nicoFoam\n'
    changes = 'Preserve -noFunctionObjects'
    prompts, progress = ([], [])

    def invoke(user, system, pydantic_obj=None):
        prompts.append((user, system))
        if pydantic_obj:
            return pydantic_obj(commands=['icoFoam'])
        if '<file_name>Allrun</file_name>' in system:
            assert changes in user
            return explicit
        if '<file_name>controlDict</file_name>' in system:
            return 'application icoFoam;'
        return '```sh\n' + automatic + '```'
    monkeypatch.setattr(writer, 'retrieve_faiss', lambda *args, **kwargs: [{'full_content': 'icoFoam help'}])
    monkeypatch.setattr(writer, 'database_path_for_config', lambda *_: database)
    llm = SimpleNamespace(invoke=invoke)
    tasks = [{'folder_name': 'system', 'file_name': 'controlDict'}]
    if folder is not None:
        tasks.append({'folder_name': folder, 'file_name': 'Allrun', 'stage': 'solver', 'action': 'create', 'changes': changes})
    if through_mcp:
        monkeypatch.syspath_prepend(str(ROOT))
        server = importlib.import_module('src.mcp.fastmcp_server')
        monkeypatch.setattr(server, 'require_target_corpus', lambda *_: None)
        monkeypatch.setattr(server, 'database_path_for_config', lambda *_: database)
        monkeypatch.setattr(server, 'resolve_case_dir', lambda **_: str(case))
        monkeypatch.setattr(server, 'retrieve_references', lambda **_: ('', '', '', '', None))
        monkeypatch.setattr(writer, 'global_llm_service', llm)

        async def log(message):
            pass

        async def report_progress(current, total, message):
            progress.append((current, total))
        context = SimpleNamespace(info=log, error=log, report_progress=report_progress)
        response = asyncio.run(_tool_fn(server.input_writer)(server.GenerateFilesRequest(case_name='case', case_solver='icoFoam', case_domain='incompressible', case_category='laminar', user_requirement='Run the case', subtasks=[{'file': t['file_name'], 'folder': t['folder_name'], **{key: t[key] for key in ('stage', 'action', 'changes') if key in t}} for t in tasks]), context))
        assert response.allrun_script == str(case / 'Allrun')
    else:
        result = writer.initial_write(case_dir=str(case), subtasks=tasks, user_requirement='Run the case', tutorial_reference='', case_solver='icoFoam', generation_mode=mode, database_path=str(database), llm_service=llm, progress_callback=lambda current, total, message: progress.append((current, total)))
        root_scripts = [f for f in result['foamfiles'].list_foamfile if f.file_name == 'Allrun' and f.folder_name == '']
        assert len(root_scripts) == 1
        assert root_scripts[0].content == (explicit if folder in ('', '.') else automatic)
    expected = explicit if folder in ('', '.') else automatic
    assert (case / 'Allrun').read_text() == expected
    if folder == 'tools':
        assert (case / 'tools/Allrun').read_text() == explicit
    assert sum(('Available OpenFOAM commands' in user for user, _ in prompts)) == (folder not in ('', '.'))
    assert progress[-1][0] == progress[-1][1]



def _check_workflow(modules, tmp_path, monkeypatch, scenario, target="foundation-v10"):
    m = modules
    if scenario in {"cli_mesh", "cli_case_mesh"}:
        monkeypatch.syspath_prepend(str(ROOT))
        cli = importlib.import_module("foambench_main")
        combined = scenario == "cli_case_mesh"
        for prompt in (["./requirements.txt", None] if combined else ["./requirements.txt"]):
            arguments = ["foambench_main.py", "--custom_mesh_path", "./mesh.msh",
                         "--openfoam_target", target]
            if combined:
                arguments.extend(["--case_path", "./case"])
            if prompt:
                arguments.extend(["--prompt_path", prompt])
            monkeypatch.setattr(sys, "argv", arguments)
            args = cli.parse_args()
            assert args.case_path == ("./case" if combined else None)
            assert args.custom_mesh_path == "./mesh.msh"
            assert args.prompt_path == prompt
            assert args.openfoam_target == target
        return
    if scenario == "standalone_plan":
        from config import Config
        monkeypatch.setattr(m.plan, "parse_requirement_to_case_info", lambda *args, **kw: {
            "case_name": "standalone", "case_domain": "incompressible",
            "case_category": "laminar", "case_solver": "icoFoam",
        })
        monkeypatch.setattr(m.plan, "retrieve_references", lambda **kw:
                            ("reference content", "reference files", "There are 2 files in Directory: system", "reference script", {}))
        class PlannerLLM:
            def invoke(self, user, system, pydantic_obj=None):
                assert "There are 2 files in Directory: system" in user
                assert "mesh is already prepared" not in user
                return pydantic_obj(subtasks=[
                    {"folder_name": "system", "file_name": "blockMeshDict", "stage": "mesh"},
                    {"folder_name": "0", "file_name": "U", "stage": "solver",
                     "action": "modify", "changes": "Set the requested inlet velocity"},
                ])
        result = m.plan.generate_simulation_plan(
            "Plan the simulation", {}, case_dir=str(tmp_path / "standalone"),
            config=Config(openfoam_target=target), llm_service=PlannerLLM(),
        )
        assert result["subtasks"][0]["stage"] == "mesh"
        assert result["subtasks"][1]["action"] == "modify"
        assert result["subtasks"][1]["changes"] == "Set the requested inlet velocity"
        assert result["tutorial_reference"] == "reference content"
        assert result["allrun_reference"] == "reference script"
        assert result["case_path_reference"] == str(tmp_path / "standalone/similar_case.txt")
        return


    if scenario.startswith("pending_physics"):
        from config import Config
        from router_func import route_after_meshing
        writer = importlib.import_module("nodes.input_writer_node")
        case = tmp_path / "case"
        _put(case, "Allrun", "#!/bin/sh\n")
        _poly_mesh(case)
        task = {"folder_name": "0", "file_name": "U", "stage": "solver",
                "action": "create", "changes": ""}
        repair = {"file": "0/p", "stage": "solver", "changes": "Repair pressure"}
        creates, edits = [], []
        def create(state):
            creates.append(state["subtasks"])
            if scenario == "pending_physics_create_failure" and len(creates) == 1:
                return {"error_logs": ["native create failure"]}
            _put(case, "0/U", "created velocity")
            return {}
        def edit(state):
            edits.append(state["rewrite_plan"]["target_files"])
            if scenario == "pending_physics_edit_failure" and len(edits) == 1:
                return {"error_logs": ["native edit failure"]}
            _put(case, "0/p", "repaired pressure")
            return {}
        monkeypatch.setattr(writer, "_initial_write_mode", create)
        monkeypatch.setattr(writer, "_rewrite_mode", edit)
        llm = SimpleNamespace(invoke=lambda *args, **kw: kw["pydantic_obj"](replan_solver=False, reason="Keep the physical plan"))
        state = {"case_dir": str(case), "config": Config(openfoam_target=target),
                 "llm_service": llm, "user_requirement": "Repair mesh and physics", "solver_planned": True,
                 "pending_solver_subtasks": [task], "input_writer_mode": "rewrite",
                 "requires_input_writer": True, "review_analysis": "Mixed repair",
                 "rewrite_plan": {"target_files": [repair]}}
        if scenario == "pending_physics_mesh_failure" and not edits:
            reviewer = importlib.import_module("nodes.reviewer_node")
            review = importlib.import_module("services.review")
            monkeypatch.setattr(review, "review_error_logs", lambda **kw: ("Retry the mesh", []))
            monkeypatch.setattr(review, "generate_rewrite_plan", lambda **kw: {
                "target_files": [], "repair_domain": "mesh", "action": "continue",
            })
            state.update(m.meshing.execute_meshing(state, lambda _: {"error_logs": ["native mesh failure"]}))
            assert route_after_meshing(state) == "reviewer"
            state.update(reviewer.reviewer_node(state))
            assert state["rewrite_plan"]["target_files"] == []
        state.update(m.meshing.execute_meshing(state, lambda _: {"error_logs": []}))
        assert route_after_meshing(state) == "input_writer"
        state.update(writer.input_writer_node(state))
        if scenario.endswith("create_failure"):
            assert state["error_logs"] == ["native create failure"]
            assert state["pending_solver_subtasks"] == [task] and not edits
        elif scenario.endswith("edit_failure"):
            assert state["error_logs"] == ["native edit failure"]
            assert state["pending_solver_subtasks"] == []
        if state.get("error_logs"):
            assert state["rewrite_plan"]["target_files"] == [repair]
            state["error_logs"] = []
            state.update(writer.input_writer_node(state))
        assert state["pending_solver_subtasks"] == [] and state["subtasks"] == []
        assert (case / "0/p").read_text() == "repaired pressure"
        assert all(targets == [repair] for targets in edits)
        _put(case, "0/U", "later physical correction")
        state["rewrite_plan"] = None
        state.update(m.meshing.execute_meshing(state, lambda _: {"error_logs": []}))
        assert route_after_meshing(state) == "input_writer"
        state.update(writer.input_writer_node(state))
        assert (case / "0/U").read_text() == "later physical correction"
        assert len(creates) == (2 if scenario.endswith("create_failure") else 1)
        assert len(edits) == (2 if scenario.endswith("edit_failure") else 1)
        return

    if scenario == "utility_lifecycle":
        from config import Config
        case = tmp_path / "case"
        _put(case, "system/fvSchemes", "user schemes")
        class StartupLLM:
            def invoke(self, user, system, pydantic_obj=None):
                return pydantic_obj(mesh_type="standard_mesh", mesh_subtasks=[{
                    "folder_name": "system", "file_name": "controlDict", "stage": "mesh",
                    "action": "create", "changes": "Use the requested mesh-tool runtime settings",
                }], mesh_commands=["plannedTool"])
        def write(**kwargs):
            assert [(t["folder_name"], t["file_name"]) for t in kwargs["subtasks"]] == [("system", "controlDict")]
            _put(case, "system/controlDict", "planned startup settings")
            return {}
        monkeypatch.setattr(m.meshing, "initial_write", write)
        def backend(state):
            assert (case / "system/controlDict").read_text() == "planned startup settings"
            assert (case / "system/fvSchemes").read_text() == "user schemes"
            assert not (case / "system/fvSolution").exists()
            _put(case, "system/controlDict", "tool-updated settings")
            _poly_mesh(case)
            return {"error_logs": []}
        state = {"case_dir": str(case), "config": Config(openfoam_target=target),
                 "user_requirement": "Prepare only the selected tool's input", "llm_service": StartupLLM()}
        state.update(m.meshing.plan_meshing(state))
        state.update(m.meshing.execute_meshing(state, backend))
        assert not state["error_logs"]
        assert (case / "system/controlDict").read_text() == "tool-updated settings"
        assert (case / "system/fvSchemes").read_text() == "user schemes"
        assert not (case / "system/fvSolution").exists()
        return

    if scenario.startswith("mesh_log_"):
        case = tmp_path / "case"
        _put(case, "log.old", "ERROR: historical failure")
        _put(case, "Allmesh", "#!/bin/sh\nmkdir -p diagnostics\nprintf 'FOAM FATAL IO ERROR: actual redirected failure\\n' > diagnostics/native.log\nprintf 'captured native stdout\\n'\nprintf 'captured native stderr\\n' >&2\nexit 7\n")
        monkeypatch.setattr(m.meshing, "build_allrun", lambda **kw: {"allrun_path": str(case / "Allmesh")})
        def run_mesh(*args, **kwargs):
            import subprocess
            process = subprocess.run(["bash", str(case / "Allmesh")], cwd=case, capture_output=True, text=True)
            _put(case, "Allmesh.out", process.stdout)
            _put(case, "Allmesh.err", process.stderr)
            if scenario == "mesh_log_timeout":
                return {"returncode": 0, "timed_out": True}
            return {"returncode": process.returncode, "timed_out": False}
        monkeypatch.setattr(m.meshing, "run_command", run_mesh)
        original_read = Path.read_text
        if scenario == "mesh_log_read_error":
            def read(path, *args, **kwargs):
                if path.name == "native.log":
                    raise OSError("native log unavailable")
                return original_read(path, *args, **kwargs)
            monkeypatch.setattr(Path, "read_text", read)
        from config import Config
        result = m.meshing._run_mesh_commands({
            "case_dir": str(case), "config": Config(openfoam_target=target),
            "mesh_type": "standard_mesh", "user_requirement": "Run the planned command", "llm_service": None,
        }, ["customCommand"])
        assert result["error_logs"]
        evidence = result["mesh_execution"]
        assert evidence["command_result"]["timed_out"] == (scenario == "mesh_log_timeout")
        assert "log.old" not in evidence["logs"] and list((case / ".foamagent/mesh-logs").glob("*/log.old"))
        assert "captured native stderr" in str(result["error_logs"])
        if scenario == "mesh_log_read_error":
            assert "native log unavailable" in str(result["error_logs"])
        else:
            assert "actual redirected failure" in str(result["error_logs"])
        assert not (case / "system/controlDict").exists()
        return

    if scenario in {"mesh_repeat_stop", "mesh_repeat_budget", "solver_repeat_unchanged"}:
        review = importlib.import_module("services.review")
        case = tmp_path / "case"
        case.mkdir()
        attempts = []
        class RepeatLLM:
            def invoke(self, user, system, pydantic_obj=None):
                name = pydantic_obj.__name__ if pydantic_obj else ""
                if name in {"WorkflowRepairPlan", "MeshRepairPlan"}:
                    attempts.append(user)
                    return pydantic_obj(target_files=[], repair_domain="mesh",
                                        action="stop" if scenario == "mesh_repeat_stop" and len(attempts) == 2 else "continue")
                if name == "MeshingPlan":
                    return pydantic_obj(mesh_type="custom_mesh", mesh_commands=[])
                assert "underlying issues" in system
                return "Try another repair" if not attempts else "The same native error persists"
        state = {"case_dir": str(case), "config": SimpleNamespace(max_loop=3, openfoam_target=target),
                 "user_requirement": "Keep the requested mesh", "llm_service": RepeatLLM(),
                 "mesh_type": "custom_mesh", "mesh_subtasks": [], "loop_count": 0,
                 "error_logs": ["native unchanged failure"]}
        if scenario == "solver_repeat_unchanged":
            state["error_fingerprints"] = [review.error_fingerprint(state)]
            result = review.review_case(state)
            assert result["termination_reason"] == "repair_made_no_progress" and not attempts
            return
        def failure(state):
            return {"error_logs": ["native unchanged failure"]}
        monkeypatch.setattr(m.meshing, "imported_meshing", failure)
        monkeypatch.setattr(m.main, "preparation_node", lambda state: {"workflow_status": "planning"})
        visits = []
        result = dict(state)
        for update in m.main.create_foam_agent_graph().compile().stream(state):
            for node, values in update.items():
                visits.append(node)
                result.update(values)
        assert visits.count("reviewer") == len(attempts)
        assert visits.count("meshing") == len(attempts)
        assert result["workflow_status"] == "failed"
        assert result["error_logs"] == ["native unchanged failure"]
        assert len(attempts) == (2 if scenario == "mesh_repeat_stop" else 3)
        assert result["termination_reason"] == ("mesh_review_stopped" if scenario == "mesh_repeat_stop" else "max_review_loop_reached")
        return

    planned = []
    original_planner = m.main.solver_planner_node
    unparsed = scenario in {"case_unparsed_boundary", "case_unparsed_native_error"}

    def observe_planning(state):
        info = state["mesh_info"]
        assert (Path(info["poly_mesh_path"]) / "boundary").is_file()
        assert info["boundary_text"] == (Path(info["poly_mesh_path"]) / "boundary").read_text()
        planned.append(state)
        return original_planner(state)

    monkeypatch.setattr(m.main, "solver_planner_node", observe_planning)

    if scenario == "target_mismatch":
        from config import Config
        source = tmp_path / "source"
        other_header = (
            "// Website: https://www.openfoam.com\n// Version: v2006\n"
            if target == "foundation-v10"
            else "// Website: https://openfoam.org\n// Version: 10\n"
        )
        _put(source, "system/controlDict", other_header + "application icoFoam;\n")
        config = Config(case_dir=str(tmp_path / "output"), openfoam_target=target)
        meshes = []
        monkeypatch.setattr(m.main, "meshing_planner_node",
                            lambda _: pytest.fail("A target mismatch must stop before mesh planning"))
        monkeypatch.setattr(m.main, "meshing_node", lambda state: meshes.append(state) or {})
        monkeypatch.setattr(m.preparation, "load_agent_resources",
                            lambda _: pytest.fail("A target mismatch must stop before loading resources"))
        state = m.main.initialize_state(
            "Run the imported case", config, workflow_mode="imported_case",
            case_import_path=str(source), requires_visualization=False,
        )
        result = m.main.create_foam_agent_graph().compile().invoke(state)
        assert result["workflow_status"] == "failed"
        assert result["termination_reason"] == "case_target_mismatch"

        unknown_source = tmp_path / "unknown-source"
        _put(unknown_source, "system/controlDict", "application icoFoam;\n")
        unknown_config = Config(case_dir=str(tmp_path / "unknown-output"), openfoam_target="")
        unknown_state = m.main.initialize_state(
            "Run the imported case", unknown_config, workflow_mode="imported_case",
            case_import_path=str(unknown_source), requires_visualization=False,
        )
        unknown = m.main.create_foam_agent_graph().compile().invoke(unknown_state)
        assert unknown["workflow_status"] == "failed"
        assert unknown["termination_reason"] == "Specify --openfoam_target foundation-v10 or esi-v2006."
        assert not meshes and not planned
        return

    imported = scenario.startswith("case_") or scenario in {"solver_retry", "solver_mixed_retry", "solver_mixed_replan"}
    uploaded = scenario in {"custom", "custom_repeated", "external_failure", "upload_generate", "case_upload", "case_external_switch"}
    has_poly = imported and scenario != "case_incomplete"
    solver_script = "#!/bin/sh\n. ${WM_PROJECT_DIR}/bin/tools/RunFunctions\nrestore0Dir\nrunApplication icoFoam -case .\n"
    original_script = solver_script.replace("restore0Dir", "runApplication blockMesh\nrunApplication checkMesh\nrestore0Dir")
    source = tmp_path / "source"
    if imported:
        _put(source, "system/controlDict", ("// Website: https://openfoam.org\n// Version: 10\n" if target == "foundation-v10" else "// Website: https://www.openfoam.com\n// Version: v2006\n") + "application icoFoam;\nendTime 1;\n")
        _put(source, "system/fvSchemes", "ddtSchemes {}\n")
        _put(source, "system/fvSolution", "solvers {}\n")
        _put(source, "constant/transportProperties", "nu 1e-5;\n")
        _put(source, "0/U", "original physical field\n")
        _put(source, "Allrun", original_script)
        if has_poly:
            _poly_mesh(source)
        if unparsed:
            boundary = source / "constant/polyMesh/boundary"
            _put(source, "constant/polyMesh/patches.inc", boundary.read_text(encoding="utf-8").split("(\n", 1)[1].rsplit(")", 1)[0])
            boundary.write_text('2\n(\n#include "patches.inc"\n)\n', encoding="utf-8")
        if scenario == "case_incomplete":
            (source / "system/fvSchemes").unlink()
        if scenario == "case_first_edit":
            _put(source, "system/unusualGeometryInput", "original geometry input")
        if scenario in {"case_regenerate", "case_initial_regenerate"}:
            _put(source, "system/blockMeshDict", "existing mesh dictionary\n")
    custom_path = tmp_path / "uploaded.msh"
    if uploaded:
        custom_path.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")

    calls, mesh_runs, solver_runs, write_calls = [], [], [], []

    class LLM:
        def invoke(self, user, system, pydantic_obj=None):
            calls.append((user, system, pydantic_obj))
            name = pydantic_obj.__name__ if pydantic_obj else ""
            if name == "MeshingPlan":
                repair = "<repair_domain>mesh</repair_domain>" in user
                assert "<custom_mesh_path>" in user and "<case_files>" in user
                if imported and uploaded:
                    assert str(tmp_path / "output/work/input" / custom_path.name) in user
                    assert "original physical field" in user
                if scenario in {"case_prompt", "case_external_switch"}:
                    assert "<user_requirement>Keep the supplied physics and use the available mesh materials.</user_requirement>" in user
                kind = "gmsh_mesh" if scenario == "gmsh" else "custom_mesh" if uploaded or has_poly and not repair else "standard_mesh"
                if scenario in {"upload_generate", "case_initial_regenerate", "standard_snappy"}:
                    kind = "standard_mesh"
                if scenario in {"external_failure", "case_external_switch"}:
                    kind = "custom_mesh"
                if scenario == "missing_mesh":
                    kind = "custom_mesh"
                tasks = []
                case_root = tmp_path / "output" / ("work" if imported else "")
                if kind == "standard_mesh" and not (case_root / "system/blockMeshDict").is_file():
                    tasks = [{"folder_name": "system", "file_name": "blockMeshDict"}]
                commands = ["blockMesh -dict system/blockMeshDict", "checkMesh"] if kind == "standard_mesh" else ["checkMesh"] if has_poly and not uploaded and kind != "gmsh_mesh" else []
                if scenario == "standard_snappy":
                    tasks = [{"folder_name": "system", "file_name": "snappyHexMeshDict"}]
                    commands = ["snappyHexMesh -overwrite", "checkMesh"]
                if scenario in {"missing_mesh", "case_no_commands"} or unparsed:
                    tasks, commands = [], []
                if scenario == "case_first_edit":
                    tasks = [{"folder_name": "system", "file_name": "unusualGeometryInput",
                              "action": "modify", "changes": "Update the supplied geometry input"}]
                if scenario == "case_deferred_physics":
                    tasks = [{"folder_name": "0", "file_name": "extraField", "stage": "solver"}]
                if scenario == "case_command_mesh" and selections:
                    assert "transformPoints" in user
                    tasks, commands = [], ["transformPoints -scale 0.001"]
                if scenario == "custom_repeated":
                    commands = ["checkMesh", "checkMesh"]
                material = (tmp_path / "output" / ("work" if imported else "") / "input" / custom_path.name)
                selected_source = str(material) if uploaded and kind in {"custom_mesh", "gmsh_mesh"} else ""
                if scenario == "case_external_switch" and "bad uploaded mesh" in user:
                    selected_source = ""
                    commands = ["checkMesh"]
                if (commands or selected_source or kind == "gmsh_mesh") and not (case_root / "system/controlDict").exists():
                    tasks.append({"folder_name": "system", "file_name": "controlDict",
                                  "changes": "Provide runtime settings for the selected mesh tools"})
                return pydantic_obj(mesh_type=kind, mesh_source=selected_source,
                                    mesh_subtasks=[{"stage": "mesh", **t} for t in tasks], mesh_commands=commands)
            if name == "ExistingCasePlan":
                if scenario == "case_command_mesh":
                    return pydantic_obj(status="ready", solver_allrun=solver_script,
                                        requires_meshing=len(selections) == 1, target_files=[],
                                        reason="Run transformPoints on the existing mesh")
                return pydantic_obj(status="ready", solver_allrun=solver_script,
                                    target_files=[{"file": "system/fvSchemes", "stage": "solver", "changes": "Restore the missing solver dictionary"}] if scenario == "case_incomplete" else [])
            if name == "OpenFOAMPlanModel":
                assert "There are 2 files in Directory: system" in user
                tasks = [
                    {"folder_name": "system", "file_name": "controlDict", "stage": "solver"},
                    {"folder_name": "0", "file_name": "U", "stage": "solver"},
                ]
                if scenario == "solver_named_mesh":
                    tasks.extend([
                        {"folder_name": "system", "file_name": "meshPhysicsDict", "stage": "solver"},
                        {"folder_name": "", "file_name": "Allrun", "stage": "solver"},
                    ])
                if scenario == "solver_requests_mesh" and len(selections) == 1:
                    tasks.append({"folder_name": "system", "file_name": "unusualGeometryInput", "stage": "mesh",
                                  "action": "modify", "changes": "Prepare the required mesh input"})
                return pydantic_obj(subtasks=tasks)
            if name == "GMSHPythonCode":
                return pydantic_obj(python_code="# mock external Gmsh execution", mesh_type="3D", geometry_type="channel")
            if name == "CommandsPydantic":
                return pydantic_obj(commands=["icoFoam -case ."])
            if name == "MeshRepairImpact":
                return pydantic_obj(replan_solver=scenario in {"solver_mixed_replan", "solver_requests_mesh", "case_command_mesh"}, reason="Impact of the observed mesh changes")
            if name in {"WorkflowRepairPlan", "MeshRepairPlan"}:
                if scenario == "case_regenerate":
                    return pydantic_obj(target_files=[], repair_domain="mesh", action="continue")
                target = "system/blockMeshDict" if scenario == "mesh_retry" else "0/U" if scenario in {"solver_retry", "case_unparsed_native_error"} else "Allmesh"
                targets = [{"file": target, "changes": "Apply the diagnosed repair"}]
                if scenario in {"solver_mixed_retry", "solver_mixed_replan"}:
                    targets.append({"file": "0/U", "changes": "Apply the field correction"})
                for item in targets:
                    item["stage"] = "solver" if item["file"] == "0/U" else "mesh"
                return pydantic_obj(target_files=targets, repair_domain="mesh" if any(t["stage"] == "mesh" for t in targets) else "solver", action="continue")
            if name == "FoamPydantic":
                plan = ast.literal_eval(re.search(r"<rewrite_plan>(.*?)</rewrite_plan>", user, re.S)[1])
                files = []
                for target in plan["target_files"]:
                    path = Path(target["file"])
                    content = "updated geometry input\n" if path.name == "unusualGeometryInput" else "fixed mesh dictionary\n" if path.name == "blockMeshDict" else "fixed physical field\n" if path.name == "U" else "ddtSchemes { default Euler; }\n" if path.name == "fvSchemes" else "#!/bin/sh\nblockMesh -dict system/blockMeshDict\ncheckMesh\n"
                    files.append({"folder_name": "" if path.parent == Path(".") else path.parent.as_posix(), "file_name": path.name, "content": content})
                return pydantic_obj(list_foamfile=files)
            if "underlying issues" in system:
                assert "<actual_mesh>" in user and "<execution>" in user
                if scenario == "case_unparsed_native_error":
                    assert "solver detected an input error" in user
                if scenario == "case_regenerate":
                    return "Regenerate the mesh using the existing blockMeshDict. <repair_domain>mesh</repair_domain>"
                domain = "mesh" if scenario in {"mesh_retry", "solver_mixed_retry", "solver_mixed_replan"} else "solver"
                return f"Repair the reported {domain} failure. <repair_domain>{domain}</repair_domain>"
            if "hpc_run" in system:
                return "local_run"
            if "yes_visualization" in system:
                return "no_visualization"
            if "The script is named Allmesh" in system:
                commands = ast.literal_eval(re.search(r"Execute these planned commands in order: (.*)", user)[1])
                return "#!/bin/sh\n" + "\n".join(commands) + "\n"
            if "The script is Allrun" in system:
                return solver_script
            match = re.search(r"<file_name>(.*?)</file_name>.*?<folder_name>(.*?)</folder_name>", system, re.S)
            if match:
                file, folder = match.groups()
                write_calls.append((folder, file, user))
                return solver_script if file == "Allrun" else "application icoFoam;\nendTime 1;\n" if file == "controlDict" else f"generated {file}\n"
            if "extract boundary names" in system:
                return "inlet,wall"
            if "polyMesh boundary file" in system:
                return "2\n(\ninlet { type patch; nFaces 1; startFace 0; }\nwall { type wall; nFaces 2; startFace 1; }\n)\n"
            raise AssertionError(f"Unexpected LLM call: {system[:120]}")

    llm = LLM()

    monkeypatch.setattr(m.preparation, "load_agent_resources", lambda _: (llm, {}))
    from config import Config
    config = Config(case_dir=str(tmp_path / "output"), database_path=str(tmp_path / "db"), max_loop=5, openfoam_target=target)
    _put(tmp_path / "db", f"{target}/raw/openfoam_commands.txt", "blockMesh\ncheckMesh\nicoFoam\n")
    monkeypatch.setattr(m.input_writer, "retrieve_faiss", lambda *_, **__: [{"full_content": "command help"}])

    selections, retrievals = [], []
    def choose_solver(requirement, *args, **kwargs):
        assert "<actual_mesh>" in requirement
        if len(selections) == 0 and (not imported or scenario == "case_incomplete"):
            assert not (tmp_path / "output" / ("work" if imported else "") / "system/fvSchemes").exists()
        assert (tmp_path / "output" / ("work" if imported else "") / "constant/polyMesh/boundary").is_file()
        selections.append(requirement)
        return {"case_name": "case", "case_domain": "incompressible", "case_category": "laminar", "case_solver": "icoFoam"}
    def retrieve(**kwargs):
        assert selections
        retrievals.append(kwargs)
        return "reference", {}, "There are 2 files in Directory: system", original_script, {}
    monkeypatch.setattr(m.plan, "parse_requirement_to_case_info", choose_solver)
    monkeypatch.setattr(m.plan, "retrieve_references", retrieve)

    def run_script(script, stdout, stderr, case_dir, *_, **__):
        case = Path(case_dir)
        content = Path(script).read_text(encoding="utf-8")
        Path(stdout).write_text("", encoding="utf-8")
        Path(stderr).write_text("", encoding="utf-8")
        if Path(script).name == "Allmesh":
            mesh_runs.append(content)
            assert "icoFoam" not in content and "restore0Dir" not in content
            assert not selections or scenario in {"solver_mixed_retry", "solver_mixed_replan", "solver_requests_mesh", "case_command_mesh"}
            assert (case / "system/controlDict").is_file()
            if not imported and not selections:
                assert not (case / "system/fvSchemes").exists()
                assert not (case / "system/fvSolution").exists()
            if scenario == "mesh_retry" and len(mesh_runs) == 1:
                Path(stderr).write_text("bad mesh dictionary", encoding="utf-8")
                return {"returncode": 1}
            if scenario == "case_regenerate" and "blockMesh" not in content:
                Path(stderr).write_text("existing mesh requires regeneration", encoding="utf-8")
                return {"returncode": 1}
            if "blockMesh" in content or "snappyHexMesh" in content:
                if scenario == "mesh_retry":
                    assert "fixed" in (case / "system/blockMeshDict").read_text(encoding="utf-8")
                _poly_mesh(case)
        else:
            solver_runs.append(content)
            assert all(token not in content for token in ("blockMesh", "checkMesh", "gmshToFoam", "Allmesh"))
            assert (case / "constant/polyMesh/boundary").is_file()
            assert "restore0Dir" in content and "icoFoam -case ." in content
            if scenario in {"solver_retry", "solver_mixed_retry", "solver_mixed_replan", "case_unparsed_native_error"} and len(solver_runs) == 1:
                Path(stderr).write_text("solver detected an input error", encoding="utf-8")
                return {"returncode": 1}
        return {"returncode": 0}
    monkeypatch.setattr(m.meshing, "run_command", run_script)
    monkeypatch.setattr(m.run_local, "run_command", run_script)

    conversions, gmsh_calls = [], []
    import subprocess
    def convert(args, **kwargs):
        conversions.append(args)
        assert (Path(kwargs["working_dir"]) / "system/controlDict").is_file()
        assert kwargs["openfoam_target"] == target
        if scenario in {"external_failure", "case_external_switch"} and len(conversions) == 1:
            raise subprocess.CalledProcessError(1, args, stderr="bad uploaded mesh; retry conversion")
        if args[0] == "gmshToFoam":
            _poly_mesh(Path(kwargs["working_dir"]))
        return SimpleNamespace(stdout="Mesh OK.", stderr="")
    monkeypatch.setattr(m.mesh, "run_openfoam_utility", convert)
    def gmsh(args, **kwargs):
        gmsh_calls.append(args)
        assert not selections and not retrievals
        _put(Path(kwargs["cwd"]), "geometry.msh", "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        return SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(m.mesh.subprocess, "run", gmsh)

    explicit_requirement = not imported or scenario in {"case_prompt", "case_external_switch"}
    requirement = "Keep the supplied physics and use the available mesh materials." if explicit_requirement and imported else "Run the case"
    state = m.main.initialize_state(
        requirement, config, str(custom_path) if uploaded else None,
        workflow_mode="imported_case" if imported else "prompt",
        case_import_path=str(source) if imported else None,
        user_requirement_explicit=explicit_requirement,
        requires_visualization=False,
    )
    if scenario == "missing_mesh":
        state.update(m.preparation.prepare_inputs(state))
        state.update(m.meshing.plan_meshing(state))
        result = m.meshing.execute_meshing(state, m.meshing.imported_meshing)
        assert result["error_logs"][0]["file"] == "mesh_handoff"
        assert result["repairing_mesh"]
        assert not (Path(state["case_dir"]) / "constant/polyMesh").exists()
        assert not planned and not selections and not solver_runs
        return
    graph = m.main.create_foam_agent_graph().compile()
    visits, result = [], None
    for mode, update in graph.stream(state, stream_mode=["updates", "values"]):
        if mode == "updates":
            visits.extend(update)
        else:
            result = update
    assert result["workflow_status"] == "success" and not result["error_logs"]
    assert result["mesh_info"]["boundary_text"] == (Path(result["mesh_info"]["poly_mesh_path"]) / "boundary").read_text()
    assert visits.index("preparation") < visits.index("meshing_planner") < visits.index("meshing") < visits.index("solver_planner") < visits.index("input_writer") < visits.index("local_runner")
    assert visits.count("meshing_planner") == visits.count("meshing") == (2 if scenario in {"solver_mixed_retry", "solver_mixed_replan", "solver_requests_mesh", "case_command_mesh",
                                                                                         "mesh_retry", "case_regenerate", "external_failure", "case_external_switch"} else 1)
    assert result["solver_allrun"] is None
    assert len(gmsh_calls) == (1 if scenario == "gmsh" else 0)
    tools = [command[0] for command in conversions]
    if scenario == "external_failure":
        assert tools == ["gmshToFoam", "gmshToFoam"]
    elif scenario == "case_external_switch":
        assert tools == ["gmshToFoam"]
        assert result["mesh_source"] is None
    elif scenario in {"custom", "custom_repeated", "case_upload"}:
        assert tools == ["gmshToFoam"]
    elif scenario == "gmsh":
        assert tools == ["gmshToFoam", "checkMesh"]
    else:
        assert not conversions
    assert len(selections) == len(retrievals) == (2 if scenario in {"solver_mixed_replan", "solver_requests_mesh", "case_command_mesh"} else 1)
    if scenario == "standard_snappy":
        assert "blockMesh" not in mesh_runs[0]
    if scenario in {"upload_generate", "case_initial_regenerate"}:
        assert "blockMesh" in mesh_runs[0]
    if imported:
        assert (source / "Allrun").read_text(encoding="utf-8") == original_script
        assert (tmp_path / "output/original/Allrun").read_text(encoding="utf-8") == original_script
        assert result["mesh_info"]["poly_mesh_path"]
        if scenario == "case_incomplete":
            assert not (source / "system/fvSchemes").exists()
            assert "Euler" in (Path(result["case_dir"]) / "system/fvSchemes").read_text()
        assert (Path(result["case_dir"]) / "0/U").read_text(encoding="utf-8") == ("fixed physical field\n" if scenario in {"solver_retry", "solver_mixed_retry", "solver_mixed_replan", "case_unparsed_native_error"} else "original physical field\n")
        assert (Path(result["case_dir"]) / "system/controlDict").read_text(encoding="utf-8") == (source / "system/controlDict").read_text(encoding="utf-8")
    if scenario == "case_no_commands" or unparsed:
        assert mesh_runs == []
    if uploaded:
        assert len(mesh_runs) == 1 and mesh_runs[0].count("checkMesh") == (3 if scenario == "custom_repeated" else 1)
    if scenario == "mesh_retry":
        assert len(mesh_runs) == 2 and visits.count("reviewer") == 1 and len(solver_runs) == 1
    if scenario == "case_regenerate":
        assert len(mesh_runs) == 2 and len(solver_runs) == 1
        assert "blockMesh" not in mesh_runs[0] and "blockMesh" in mesh_runs[1]
        assert (Path(result["case_dir"]) / "system/blockMeshDict").read_text() == "existing mesh dictionary\n"
    if scenario in {"solver_retry", "solver_mixed_retry", "solver_mixed_replan", "case_unparsed_native_error"}:
        assert len(solver_runs) == 2 and visits.count("reviewer") == 1
    if not imported:
        field_prompt = next(prompt for folder, name, prompt in write_calls if folder == "0" and name == "U")
        assert "inlet" in field_prompt and "nFaces" in field_prompt
    assert result["solver_planned"] is True
    assert result["user_requirement"] == requirement
    assert visits.count("solver_planner") == len(planned) == len(selections)
    planning_prompts = [
        user for user, _, model in calls
        if model and model.__name__ in {"OpenFOAMPlanModel", "ExistingCasePlan"}
    ]
    if not imported:
        assert len(planning_prompts) == len(selections)
    for prompt in planning_prompts:
        assert "<actual_mesh>" in prompt
        if unparsed:
            assert "patches.inc" in prompt and "boundary_text" in prompt
        else:
            assert "inlet" in prompt and "nFaces" in prompt
    if unparsed:
        assert "patches.inc" in selections[0] and "boundary_text" in selections[0]
        assert not any("underlying issues" in system for _, system, _ in calls) or scenario == "case_unparsed_native_error"
    if scenario in {"solver_mixed_retry", "solver_mixed_replan"}:
        assert result["rewrite_plan"]["target_files"] == [
            {"file": "0/U", "changes": "Apply the field correction", "stage": "solver"},
        ]
    if scenario == "solver_mixed_replan":
        assert "Impact of the observed mesh changes" in selections[-1]
        assert "Apply the field correction" in selections[-1]

    if scenario == "case_first_edit":
        assert (source / "system/unusualGeometryInput").read_text() == "original geometry input"
        assert (Path(result["case_dir"]) / "system/unusualGeometryInput").read_text() == "updated geometry input\n"
        assert result["loop_count"] == 0
    if scenario == "solver_requests_mesh":
        assert (Path(result["case_dir"]) / "system/unusualGeometryInput").is_file()
        assert visits.count("solver_planner") == 2 and len(solver_runs) == 1
    if scenario == "case_command_mesh":
        assert mesh_runs[-1].strip() == "#!/bin/sh\ntransformPoints -scale 0.001"
        assert visits.count("solver_planner") == 2 and len(solver_runs) == 1
    if scenario == "solver_requests_mesh":
        assert not result.get("pending_solver_subtasks")
        assert sum(folder == "0" and file == "U" for folder, file, _ in write_calls) == 1
    if scenario == "solver_named_mesh":
        assert (Path(result["case_dir"]) / "system/meshPhysicsDict").is_file()
        assert any(file == "Allrun" for _, file, _ in write_calls)
        assert visits.count("meshing") == 1

    if scenario == "case_deferred_physics":
        assert (Path(result["case_dir"]) / "0/extraField").is_file()
        assert not (source / "0/extraField").exists()
        assert result["loop_count"] == 0


_WORKFLOW_CASES = [
    (scenario.replace("_", "-"), {"scenario": scenario})
    for scenario in (
        "cli_mesh cli_case_mesh standard gmsh "
        "custom case_poly case_upload case_prompt "
        "target_mismatch standalone_plan upload_generate case_initial_regenerate "
        "standard_snappy case_no_commands case_external_switch case_incomplete "
        "case_first_edit solver_named_mesh solver_requests_mesh case_deferred_physics "
        "case_command_mesh external_failure missing_mesh case_unparsed_boundary "
        "case_unparsed_native_error utility_lifecycle mesh_log_capture mesh_log_timeout "
        "mesh_log_read_error custom_repeated mesh_retry case_regenerate "
        "solver_retry solver_mixed_retry solver_mixed_replan mesh_repeat_stop "
        "mesh_repeat_budget solver_repeat_unchanged pending_physics pending_physics_create_failure "
        "pending_physics_edit_failure pending_physics_mesh_failure "
    ).split()
]
_WORKFLOW_CASES += [
    (scenario.replace("_", "-") + "-esi", {"scenario": scenario, "target": "esi-v2006"})
    for scenario in ("standard", "gmsh", "custom", "case_poly", "target_mismatch")
]


def _mixed_repair_graph(modules, monkeypatch, tmp_path):
    """Run the real graph/routes/reviewer with successful mesh-stage stand-ins."""
    review = importlib.import_module("services.review")
    attempts, visits = [], []
    options = SimpleNamespace(repairs=None, replan=False)

    def solver_planner(state):
        visits.append("solver_planner")
        return {"solver_planned": True, "replan_solver": False,
                "repairing_mesh": False}

    def runner(state):
        attempts.append(len(attempts) + 1)
        # Make every repair observable to the real no-progress guard.
        _put(tmp_path, "system/controlDict", f"deltaT {1 / len(attempts)};\n")
        if options.repairs is None or len(attempts) <= options.repairs:
            return {"error_logs": [{"file": "log.solver",
                                    "error_content": "A further mesh repair is required."}]}
        return {"error_logs": [], "workflow_status": "success", "termination_reason": None}

    nodes = {
        "preparation_node": lambda _: {"case_dir": str(tmp_path), "workflow_status": "running",
                                      "requires_hpc": False, "requires_visualization": False},
        "meshing_planner_node": lambda _: {"mesh_type": "standard_mesh"},
        "meshing_node": lambda _: {"error_logs": [], "repairing_mesh": False,
                                   "replan_solver": options.replan},
        "solver_planner_node": solver_planner,
        "input_writer_node": lambda _: {"error_logs": []},
        "local_runner_node": runner,
    }
    for name, node in nodes.items():
        monkeypatch.setattr(modules.main, name, node)
    monkeypatch.setattr(review, "review_error_logs", lambda **_: ("Repair mesh and retain physics.", []))
    monkeypatch.setattr(review, "generate_rewrite_plan", lambda **_: {
        "target_files": [], "repair_domain": "mesh",
    })

    def run(config, repairs, replan):
        options.repairs, options.replan = repairs, replan
        return modules.main.main("Run the case and repair mesh errors.", config)

    return SimpleNamespace(run=run, attempts=attempts, visits=visits)


def _check_recursion(
    modules, tmp_path, monkeypatch, max_loop=25, repairs=24,
    replan_solver=False, explicit_limit=None, expect_error=False,
):
    """The review budget must fit the graph unless the caller sets a smaller limit."""
    from config import Config

    graph = _mixed_repair_graph(modules, monkeypatch, tmp_path)
    overrides = {"recursion_limit": explicit_limit} if explicit_limit is not None else {}
    config = Config(max_loop=max_loop, **overrides)
    if expect_error:
        from langgraph.errors import GraphRecursionError
        with pytest.raises(GraphRecursionError):
            graph.run(config, repairs, replan_solver)
        assert config.recursion_limit == explicit_limit
        return

    result = graph.run(config, repairs, replan_solver)
    exhausted = repairs is None
    assert result["workflow_status"] == ("failed" if exhausted else "success")
    assert result["termination_reason"] == ("max_review_loop_reached" if exhausted else None)
    assert result["loop_count"] == (max_loop if exhausted else repairs)
    assert len(graph.attempts) == (max_loop if exhausted else repairs + 1)
    assert graph.visits.count("solver_planner") == (
        1 + (result["loop_count"] - exhausted) * replan_solver
    )
    if explicit_limit is not None:
        assert config.recursion_limit == explicit_limit


_RECURSION_CASES = [
    ("default-budget-direct-writer", {}),
    ("default-budget-solver-replan", {"replan_solver": True}),
    ("shared-review-budget-exhausted", {"repairs": None, "replan_solver": True}),
    ("custom-review-budget", {"max_loop": 40, "repairs": 39}),
    ("explicit-larger-recursion-limit", {"explicit_limit": 200, "replan_solver": True}),
    ("explicit-smaller-recursion-limit", {
        "repairs": 20, "explicit_limit": 12, "replan_solver": True, "expect_error": True,
    }),
]




class _HpcScheduler:
    """Fake Slurm/time while real bash exercises the generated batch and wrapper."""

    def __init__(self, monkeypatch, hpc):
        self.clock = self.queue_until = self.query_seconds = 0
        self.execute = self.confirm_cancel = True
        self.submissions, self.calls, self.batch = [], [], None
        self.real_run = subprocess.run
        monkeypatch.setattr(hpc.subprocess, "run", self.run)
        monkeypatch.setattr(hpc, "time", SimpleNamespace(monotonic=lambda: self.clock, sleep=self.sleep))

    def sleep(self, seconds):
        self.clock += seconds

    def run(self, args, **kwargs):
        if args[0] not in {"sbatch", "squeue", "scancel"}:
            return self.real_run(args, **kwargs)
        self.calls.append(args)
        if args[0] == "sbatch":
            outcome = self.submissions.pop(0) if self.submissions else "accepted"
            if outcome == "rejected":
                raise subprocess.CalledProcessError(1, args, stderr="invalid partition")
            if outcome == "unknown":
                return subprocess.CompletedProcess(args, 0, "ambiguous response", "")
            self.batch = args[-1]
            output = f"Submitted batch job {100 + len(self.calls)}\n"
        elif args[0] == "squeue":
            elapsed = min(self.query_seconds, kwargs["timeout"])
            self.clock += elapsed
            if elapsed == kwargs["timeout"]:
                raise subprocess.TimeoutExpired(args, elapsed)
            output = "PENDING" if self.clock < self.queue_until else ""
            if not output and self.execute and self.batch:
                self.real_run(["bash", self.batch], capture_output=True)
                self.batch = None
        else:
            if self.confirm_cancel:
                self.queue_until, self.batch = 0, None
            output = ""
        return subprocess.CompletedProcess(args, 0, output, "")


def _hpc_context(modules, tmp_path, monkeypatch):
    from config import Config

    hpc = importlib.import_module("services.run_hpc")
    prompts, options = [], SimpleNamespace(partition="normal", reviews=0)

    def invoke(user, system, pydantic_obj=None):
        prompts.append(user)
        if "Extract cluster information and return as JSON object." in user:
            count = re.search(r"numberOfSubdomains\s+(\d+)", user)
            return '{"ntasks_per_node": ' + (count[1] if count else "1") + '}'
        if "Create a SLURM script" in user:
            wrapper = re.search(r"using the (.+) script\.", user)[1]
            count = re.search(r"Tasks per node: (\d+)", user)[1]
            return f'#!/bin/bash\n#SBATCH -n {count}\n#SBATCH -p {options.partition}\nexec "{wrapper}"\n'
        if pydantic_obj:
            return pydantic_obj(target_files=[], repair_domain="solver")
        options.reviews += 1
        return "Repair the solver failure."

    state = _state(tmp_path, user_requirement="Run on HPC", loop_count=0,
                   config=Config(openfoam_target="foundation-v10", max_loop=3, max_time_limit=1),
                   llm_service=SimpleNamespace(invoke=invoke))
    _put(tmp_path, "Allrun", "exit 0\n")
    monkeypatch.setenv("WM_PROJECT_VERSION", "10")
    return SimpleNamespace(hpc=hpc, node=importlib.import_module("nodes.hpc_runner_node"),
                           review=importlib.import_module("services.review"), state=state,
                           script=tmp_path / "Allrun", scheduler=_HpcScheduler(monkeypatch, hpc),
                           prompts=prompts, options=options)


_HPC_CASES = (
    [(f"hpc-exit-{code if code is not None else 'missing'}", {"scenario": "exit", "exit_code": code})
     for code in (0, 7, None)]
    + [("hpc-mesh-deadline-resource-refresh", {"scenario": "resources"})]
    + [(f"hpc-shared-budget-{budget}", {"scenario": "budget", "budget": budget}) for budget in (2, 3)]
    + [(f"hpc-stop-{failure}", {"scenario": "uncertain", "failure": failure})
       for failure in ("unknown", "active_job")]
    + [("hpc-query-deadline", {"scenario": "query_deadline"})]
    + [(f"hpc-mesh-stop-{backend}", {"scenario": "mesh_stop", "backend": backend})
       for backend in ("custom", "gmsh")]
    + [(f"hpc-fingerprint-{change or ('missing-result' if code is None else 'unchanged')}",
        {"scenario": "fingerprint", "change": change, "exit_code": code})
       for change, code in ((None, 7), (None, None), ("input", 7), ("submission", 7))]
    + [(f"hpc-prompt-{'retry' if retry else 'initial'}-{'wrapper' if custom else 'default'}",
        {"scenario": "prompt", "retry": retry, "custom_script": custom})
       for retry in (False, True) for custom in (False, True)]
)


def _check_hpc(modules, tmp_path, monkeypatch, scenario, **options):
    if scenario == "prompt":
        hpc = importlib.import_module("services.run_hpc")
        case, captured = tmp_path / "Allrun-case", []
        case.mkdir()
        cluster = dict(cluster_name="Allrun-cluster", account_number="Allrun-account",
                       partition="Allrun-partition", nodes=2, ntasks_per_node=4, time_limit=3, memory=8)
        error = "Allrun failed in Allrun-case\nCheck the Allrun-partition account."
        previous = "#!/bin/bash\ncd /scratch/Allrun-case\n./Allrun\n"
        script = str(case / ".foamagent/solver-jobs/Allrun-attempt/run-stage.sh") if options["custom_script"] else "Allrun"
        generated = "#!/bin/bash\nprintf 'Allrun context preserved\\n'\n"
        kwargs = {"llm_service": SimpleNamespace(invoke=lambda user, system: captured.append((user, system)) or generated)}
        if options["custom_script"]:
            kwargs["execution_script"] = script
        result = (hpc.create_slurm_script_with_error_context(str(case), cluster, error, previous, **kwargs)
                  if options["retry"] else hpc.create_slurm_script(str(case), cluster, **kwargs))
        user, system = captured[0]
        for line in (f"Case directory: {case}", "Cluster: Allrun-cluster", "Account: Allrun-account", "Partition: Allrun-partition"):
            assert line + "\n" in user
        assert f"using the {script} script." in user
        assert f"Directory navigation and execution of the {script} script" in system
        if options["retry"]:
            assert f"Previous submission failed with error: {error}\n" in user
            assert f"Previous SLURM script that failed:\n```bash\n{previous}\n```\n" in user
        assert Path(result) == case / "submit_job.slurm" and Path(result).read_text() == generated.strip()
        return

    if scenario == "mesh_stop":
        from services.run_hpc import HpcWorkflowStop
        from langgraph.graph import END
        from router_func import route_after_meshing
        calls = []

        def stop(*args, **kwargs):
            calls.append(args)
            raise HpcWorkflowStop("hpc_job_state_unconfirmed", "job still active", {"job_id": "123"})

        monkeypatch.setattr(modules.meshing, "_native_mesh_runners", lambda _: {"utility_runner": stop, "python_runner": stop})
        monkeypatch.setattr(modules.mesh, "extract_boundary_names_from_requirements", lambda *a, **kw: [])
        llm = SimpleNamespace(invoke=lambda *a, **kw: modules.mesh.GMSHPythonCode(python_code="pass", mesh_type="2D", geometry_type="test"))
        _put(tmp_path, "source.msh", "source")
        state = _state(tmp_path, llm_service=llm, mesh_source=str(tmp_path / "source.msh"))
        backend = modules.meshing.imported_meshing if options["backend"] == "custom" else modules.meshing.gmsh_meshing
        result = modules.meshing.execute_meshing(state, backend)
        assert len(calls) == 1 and result["termination_reason"] == "hpc_job_state_unconfirmed"
        assert route_after_meshing(result) == END
        return

    c = _hpc_context(modules, tmp_path, monkeypatch)
    if scenario == "exit":
        code = options["exit_code"]
        c.script.write_text(f"exit {code or 0}\n")
        c.scheduler.execute = code is not None
        old = tmp_path / ".foamagent/solver-jobs/old"
        _put(old, "exit-code", "0")
        result = c.hpc.run_stage_hpc(c.state, str(c.script), "solver", 1)
        assert (result["returncode"] == 0) == (code == 0)
        assert result["job_id"] and Path(result["script"]).parent != old
        if code is None:
            assert "No exit result" in result["diagnostic"]
    elif scenario == "resources":
        c.state["requires_hpc"], c.state["config"].max_time_limit = True, 120
        c.state["cluster_info"], c.scheduler.queue_until = {"ntasks_per_node": 99}, 60
        _put(tmp_path, "system/decomposeParDict", "numberOfSubdomains 2;")
        runner = modules.meshing._native_mesh_runners(c.state)["utility_runner"]
        result = runner(["bash", "-c", "printf mesh-ran"], timeout=10)
        assert result.returncode == 0 and result.stdout == "mesh-ran" and c.scheduler.clock == 60
        for stage, count in (("mesh", 2), ("solver", 16)):
            if stage == "solver":
                _put(tmp_path, "system/decomposeParDict", "numberOfSubdomains 16;")
                assert c.hpc.run_stage_hpc(c.state, str(c.script), stage, 1)["returncode"] == 0
            job = c.state[stage + "_jobs"][-1]
            assert job["returncode"] == 0 and c.state["cluster_info"]["ntasks_per_node"] == count
            assert f"#SBATCH -n {count}\n" in Path(job["script"]).read_text()
        assert sum(a[0] == "sbatch" for a in c.scheduler.calls) == 2
    elif scenario == "budget":
        budget = options["budget"]
        c.state["loop_count"], c.state["config"].max_loop = 1, budget
        c.scheduler.submissions = ["rejected", "accepted"]
        if budget == 2:
            with pytest.raises(c.hpc.HpcWorkflowStop, match="shared repair budget"):
                c.hpc.run_stage_hpc(c.state, str(c.script), "solver", 1)
        else:
            assert c.hpc.run_stage_hpc(c.state, str(c.script), "solver", 1)["returncode"] == 0
            assert any("invalid partition" in p and "Previous SLURM script" in p for p in c.prompts)
        assert c.state["loop_count"] == 2
        assert sum(a[0] == "sbatch" for a in c.scheduler.calls) == budget - 1
    elif scenario == "uncertain":
        from langgraph.graph import END
        from router_func import route_after_runner
        unknown = options["failure"] == "unknown"
        c.scheduler.submissions, c.scheduler.queue_until, c.scheduler.confirm_cancel = ["unknown" if unknown else "accepted"], float("inf"), False
        result = c.node.hpc_runner_node(c.state)
        assert result["workflow_status"] == "failed"
        assert result["termination_reason"] == ("hpc_submission_unconfirmed" if unknown else "hpc_job_state_unconfirmed")
        assert route_after_runner(result) == END
        assert sum(a[0] == "sbatch" for a in c.scheduler.calls) == 1
        if not unknown:
            assert result["job_id"] and c.scheduler.clock == 61
            assert sum(a[0] == "scancel" for a in c.scheduler.calls) == 1
    elif scenario == "query_deadline":
        c.scheduler.query_seconds = 10
        status, ok, error = c.hpc.wait_for_job("123", max_wait_time=1)
        assert status == "TIMEOUT" and not ok and "Monitoring exceeded" in error and c.scheduler.clock == 1
    elif scenario == "fingerprint":
        change = options["change"]
        c.scheduler.execute = options["exit_code"] is not None
        c.script.write_text("printf 'specific failure' >&2\nexit 7\n")
        _put(tmp_path, "system/controlDict", "deltaT 0.1;\n")
        c.state.update(c.node.hpc_runner_node(c.state))
        errors = c.state["error_logs"]
        c.state.update(c.review.review_case(c.state))
        if change == "input":
            _put(tmp_path, "system/controlDict", "deltaT 0.01;\n")
        elif change == "submission":
            c.options.partition = "debug"
        c.state.update(c.node.hpc_runner_node(c.state))
        result = c.review.review_case(c.state)
        assert errors == c.state["error_logs"] and "attempt-" not in str(errors)
        assert (result.get("termination_reason") == "repair_made_no_progress") == (change is None)
        assert c.options.reviews == (1 if change is None else 2)
        jobs = c.state["solver_jobs"]
        assert jobs[0]["job_id"] != jobs[1]["job_id"] and jobs[0]["script"] != jobs[1]["script"]
    else:
        raise AssertionError(f"Unknown HPC scenario: {scenario}")



_IMPORTED_CASES = [
    ("imported_script_extraction", {"scenario": "script"}),
    ("imported_complete_case", {"scenario": "complete"}),
    *[(f"imported_route_mesh{int(mesh)}_writer{int(writer)}_hpc{int(hpc)}",
       {"scenario": "route", "mesh": mesh, "writer": writer, "hpc": hpc})
      for mesh, writer, hpc in [
          (False, False, False), (False, True, False), (True, False, False),
          (True, True, False), (True, True, True),
      ]],
    ("imported_solver_repair", {"scenario": "repair"}),
]


def _check_imported(modules, tmp_path, monkeypatch, scenario, **options):
    from models import ExistingCasePlan
    from router_func import route_after_solver_planner
    from services.plan import DEFAULT_IMPORTED_REQUIREMENT

    if scenario == "script":
        script, calls = "#!/bin/sh\nicoFoam\n", []

        def invoke(user, system, pydantic_obj=None):
            calls.append((user, system))
            return ExistingCasePlan(status="ready", solver_allrun=script)

        state = {
            "case_context": {"platform": "foundation-v10", "has_allrun": True,
                             "issues": [], "candidate_cases": []},
            "case_dir": str(tmp_path / "work"),
            "user_requirement": DEFAULT_IMPORTED_REQUIREMENT,
            "user_requirement_explicit": False,
            "llm_service": SimpleNamespace(invoke=invoke),
        }
        result = modules.plan.plan_imported_case(state)
        assert result["workflow_status"] == "running"
        assert not result["requires_input_writer"] and not result["requires_meshing"]
        assert len(calls) == 1 and "solver-only remainder" in calls[0][0]
        assert result["solver_allrun"] == script
        assert route_after_solver_planner({**state, **result, "case_origin": "imported"}) == "input_writer"
        return

    assert scenario in {"complete", "route", "repair"}
    repair = scenario == "repair"
    mesh = options.get("mesh", repair)
    writer = options.get("writer", False)
    hpc = options.get("hpc", False)
    runner_name = "hpc_runner" if hpc else "local_runner"
    visited, writer_calls = [], []

    def stage(name, update=None):
        def run(state):
            visited.append(name)
            return {"error_logs": [], **(update or {})}
        return run

    def write(state):
        visited.append("input_writer")
        writer_calls.append(dict(state))
        return {"error_logs": []}

    def run(state):
        visited.append(runner_name)
        if repair and visited.count(runner_name) == 1:
            return {"error_logs": ["solver error"]}
        return {"workflow_status": "success", "error_logs": []}

    nodes = {
        "preparation": stage("preparation", {
            "case_origin": "imported", "case_dir": str(tmp_path / "work"),
            "workflow_status": "planning",
            "case_context": {"platform": "foundation-v10", "has_allrun": True,
                             "issues": [], "candidate_cases": []},
        }),
        "meshing_planner": stage("meshing_planner"),
        "meshing": stage("meshing"),
        "solver_planner": stage("solver_planner", {
            "workflow_status": "running", "requires_meshing": mesh,
            "requires_input_writer": writer, "requires_hpc": hpc,
        }),
        "input_writer": write,
        "local_runner": run if not hpc else stage("local_runner", {"workflow_status": "success"}),
        "hpc_runner": run if hpc else stage("hpc_runner", {"workflow_status": "success"}),
        "reviewer": stage("reviewer", {"loop_count": 1, "input_writer_mode": "rewrite"}),
    }
    for name, function in nodes.items():
        monkeypatch.setattr(modules.main, name + "_node", function)
    config = SimpleNamespace(openfoam_target="foundation-v10", recursion_limit=20, max_loop=3)
    state = modules.main.initialize_state(
        DEFAULT_IMPORTED_REQUIREMENT, config, workflow_mode="imported_case",
        case_import_path=str(tmp_path / "source"), requires_visualization=False,
        user_requirement_explicit=False,
    )
    result = modules.main.create_foam_agent_graph().compile().invoke(
        state, config={"recursion_limit": 20},
    )
    expected = ["preparation", "meshing_planner", "meshing", "solver_planner",
                "input_writer", runner_name]
    if repair:
        expected += ["reviewer", "input_writer", runner_name]
    assert visited == expected
    assert result["workflow_status"] == "success"
    assert len(writer_calls) == 1 + repair
    assert writer_calls[0]["requires_input_writer"] is writer


_TRANSLATION_CASES = [
    ("translation_allmesh_legacy_esi", {"fork": "esi", "target": ""}),
    ("translation_allmesh_foundation", {"fork": "foundation", "target": ""}),
    ("translation_allmesh_native_esi", {"fork": "esi", "target": "esi-v2006"}),
]


def _check_translation(modules, tmp_path, monkeypatch, fork, target):
    from translation.esi_translator import convert_case_to_esi_if_needed

    case = tmp_path / "case"
    _put(case, "system/controlDict", "application simpleFoam;\n")
    _put(case, "constant/momentumTransport",
         "simulationType RAS;\nRAS { model kEpsilon; turbulence on; }\n")
    (case / "0").mkdir()
    script = b"#!/bin/sh\nset -e\ncat <<'EOF'\n$(literal) `literal`\nEOF\nblockMesh\n"
    allmesh = case / "Allmesh"
    allmesh.write_bytes(script)
    config = SimpleNamespace(
        openfoam_fork=fork, openfoam_target=target,
        esi_translation_rules_path=ROOT / "src/translation/esi_translation_rules.json",
    )
    for _ in range(2):
        convert_case_to_esi_if_needed(case, config)
        assert allmesh.read_bytes() == script
    if fork == "esi" and not target:
        assert "RASModel kEpsilon;" in (case / "constant/turbulenceProperties").read_text()


_FILE_CASES = [
    pytest.param(_check_default_runs_skip_occupied_names, {}, id="default_runs_skip_occupied_names"),
    pytest.param(_check_concurrent_default_runs_reserve_different_directories, {}, id="concurrent_default_runs_reserve_different_directories"),
    pytest.param(_check_explicit_output_keeps_existing_overwrite_rules, {}, id="explicit_output_keeps_existing_overwrite_rules"),
    pytest.param(_check_mesh_planner_uses_current_sources_and_early_hpc_routing, {}, id="mesh_planner_uses_current_sources_and_early_hpc_routing"),
    pytest.param(_check_custom_mesh_source_retries_switches_and_same_file, {}, id="custom_mesh_source_retries_switches_and_same_file"),
    pytest.param(_check_mesh_handoff_and_cleanup_preserve_required_inputs, {}, id="mesh_handoff_and_cleanup_preserve_required_inputs"),
    pytest.param(_check_pending_solver_tasks_survive_retries_once_with_distinct_edits, {}, id="pending_solver_tasks_survive_retries_once_with_distinct_edits"),
    pytest.param(_check_mcp_plan_writer_preserves_root_paths_and_task_metadata, {}, id="mcp_plan_writer_preserves_root_paths_and_task_metadata"),
    pytest.param(_check_preparation_allocates_independent_default_runs, {"start": 1, "overwrite": False}, id="output-default-1-false"),
    pytest.param(_check_preparation_allocates_independent_default_runs, {"start": 7, "overwrite": True}, id="output-default-7-true"),
    pytest.param(_check_default_runs_retain_output_path_checks, {"unsafe": "symlink_parent"}, id="output-unsafe-symlink_parent"),
    pytest.param(_check_default_runs_retain_output_path_checks, {"unsafe": "whitespace"}, id="output-unsafe-whitespace"),
    pytest.param(_check_default_runs_retain_output_path_checks, {"unsafe": "source_directory"}, id="output-unsafe-source_directory"),
    pytest.param(_check_default_runs_retain_output_path_checks, {"unsafe": "parent_file"}, id="output-unsafe-parent_file"),
    pytest.param(_check_final_reconstruction_requires_complete_sources_and_new_output, {"outcome": "missing_source"}, id="reconstruction-missing_source"),
    pytest.param(_check_final_reconstruction_requires_complete_sources_and_new_output, {"outcome": "failed"}, id="reconstruction-failed"),
    pytest.param(_check_final_reconstruction_requires_complete_sources_and_new_output, {"outcome": "incomplete"}, id="reconstruction-incomplete"),
    pytest.param(_check_final_reconstruction_requires_complete_sources_and_new_output, {"outcome": "success"}, id="reconstruction-success"),
    pytest.param(_check_explicit_allmesh_and_mixed_file_context, {"action": "create", "repair": False}, id="explicit-allmesh-create-first"),
    pytest.param(_check_explicit_allmesh_and_mixed_file_context, {"action": "create", "repair": True}, id="explicit-allmesh-create-repair"),
    pytest.param(_check_explicit_allmesh_and_mixed_file_context, {"action": "modify", "repair": False}, id="explicit-allmesh-modify-first"),
    pytest.param(_check_explicit_allmesh_and_mixed_file_context, {"action": "modify", "repair": True}, id="explicit-allmesh-modify-repair"),
    pytest.param(_check_mesh_retry_rotates_tool_logs_and_preserves_inputs, {"source_key": "mesh_source"}, id="mesh-log-preservation-mesh_source"),
    pytest.param(_check_mesh_retry_rotates_tool_logs_and_preserves_inputs, {"source_key": "custom_mesh_path"}, id="mesh-log-preservation-custom_mesh_path"),
    pytest.param(_check_explicit_solver_script_preserves_instructions_and_shell, {"action": "create", "mode": "sequential_dependency"}, id="explicit-solver-script-create-sequential_dependency"),
    pytest.param(_check_explicit_solver_script_preserves_instructions_and_shell, {"action": "create", "mode": "parallel_no_context"}, id="explicit-solver-script-create-parallel_no_context"),
    pytest.param(_check_explicit_solver_script_preserves_instructions_and_shell, {"action": "modify", "mode": "sequential_dependency"}, id="explicit-solver-script-modify-sequential_dependency"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": False, "folder": ""}, id="writer-sequential_dependency-root"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": False, "folder": "."}, id="writer-sequential_dependency-dot"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": False, "folder": None}, id="writer-sequential_dependency-automatic"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": False, "folder": "tools"}, id="writer-sequential_dependency-nested"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "parallel_no_context", "through_mcp": False, "folder": ""}, id="writer-parallel_no_context-root"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "parallel_no_context", "through_mcp": False, "folder": "."}, id="writer-parallel_no_context-dot"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "parallel_no_context", "through_mcp": False, "folder": None}, id="writer-parallel_no_context-automatic"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "parallel_no_context", "through_mcp": False, "folder": "tools"}, id="writer-parallel_no_context-nested"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": True, "folder": ""}, id="writer-mcp-root"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": True, "folder": "."}, id="writer-mcp-dot"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": True, "folder": None}, id="writer-mcp-automatic"),
    pytest.param(_check_writer_preserves_explicit_allrun_and_automatic_fallback, {"mode": "sequential_dependency", "through_mcp": True, "folder": "tools"}, id="writer-mcp-nested"),
]


@pytest.mark.parametrize("check, options", [
    *[pytest.param(_check_workflow, options, id=name) for name, options in _WORKFLOW_CASES],
    *[pytest.param(_check_recursion, options, id=name) for name, options in _RECURSION_CASES],
    *_FILE_CASES,
    *[pytest.param(_check_hpc, options, id=name) for name, options in _HPC_CASES],
    *[pytest.param(_check_imported, options, id=name) for name, options in _IMPORTED_CASES],
    *[pytest.param(_check_translation, options, id=name) for name, options in _TRANSLATION_CASES],
])
def test_pr44(check, options, modules, tmp_path, monkeypatch):
    check(modules, tmp_path, monkeypatch, **options)
