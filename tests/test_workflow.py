"""Stage orchestration, clean-input handoff, mesh evidence, and review."""

import asyncio
import gzip
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest


def _put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _poly_mesh(root):
    for name in ("points", "faces", "owner", "neighbour"):
        _put(root, f"constant/polyMesh/{name}", "0\n()\n")
    _put(root, "constant/polyMesh/boundary",
         "1\n(wall {type wall; nFaces 1; startFace 0;})\n")


def _operations(*commands, stage="solver"):
    return [dict(command=c, stage=stage, purpose="Declared test operation", executor="script",
                 required_inputs=[], generated_inputs=[]) for c in commands]


def _state(case, **updates):
    from config import Config
    state = dict(
        case_dir=str(case), config=Config(openfoam_target="foundation-v10", max_loop=3),
        user_requirement="Run the supplied case", llm_service=None, loop_count=0,
        mesh_type="standard_mesh", mesh_source=None, mesh_script_action="none",
        mesh_operations=[], pending_solver_operations=[], solver_operations=_operations("icoFoam"),
        mesh_review_history=[], solver_review_history=[], solver_script_dirty=False,
        mesh_reference=dict(case=None, content="", script="", reason="fixture"),
        mesh_boundary_plan=[], dir_structure={}, foamfiles=None, allrun_reference="",
        cross_stage_request=None,
    )
    state.update(updates)
    return state


def _solver_state(case, llm=None):
    from utils import load_case_files
    state = _state(case)
    state.update(case_origin="imported", llm_service=llm, solver_planned=True,
                 input_writer_mode="rewrite", requires_input_writer=True, requires_hpc=False,
                 requires_visualization=False, review_analysis="Apply the declared correction",
                 rewrite_plan=None, tutorial_reference="", case_solver="icoFoam")
    return {**state, **load_case_files(state)}


def test_standard_mesh(modules, tmp_path, monkeypatch):
    """A retained Allmesh runs only after its declared startup input is written."""
    _poly_mesh(tmp_path)
    script = "#!/bin/bash\nset -e\ncat system/startup > mesh-result\n"
    _put(tmp_path, "Allmesh", script)
    operations = _operations(script, stage="mesh")
    operations[0]["required_inputs"] = [dict(path="system/startup", evidence="Allmesh reads it")]
    state = _state(tmp_path, mesh_script_action="run_existing", mesh_operations=operations)

    def execute(script, out, err, case, *args, **kwargs):
        result = subprocess.run(["bash", script], cwd=case, capture_output=True, text=True)
        Path(out).write_text(result.stdout)
        Path(err).write_text(result.stderr)
        return {"returncode": result.returncode, "timed_out": False}

    monkeypatch.setattr(modules.meshing, "run_command", execute)
    first = modules.meshing.execute_meshing(state, modules.meshing.standard_meshing)
    assert first["error_logs"] and not (tmp_path / "mesh-result").exists()
    state.update(first)
    state.update(llm_service=SimpleNamespace(invoke=lambda *a, **kw: "supplied startup\n"),
                 mesh_subtasks=[dict(folder_name="system", file_name="startup", stage="mesh",
                                     action="create", changes="Create the declared startup input")])
    result = modules.meshing.execute_meshing(state, modules.meshing.standard_meshing)
    assert not result["error_logs"]
    assert (tmp_path / "Allmesh").read_text() == script
    assert (tmp_path / "mesh-result").read_text() == "supplied startup\n"


def test_gmsh_mesh(modules, tmp_path):
    """An explicitly prepared Gmsh program is followed by conversion and checking."""
    script = _put(tmp_path, "generate_mesh.py", "# supplied generation program\n")
    calls = []

    def generate(argv, **kwargs):
        assert Path(argv[1]).read_text() == script.read_text()
        _put(tmp_path, "geometry.msh", "generated mesh")
        return subprocess.CompletedProcess(argv, 0, "generated", "")

    def utility(argv, **kwargs):
        calls.append(argv[0])
        _poly_mesh(tmp_path)
        return subprocess.CompletedProcess(argv, 0, "Mesh OK.", "")

    result = modules.mesh.handle_gmsh_mesh(
        "Use the supplied program", str(tmp_path), timeout=10, script_prepared=True,
        python_runner=generate, utility_runner=utility,
        llm_service=SimpleNamespace(invoke=lambda *a, **kw: pytest.fail("Unexpected model call")))
    assert not result["error_logs"] and result["mesh_info"]["check_mesh_passed"]
    assert calls == ["gmshToFoam", "checkMesh"]


def test_custom_mesh_repair(modules, tmp_path, monkeypatch):
    """Retry converts the repaired working mesh without recopying the original."""
    source = _put(tmp_path, "input/source.msh", "original mesh")
    converted = []

    def utility(argv, **kwargs):
        if argv[0] == "gmshToFoam":
            converted.append((tmp_path / "geometry.msh").read_text())
            _poly_mesh(tmp_path)
        summary = "Failed 1 mesh checks." if len(converted) == 1 else "Mesh OK."
        return subprocess.CompletedProcess(argv, 0, summary, "")

    monkeypatch.setattr(modules.mesh, "run_openfoam_utility", utility)
    first = modules.mesh.copy_custom_mesh(str(source), str(tmp_path), timeout=10)
    assert first["error_logs"]
    _put(tmp_path, "geometry.msh", "repaired working mesh")
    result = modules.mesh.copy_custom_mesh(str(source), str(tmp_path), timeout=10, retry=True)
    assert not result["error_logs"]
    assert converted == ["original mesh", "repaired working mesh"]
    assert source.read_text() == "original mesh"


def _cube(root, region, offset):
    """A valid one-cell mesh for the real VTK reader."""
    points = [(offset + x, y, z) for x, y, z in
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]
    bodies = {
        "points": ("vectorField", "8\n(\n" + "\n".join(f"({x} {y} {z})" for x, y, z in points) + "\n)"),
        "faces": ("faceList", "6\n(4(0 3 2 1) 4(4 5 6 7) 4(0 1 5 4) 4(1 2 6 5) 4(2 3 7 6) 4(3 0 4 7))"),
        "owner": ("labelList", "6\n(0 0 0 0 0 0)"),
        "neighbour": ("labelList", "0\n()"),
        "boundary": ("polyBoundaryMesh", "1\n(walls {type wall; nFaces 6; startFace 0;})"),
    }
    for name, (kind, body) in bodies.items():
        _put(root, f"constant/{region}/polyMesh/{name}",
             f"FoamFile {{version 2.0; format ascii; class {kind}; object {name};}}\n{body}\n")
    _put(root, "system/controlDict", "application checkMesh; startTime 0; endTime 1; deltaT 1;\n")
    _put(root, f"system/{region}/fvSolution", "solvers {}\n")
    _put(root, f"system/{region}/fvSchemes", "ddtSchemes {default steadyState;}\n")


@pytest.mark.parametrize('target,key', [
    ('foundation-v10', 'regions'),
    ('esi-v2006', '"reg.*"'),
])
def test_multiregion_handoff(modules, tmp_path, monkeypatch, target, key):
    """Region-only meshes retain actual regional patches and measured bounds."""
    _cube(tmp_path, "fluid", 0)
    _cube(tmp_path, "solid", 2)
    _put(tmp_path, "constant/regionProperties", "regions (fluid (fluid) solid (solid));\n")
    monkeypatch.setattr(modules.meshing, "read_patch_bounds", modules.read_patch_bounds)
    queries = []
    def native(argv, **kwargs):
        queries.append(argv)
        output = ('Using #codeStream at line 2 in file "constant/regionProperties"\n'
                  'Using #codeStream with "dynamicCode/library.so"\n'
                  'codeStream : dictionary:"constant/regionProperties" master-only-reading:0\n'
                  + key + '\n2\n(\nfluid 1(fluid)\nsolid 1(solid)\n);\n')
        return subprocess.CompletedProcess(argv, 0, output, '')
    monkeypatch.setattr(modules.meshing, "run_openfoam_utility", native)
    state = _state(tmp_path)
    state['config'].openfoam_target = target
    result = modules.meshing.execute_meshing(state, lambda _: {"error_logs": []})
    assert not result["error_logs"] and not (tmp_path / "constant/polyMesh").exists()
    assert set(result["mesh_info"]["regions"]) == {"fluid", "solid"}
    assert result["mesh_info"]["patch_bounds"] == {
        "fluid/walls": [0, 1, 0, 1, 0, 1], "solid/walls": [2, 3, 0, 1, 0, 1]}
    assert queries == [['foamDictionary', *(['-expand'] if target == 'foundation-v10' else []),
                        'constant/regionProperties', '-entry', 'regions']]


@pytest.mark.parametrize('partition', ['processor0', 'processors1', 'processors1_0-0'])
def test_mesh_reconstruction(modules, tmp_path, monkeypatch, partition):
    """Successful reconstruction replaces the mesh; a later native failure rolls back."""
    _poly_mesh(tmp_path)
    _poly_mesh(tmp_path / partition)
    initial = _put(tmp_path, "0/U", "preserve initial field")
    fail = False

    def reconstruct(argv, **kwargs):
        _poly_mesh(tmp_path)
        if fail:
            raise subprocess.CalledProcessError(1, argv, stderr="partition read failed")
        _put(tmp_path, "constant/polyMesh/points", "reconstructed mesh")
        return subprocess.CompletedProcess(argv, 0, "reconstructed", "")

    monkeypatch.setattr(modules.meshing, "run_openfoam_utility", reconstruct)
    previous_mesh = {"check_mesh_passed": True}
    state = _state(tmp_path, mesh_info=previous_mesh, final_reconstruction=dict(
        sources=[f"{partition}/constant/polyMesh"], argv=["reconstructParMesh", "-constant"]))
    first = modules.meshing.execute_meshing(state, lambda _: {"error_logs": []})
    assert not first["error_logs"]
    assert first["mesh_info"]["check_mesh_passed"] is None
    assert previous_mesh["check_mesh_passed"] is True
    fail = True
    state.update(first)
    state['final_reconstruction'] = dict(sources=[f"{partition}/constant/polyMesh"], argv=["reconstructParMesh", "-constant"])
    second = modules.meshing.execute_meshing(state, lambda _: {"error_logs": []})
    assert second["error_logs"]
    assert (tmp_path / "constant/polyMesh/points").read_text() == "reconstructed mesh"
    assert initial.read_text() == "preserve initial field"


def test_pending_physical_operations(modules, tmp_path, monkeypatch):
    """Unexecuted initialization survives mesh handoff until the writer emits Allrun."""
    from nodes.input_writer_node import input_writer_node
    from router_func import route_after_meshing

    _poly_mesh(tmp_path)
    _put(tmp_path, "system/controlDict", "application icoFoam;\n")
    _put(tmp_path, "Allrun", "icoFoam\n")
    operations = _operations("setFields", "icoFoam")
    llm = SimpleNamespace(invoke=lambda *a, **kw: modules.meshing.MeshRepairImpact(
        replan_solver=False, reason="Keep pending physical preparation"))
    state = _solver_state(tmp_path, llm)
    state.update(case_origin="generated", case_info="supplied physics", mesh_type="custom_mesh",
                 pending_solver_operations=operations, solver_operations=operations)
    state.update(modules.meshing.execute_meshing(state, modules.meshing.custom_meshing))
    assert route_after_meshing(state) == "solver_planner"
    assert state["pending_solver_operations"] == operations
    planner = modules.solver_planner_node
    monkeypatch.setattr(planner, "prepare_solver_context", lambda _: dict(
        dir_counts_str="", dir_structure_reference="", tutorial_reference=""))
    monkeypatch.setattr(planner, "_build_advice", lambda *a: modules.plan.SimilarCaseAdviceModel())

    def plan(requirement, *args, **kwargs):
        assert "setFields" in requirement
        return modules.plan.OpenFOAMPlanModel(operations=operations, subtasks=[])

    monkeypatch.setattr(planner, "decompose_to_subtasks", plan)
    state.update(planner.solver_planner_node(state))
    assert state["pending_solver_operations"] == operations
    state.update(input_writer_node(state))
    assert not state["error_logs"] and state["pending_solver_operations"] == []
    assert (tmp_path / "Allrun").read_text().count("setFields") == 1


def test_solver_completion(modules, tmp_path, monkeypatch):
    """Quoted application works, while completion without a new result is rejected."""
    from services import solver_execution

    _put(tmp_path, "system/controlDict", 'application "icoFoam";\n')
    _put(tmp_path, "0/U", "FoamFile {class volVectorField;}\n")
    before = solver_execution.snapshot_result_fields(tmp_path)
    _put(tmp_path, "log.solve", "Exec : icoFoam\nEnd\n")
    monkeypatch.setattr(solver_execution, "run_openfoam_utility", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, 'application "icoFoam";\n', ""))
    assert solver_execution.check_solver_execution(str(tmp_path), fields_before=before, logs_before={})
    _put(tmp_path, "1/U", "FoamFile {class volVectorField;}\n")
    assert not solver_execution.check_solver_execution(str(tmp_path), fields_before=before, logs_before={})


def test_repair_progress(modules, tmp_path, monkeypatch):
    """Correcting an invalid operation stage allows execution without a file edit."""
    from nodes.input_writer_node import input_writer_node
    from nodes.local_runner_node import local_runner_node
    from services import review

    script = _put(tmp_path, "Allrun", "icoFoam\n")
    _put(tmp_path, "system/controlDict", "application icoFoam;\n")
    replacement = _operations("icoFoam\n")
    llm = SimpleNamespace(invoke=lambda *a: json.dumps(dict(
        action="repair_solver", target_files=[], operations=replacement, operation_request=None,
        diagnosis="Correct operation ownership", evidence=["preflight error"], reason="Fix stage")))
    state = _solver_state(tmp_path, llm)
    state["solver_operations"] = _operations("icoFoam\n", stage="mesh")
    calls = []
    monkeypatch.setattr("nodes.local_runner_node.run_allrun_and_collect_errors",
                        lambda *a, **kw: calls.append(a) or [])
    monkeypatch.setattr(review, "log_review", lambda *a: None)
    state.update(input_writer_node(state))
    assert state["error_logs"] and not calls
    state.update(review.review_solver(state))
    state.update(input_writer_node(state))
    assert not state["error_logs"] and not state.get("termination_reason")
    state.update(local_runner_node(state))
    assert state["workflow_status"] == "success" and len(calls) == 1
    assert script.read_text() == "icoFoam\n"


def test_mcp_run_contract(modules, tmp_path, monkeypatch):
    """Standalone MCP run forwards its request and retains the supplied script."""
    server = importlib.import_module("src.mcp.fastmcp_server")
    script = _put(tmp_path, "Allrun", "#!/bin/bash\ncustomSolver --user-option\n")
    calls = []
    monkeypatch.setattr(server, "run_allrun_and_collect_errors", lambda **kw: calls.append(kw) or [])

    async def log(*args, **kwargs):
        pass

    run = getattr(server.run, "fn", server.run)
    response = asyncio.run(run(server.RunSimulationRequest(case_dir=str(tmp_path), timeout=17),
                               SimpleNamespace(info=log, error=log)))
    assert response.status == "success" and not response.errors
    assert calls == [dict(case_dir=str(tmp_path), timeout=17, max_retries=3,
                          openfoam_target=server.runtime_openfoam_target(server.global_config))]
    assert script.read_text() == "#!/bin/bash\ncustomSolver --user-option\n"


def test_visualization_failure_status(modules, tmp_path, monkeypatch):
    """A rendering failure preserves simulation success as partial_success."""
    from config import Config

    node = importlib.import_module("nodes.visualization_node")
    monkeypatch.setattr(node, "visualize_case", lambda *a, **kw:
                        {"pyvista_visualization": {"success": False, "error": "render failed"}})
    result = node.visualization_node(dict(case_dir=str(tmp_path), config=Config(), workflow_status="success"))
    assert result["workflow_status"] == "partial_success"
    assert result["termination_reason"] == "visualization_failed"
    assert modules.main.workflow_exit_code(result) != 0


def test_output_ownership(modules, tmp_path):
    """Replacing an output requires explicit overwrite and the ownership marker."""
    from config import Config
    from services.output_safety import OUTPUT_MARKER_NAME, OutputDirectorySafetyError

    case = tmp_path / "case"
    config = Config(case_dir=str(case))
    modules.preparation._prepare_generated_case_dir(config, None)
    original = _put(case, "0/U", "initial field")
    with pytest.raises(FileExistsError):
        modules.preparation._prepare_generated_case_dir(config, None)
    assert original.read_text() == "initial field"
    config.overwrite_case_dir = True
    modules.preparation._prepare_generated_case_dir(config, None)
    assert not original.exists()
    (case / OUTPUT_MARKER_NAME).unlink()
    valuable = _put(case, "valuable-input", "preserve me")
    with pytest.raises(OutputDirectorySafetyError, match="unowned"):
        modules.preparation._prepare_generated_case_dir(config, None)
    assert valuable.read_text() == "preserve me"


@pytest.mark.parametrize('failure,compressed', [
    ('native', False),
    ('incomplete', True),
    ('commit', False),
    ('hpc_unconfirmed', False),
    ('none', True),
])
def test_regional_reconstruction_rolls_back_every_mesh_and_keeps_retry_sources(
        modules, tmp_path, monkeypatch, failure, compressed):
    import gzip
    from services.case_workspace import CaseWorkspace
    from services.run_hpc import HpcWorkflowStop
    for root in (tmp_path, tmp_path / 'processor0'):
        for region in ('fluid', 'solid'):
            _cube(root, region, 0)
    _put(tmp_path, 'constant/regionProperties', 'regions (fluid (fluid) solid (solid));\n')
    initial = _put(tmp_path, '0/U', 'original initial field')
    if compressed:
        for directory in tmp_path.rglob('polyMesh'):
            for path in directory.iterdir():
                path.with_name(path.name + '.gz').write_bytes(gzip.compress(path.read_bytes()))
                path.unlink()
    workspace = CaseWorkspace.create(str(tmp_path), str(tmp_path / '.foamagent'))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*')
              if p.is_file() and '.foamagent' not in p.parts and 'polyMesh' in p.parts}
    monkeypatch.setattr(modules.meshing, 'read_patch_bounds', modules.read_patch_bounds)

    def native(argv, **kwargs):
        if argv[0] == 'foamDictionary':
            return subprocess.CompletedProcess(argv, 0,
                '' if '-includes' in argv else 'regions (fluid (fluid) solid (solid));\n', '')
        _cube(tmp_path, 'fluid', 10)
        if failure.startswith('hpc_'):
            raise HpcWorkflowStop(
                'hpc_job_state_unconfirmed' if failure == 'hpc_unconfirmed' else 'hpc_monitoring_timeout',
                'Reconstruction stop could not be confirmed' if failure == 'hpc_unconfirmed' else 'Reconstruction stop confirmed',
                {'job_id': '123'})
        if failure == 'native':
            return subprocess.CompletedProcess(argv, 1, '', 'reconstruction failed')
        _cube(tmp_path, 'solid', 10)
        if failure == 'incomplete':
            (tmp_path / 'constant/solid/polyMesh/owner').unlink()
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(modules.meshing, 'run_openfoam_utility', native)
    rename = Path.rename
    failed = False

    def fail_commit(path, target):
        nonlocal failed
        if failure == 'commit' and not failed and Path(target) == workspace.source_dir / 'constant/solid/polyMesh':
            failed = True
            raise OSError('second-region handoff failed')
        return rename(path, target)

    monkeypatch.setattr(Path, 'rename', fail_commit)
    previous_mesh = {'regions': {region: {'poly_mesh_path': str(tmp_path / 'constant' / region / 'polyMesh')}
                                 for region in ('fluid', 'solid')},
                     'patch_bounds': {'fluid/walls': [0, 1, 0, 1, 0, 1]}, 'check_mesh_passed': True}
    pending = {'file': '0/U', 'stage': 'solver', 'changes': 'Retain the approved physical correction'}
    state = _state(tmp_path, mesh_info=previous_mesh, rewrite_plan={'target_files': [pending]},
        case_input_source_dir=str(workspace.source_dir), final_reconstruction=dict(
        sources=[f'processor0/constant/{r}/polyMesh' for r in ('fluid', 'solid')],
        argv=['reconstructParMesh', '-constant', '-allRegions']))
    result = modules.meshing.execute_meshing(state, lambda _: {'error_logs': []})
    assert previous_mesh['check_mesh_passed'] is True
    assert {**state, **result}['rewrite_plan']['target_files'] == [pending]
    if failure == 'hpc_unconfirmed':
        assert result['termination_reason'] == 'hpc_job_state_unconfirmed'
        reconstruction = result['mesh_execution']['final_reconstruction']
        assert reconstruction['status'] == 'state_unconfirmed'
        assert all(Path(path).is_dir() for path in reconstruction['backup_paths'].values())
        assert (tmp_path / 'constant/fluid/polyMesh/points').is_file()
        assert not (tmp_path / 'constant/solid/polyMesh').exists()
        assert 'mesh_info' not in result
        assert initial.read_text() == 'original initial field'
        return
    if failure == 'none':
        assert not result['error_logs']
        assert result['mesh_execution']['final_reconstruction']['status'] == 'complete'
        assert result['final_reconstruction'] is None
        assert (workspace.source_dir / '.foamagent/mesh-handoff').exists()
        workspace.restore_inputs()
        assert initial.read_text() == 'original initial field'
        assert '10' in (tmp_path / 'constant/fluid/polyMesh/points').read_text()
        return
    assert result['error_logs']
    assert result['mesh_execution']['final_reconstruction']['status'] == 'rolled_back'
    assert result['mesh_info'] is None and {**state, **result}['mesh_info'] is None
    if failure == 'hpc_confirmed':
        assert result['termination_reason'] == 'hpc_monitoring_timeout'
    assert all((tmp_path / path).read_bytes() == content for path, content in before.items())
    assert initial.read_text() == 'original initial field'
    assert not (workspace.source_dir / '.foamagent/mesh-handoff').exists()


def mesh(root, relative="constant/polyMesh", value="1000", compressed=False):
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        data = value.encode()
        (directory / (name + (".gz" if compressed else ""))).write_bytes(
            gzip.compress(data) if compressed else data)
    return directory


@pytest.mark.parametrize('compressed,regions', [
    (False, False),
    (True, True),
])
def test_mesh_handoff_commits_only_mesh_and_preserves_clean_geometry(tmp_path, compressed, regions):
    from services.case_workspace import CaseWorkspace
    geometry = tmp_path / "geometry.stl"
    geometry.write_text("1000")
    initial = tmp_path / "0/U"
    initial.parent.mkdir()
    initial.write_text("clean field template")
    paths = {name: mesh(tmp_path, f"constant/{name}/polyMesh", compressed=compressed)
             for name in ("fluid", "solid")} if regions else {"": mesh(tmp_path, compressed=compressed)}
    if regions:
        (tmp_path / "constant/regionProperties").write_text("regions (fluid (fluid) solid (solid));")
    workspace = CaseWorkspace.create(str(tmp_path), str(tmp_path / ".foamagent"))
    for _ in range(2):
        workspace.restore_inputs()
        assert geometry.read_text() == "1000"
        assert initial.read_text() == "clean field template"
        geometry.write_text("1")
        initial.write_text("runtime-expanded field")
        for name, directory in paths.items():
            mesh(tmp_path, directory.relative_to(tmp_path).as_posix(), value="1", compressed=compressed)
        workspace.commit_mesh(paths)
    workspace.restore_inputs()
    assert geometry.read_text() == "1000" and initial.read_text() == "clean field template"
    for directory in paths.values():
        point = directory / ("points.gz" if compressed else "points")
        assert (gzip.decompress(point.read_bytes()).decode() if compressed else point.read_text()) == "1"


@pytest.mark.parametrize("stage", ["mesh", "solver"])
def test_careful_initial_reviews_use_stage_specific_model_doubles(tmp_path, stage):
    from config import Config
    from services.static_review import StaticReviewResult, review_stage_inputs

    (tmp_path / "Allmesh").write_text("blockMesh\n")
    (tmp_path / "Allrun").write_text("icoFoam\n")
    (tmp_path / 'postProcessing').mkdir()
    (tmp_path / 'postProcessing/inlet-profile.dat').write_text('declared inlet data')
    operation = dict(stage=stage, purpose='Use supplied inlet profile', executor='script',
                     command='blockMesh' if stage == 'mesh' else 'icoFoam', generated_inputs=[],
                     required_inputs=[dict(path='postProcessing/inlet-profile.dat', evidence='Supplied input')])
    requests = []

    def invoke(payload, prompt, *, pydantic_obj):
        requests.append(json.loads(payload))
        assert pydantic_obj is StaticReviewResult
        return StaticReviewResult(summary="No demonstrated defect", findings=[])

    state = dict(case_dir=str(tmp_path), config=Config(), user_requirement="Run supplied inputs",
                 llm_service=SimpleNamespace(invoke=invoke), mesh_reference={},
                 mesh_operations=[operation] if stage == 'mesh' else [],
                 solver_operations=[operation] if stage == 'solver' else [], loop_count=0)
    result = review_stage_inputs(state, stage)
    assert result["static_review_complete"] == {stage: True}
    assert requests[0]["stage_context"]["stage"] == stage
    assert any(f['folder_name'] == 'postProcessing' and f['content'] == 'declared inlet data'
               for f in requests[0]['current_case_files'])
    assert not review_stage_inputs({**state, **result}, stage)
    assert len(requests) == 1


@pytest.mark.parametrize('outputs,filename', [
    (['plot.png', './plot.png'], 'visualization.py'),
    (['plot.png', '../outside.png'], 'visualization.py'),
])
def test_all_output_paths_are_checked_before_any_mutation(tmp_path, outputs, filename):
    from services.visualization import run_pyvista_script

    case = tmp_path / 'case'
    case.mkdir()
    image = case / 'plot.png'
    image.write_bytes(b'keep previous image')
    script = case / 'visualization.py'
    script.write_text('# keep previous script\n')
    (case / 'directory.png').mkdir()
    external = tmp_path / 'external'
    external.mkdir()
    (case / 'linked').symlink_to(external, target_is_directory=True)
    success, _, errors = run_pyvista_script(str(case), 'raise AssertionError("must not execute")',
                                           filename=filename, expected_pngs=outputs)
    assert not success and errors
    assert image.read_bytes() == b'keep previous image'
    assert script.read_text() == '# keep previous script\n'
    assert not (tmp_path / 'outside.py').exists()
    assert not list(external.iterdir())


@pytest.mark.parametrize('duplicate', [False, True])
def test_visualization_counts_only_unique_verified_images(tmp_path, duplicate):
    from services.visualization import visualize_case, VisualizationScript, VisualizationReview

    names = ['plot.png', './plot.png' if duplicate else 'second.png']
    proposal = VisualizationScript(
        script='from PIL import Image\n' + '\n'.join(
            f'Image.new("RGB", (8, 6)).save({name!r})' for name in names),
        plots=[dict(output_png=name, field_name='U', time_step='latest', description='Requested view')
               for name in names],
    )
    reviews = []

    def invoke(payload, prompt, *, pydantic_obj):
        if pydantic_obj is VisualizationScript:
            return proposal
        reviews.append(payload)
        return VisualizationReview(meets_requirements=True, feedback='Both requested views exist')

    result = visualize_case(str(tmp_path), 'Two velocity views', max_loop=1,
                            llm_service=SimpleNamespace(invoke=invoke))
    if duplicate:
        assert not result['pyvista_visualization']['success']
        assert not reviews
        assert not (tmp_path / 'visualization.py').exists()
    else:
        assert result['pyvista_visualization']['success']
        assert result['visualization_summary']['total_plots_generated'] == 2
        assert len(set(result['plot_outputs'])) == 2
        for output in result['plot_outputs']:
            with Image.open(output) as image:
                assert image.format == 'PNG' and image.size == (8, 6)
