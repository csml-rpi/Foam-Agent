"""Exercise meshing, solver handoff, review, execution, and the MCP run entry."""

import asyncio
import importlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from config import Config
from main import initialize_state
from services import mesh, meshing


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    import utils

    def forbidden(*args, **kwargs):
        pytest.fail("Feature tests must supply model responses without network access")

    monkeypatch.setattr(utils.LLMService, "invoke", forbidden)
    monkeypatch.setattr(utils, "get_embedding_model", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def mesh_services(monkeypatch):
    monkeypatch.setattr(meshing, "read_patch_bounds", lambda _: {"wall": [0, 1, 0, 1, 0, 1]})
    monkeypatch.setattr(meshing, "retrieve_mesh_reference", lambda *a: dict(
        case=None, content="", script="", reason="Supplied test mesh"))
    return SimpleNamespace(mesh=mesh, meshing=meshing)


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
    state = initialize_state("Run the supplied case", Config(execution_mode="fast", max_loop=3))
    state.update(
        case_dir=str(case), mesh_type="standard_mesh", mesh_source=None, mesh_script_action="none",
        solver_operations=_operations("icoFoam"), mesh_boundary_plan=[], dir_structure={},
        mesh_reference=dict(case=None, content="", script="", reason="Supplied test mesh"),
        allrun_reference="")
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


def test_standard_mesh(mesh_services, tmp_path, monkeypatch):
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

    monkeypatch.setattr(mesh_services.meshing, "run_command", execute)
    first = mesh_services.meshing.execute_meshing(state, mesh_services.meshing.standard_meshing)
    assert first["error_logs"] and not (tmp_path / "mesh-result").exists()
    state.update(first)
    state.update(llm_service=SimpleNamespace(invoke=lambda *a, **kw: "supplied startup\n"),
                 mesh_subtasks=[dict(folder_name="system", file_name="startup", stage="mesh",
                                     action="create", changes="Create the declared startup input")])
    result = mesh_services.meshing.execute_meshing(state, mesh_services.meshing.standard_meshing)
    assert not result["error_logs"]
    assert (tmp_path / "Allmesh").read_text() == script
    assert (tmp_path / "mesh-result").read_text() == "supplied startup\n"


def test_gmsh_mesh(mesh_services, tmp_path):
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

    result = mesh_services.mesh.handle_gmsh_mesh(
        "Use the supplied program", str(tmp_path), timeout=10, script_prepared=True,
        python_runner=generate, utility_runner=utility,
        llm_service=SimpleNamespace(invoke=lambda *a, **kw: pytest.fail("Unexpected model call")))
    assert not result["error_logs"] and result["mesh_info"]["check_mesh_passed"]
    assert calls == ["gmshToFoam", "checkMesh"]


def test_custom_mesh_repair(mesh_services, tmp_path, monkeypatch):
    """Retry converts the repaired working mesh without recopying the original."""
    source = _put(tmp_path, "input/source.msh", "original mesh")
    converted = []

    def utility(argv, **kwargs):
        if argv[0] == "gmshToFoam":
            converted.append((tmp_path / "geometry.msh").read_text())
            _poly_mesh(tmp_path)
        summary = "Failed 1 mesh checks." if len(converted) == 1 else "Mesh OK."
        return subprocess.CompletedProcess(argv, 0, summary, "")

    monkeypatch.setattr(mesh_services.mesh, "run_openfoam_utility", utility)
    first = mesh_services.mesh.copy_custom_mesh(str(source), str(tmp_path), timeout=10)
    assert first["error_logs"]
    _put(tmp_path, "geometry.msh", "repaired working mesh")
    result = mesh_services.mesh.copy_custom_mesh(str(source), str(tmp_path), timeout=10, retry=True)
    assert not result["error_logs"]
    assert converted == ["original mesh", "repaired working mesh"]
    assert source.read_text() == "original mesh"


def test_pending_physical_operations(mesh_services, tmp_path, monkeypatch):
    """Unexecuted initialization survives mesh handoff until the writer emits Allrun."""
    from nodes.input_writer_node import input_writer_node
    from router_func import route_after_meshing
    from services import plan as plans

    _poly_mesh(tmp_path)
    _put(tmp_path, "system/controlDict", "application icoFoam;\n")
    _put(tmp_path, "Allrun", "icoFoam\n")
    operations = _operations("setFields", "icoFoam")
    llm = SimpleNamespace(invoke=lambda *a, **kw: mesh_services.meshing.MeshRepairImpact(
        replan_solver=False, reason="Keep pending physical preparation"))
    state = _solver_state(tmp_path, llm)
    state.update(case_origin="generated", case_info="supplied physics", mesh_type="custom_mesh",
                 pending_solver_operations=operations, solver_operations=operations)
    state.update(mesh_services.meshing.execute_meshing(state, mesh_services.meshing.custom_meshing))
    assert route_after_meshing(state) == "solver_planner"
    assert state["pending_solver_operations"] == operations
    planner = importlib.import_module('nodes.solver_planner_node')
    monkeypatch.setattr(planner, "prepare_solver_context", lambda _: dict(
        dir_counts_str="", dir_structure_reference="", tutorial_reference=""))
    monkeypatch.setattr(planner, "_build_advice", lambda *a: plans.SimilarCaseAdviceModel())

    def plan(requirement, *args, **kwargs):
        assert "setFields" in requirement
        return plans.OpenFOAMPlanModel(operations=operations, subtasks=[])

    monkeypatch.setattr(planner, "decompose_to_subtasks", plan)
    state.update(planner.solver_planner_node(state))
    assert state["pending_solver_operations"] == operations
    state.update(input_writer_node(state))
    assert not state["error_logs"] and state["pending_solver_operations"] == []
    assert (tmp_path / "Allrun").read_text().count("setFields") == 1


@pytest.mark.parametrize("stage", ["mesh", "solver"])
def test_initial_static_review_uses_current_stage_and_declared_inputs(tmp_path, stage):
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


@pytest.mark.parametrize("already_exists", [False, True], ids=["new-input", "unchanged-input"])
def test_mixed_creation_and_rewrite_stop_only_when_no_inputs_change(tmp_path, already_exists):
    from config import Config
    from main import initialize_state
    from nodes.input_writer_node import input_writer_node
    from router_func import route_after_input_writer
    from services.case_workspace import CaseWorkspace
    from services.stage_plan import render_stage_script
    from utils import FoamPydantic

    _put(tmp_path, "system/controlDict", "application icoFoam;\n")
    _put(tmp_path, "system/B", "corrected B")
    if already_exists:
        _put(tmp_path, "system/A", "created A")
    operations = _operations("true")
    _put(tmp_path, "Allrun", render_stage_script(operations))
    workspace = CaseWorkspace.create(str(tmp_path), str(tmp_path / ".foamagent"))

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj is FoamPydantic:
            return FoamPydantic(list_foamfile=[
                dict(folder_name="system", file_name="B", content="corrected B")])
        return "created A"

    task = dict(folder_name="system", file_name="A", stage="solver", action="create",
                changes="Supply A")
    previous_errors = [dict(file="system/A", error_content="Previous solver failure")]
    state = initialize_state("Supply A and retain B", Config(execution_mode="fast"))
    state.update(case_dir=str(tmp_path), case_input_source_dir=str(workspace.source_dir),
                 llm_service=SimpleNamespace(invoke=invoke), requires_hpc=False,
                 solver_operations=operations, case_solver="icoFoam",
                 review_analysis="Supply A and retain B", error_logs=previous_errors,
                 input_writer_mode="initial", subtasks=[task], pending_solver_subtasks=[task],
                 rewrite_plan=dict(target_files=[
                     dict(file="system/B", stage="solver", changes="Retain the correction")]))

    result = input_writer_node(state)

    assert (workspace.source_dir / "system/A").read_text() == "created A"
    assert (workspace.source_dir / "system/B").read_text() == "corrected B"
    assert result["subtasks"] == result["pending_solver_subtasks"] == []
    if already_exists:
        assert result["termination_reason"] == "repair_made_no_progress"
        assert result["error_logs"] == previous_errors
        assert route_after_input_writer({**state, **result}) == "__end__"
    else:
        assert result["updated_files"] == ["system/A"]
        assert not result["error_logs"]
        assert route_after_input_writer({**state, **result}) == "local_runner"


@pytest.mark.parametrize("scenario", ["old-log", "missing-result", "completed"])
def test_solver_success_requires_current_completion_and_result(tmp_path, monkeypatch, scenario):
    from services import solver_execution
    from services.execution_logs import snapshot_execution_logs

    _put(tmp_path, "log.solve", "Exec : icoFoam\nEnd\n")
    logs_before = snapshot_execution_logs(tmp_path) if scenario == "old-log" else {}
    if scenario != "missing-result":
        _put(tmp_path, "1/U", "FoamFile {class volVectorField;}\n")
    monkeypatch.setattr(solver_execution, "run_openfoam_utility", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, "application icoFoam;\n", ""))

    errors = solver_execution.check_solver_execution(
        str(tmp_path), fields_before={}, logs_before=logs_before)

    assert (not errors) is (scenario == "completed")


def test_mesh_handoff_preserves_clean_solver_inputs(tmp_path):
    from services.case_workspace import CaseWorkspace

    _poly_mesh(tmp_path)
    geometry = _put(tmp_path, "geometry.stl", "clean geometry")
    field = _put(tmp_path, "0/U", "clean initial field")
    workspace = CaseWorkspace.create(str(tmp_path), str(tmp_path / ".foamagent"))
    geometry.write_text("runtime geometry")
    field.write_text("runtime field")
    points = _put(tmp_path, "constant/polyMesh/points", "new mesh points")

    workspace.commit_mesh({"": tmp_path / "constant/polyMesh"})
    workspace.restore_inputs()

    assert geometry.read_text() == "clean geometry"
    assert field.read_text() == "clean initial field"
    assert points.read_text() == "new mesh points"


def test_visualization_failure_status(tmp_path, monkeypatch):
    """A rendering failure preserves simulation success as partial_success."""
    from config import Config

    node = importlib.import_module("nodes.visualization_node")
    monkeypatch.setattr(node, "visualize_case", lambda *a, **kw:
                        {"pyvista_visualization": {"success": False, "error": "render failed"}})
    result = node.visualization_node(dict(case_dir=str(tmp_path), config=Config(), workflow_status="success"))
    assert result["workflow_status"] == "partial_success"
    assert result["termination_reason"] == "visualization_failed"
    from main import workflow_exit_code
    assert workflow_exit_code(result) != 0


def test_standalone_run_preserves_script_and_forwards_request(tmp_path, monkeypatch):
    """Standalone MCP run forwards its request and retains the supplied script."""
    server = importlib.import_module("src.mcp.fastmcp_server")
    script = tmp_path / "Allrun"
    script.write_text("#!/bin/bash\ncustomSolver --user-option\n")
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
