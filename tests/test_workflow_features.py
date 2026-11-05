"""One representative test per workflow feature; model/native/Slurm calls are mocked.

The regional handoff test uses the real PyVista reader. Run a feature with
pytest tests/test_workflow_features.py::test_<feature> -q.
"""

import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture
def modules(monkeypatch, tmp_path):
    """Load workflow services with model and native-tool calls isolated."""
    utils = importlib.import_module("utils")
    monkeypatch.setattr(utils, "LLMService", lambda *_: None)
    names = ("main", "nodes.solver_planner_node", "services.meshing", "services.mesh",
             "services.input_writer", "services.run_local", "services.plan", "services.preparation")
    result = SimpleNamespace(**{name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in names})
    result.read_patch_bounds = result.meshing.read_patch_bounds
    monkeypatch.setattr(result.meshing, "read_patch_bounds", lambda _: {"inlet": [0, 0, 0, 1, 0, 1]})
    from services import solver_execution
    from services.case_import import _application
    monkeypatch.setattr(solver_execution, "run_openfoam_utility", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, _application(Path(kw["working_dir"])) or "", ""))
    monkeypatch.setattr(result.preparation, "setup_logging", lambda *_: None)
    monkeypatch.setattr(result.meshing, "retrieve_mesh_reference", lambda _: {
        "case": None, "content": "", "script": "", "reason": "No tutorial required by this test"})
    yield result
    for path in tmp_path.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)


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
        return {"returncode": result.returncode}

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


def test_multiregion_handoff(modules, tmp_path, monkeypatch):
    """Region-only meshes retain actual regional patches and measured bounds."""
    _cube(tmp_path, "fluid", 0)
    _cube(tmp_path, "solid", 2)
    _put(tmp_path, "constant/regionProperties", "regions (fluid (fluid) solid (solid));\n")
    monkeypatch.setattr(modules.meshing, "read_patch_bounds", modules.read_patch_bounds)
    monkeypatch.setattr(modules.meshing, "run_openfoam_utility", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, "(fluid (fluid) solid (solid))", ""))
    result = modules.meshing.execute_meshing(_state(tmp_path), lambda _: {"error_logs": []})
    assert not result["error_logs"] and not (tmp_path / "constant/polyMesh").exists()
    assert set(result["mesh_info"]["regions"]) == {"fluid", "solid"}
    assert result["mesh_info"]["patch_bounds"] == {
        "fluid/walls": [0, 1, 0, 1, 0, 1], "solid/walls": [2, 3, 0, 1, 0, 1]}


def test_mesh_reconstruction(modules, tmp_path, monkeypatch):
    """Successful reconstruction replaces the mesh; a later native failure rolls back."""
    _poly_mesh(tmp_path)
    _poly_mesh(tmp_path / "processor0")
    initial = _put(tmp_path, "0/U", "preserve initial field")
    fail = False

    def reconstruct(argv, **kwargs):
        _poly_mesh(tmp_path)
        if fail:
            raise subprocess.CalledProcessError(1, argv, stderr="partition read failed")
        _put(tmp_path, "constant/polyMesh/points", "reconstructed mesh")
        return subprocess.CompletedProcess(argv, 0, "reconstructed", "")

    monkeypatch.setattr(modules.meshing, "run_openfoam_utility", reconstruct)
    state = _state(tmp_path, final_reconstruction=dict(
        sources=["processor0/constant/polyMesh"], argv=["reconstructParMesh", "-constant"]))
    first = modules.meshing.execute_meshing(state, lambda _: {"error_logs": []})
    assert not first["error_logs"]
    fail = True
    state.update(first)
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
                        subprocess.CompletedProcess(argv, 0, '"icoFoam"\n', ""))
    assert solver_execution.check_solver_execution(str(tmp_path), fields_before=before)
    _put(tmp_path, "1/U", "FoamFile {class volVectorField;}\n")
    assert not solver_execution.check_solver_execution(str(tmp_path), fields_before=before)


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


def test_hpc_submission_confirmation(modules, monkeypatch, tmp_path):
    """An ambiguous submission response stops instead of inventing a job ID."""
    from services import run_hpc

    calls = []

    def submit(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ambiguous submission response", "")

    monkeypatch.setattr(run_hpc.subprocess, "run", submit)
    with pytest.raises(run_hpc.HpcWorkflowStop) as stopped:
        run_hpc.submit_slurm_job(str(tmp_path / "submit.slurm"))
    assert stopped.value.reason == "hpc_submission_unconfirmed" and len(calls) == 1


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
