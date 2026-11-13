"""Exercise meshing, solver handoff, review, execution, and the MCP run entry."""

import asyncio
import importlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event
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


@pytest.mark.parametrize("failure", ["native", "handoff"])
def test_failed_regional_reconstruction_restores_all_meshes_and_pending_physics(
        tmp_path, monkeypatch, failure):
    """A partial reconstruction cannot replace either canonical or working inputs."""
    from services.case_workspace import CaseWorkspace

    for root in (tmp_path, tmp_path / "processor0"):
        for region in ("fluid", "solid"):
            for name in ("points", "faces", "owner", "neighbour"):
                _put(root, f"constant/{region}/polyMesh/{name}", f"original {region} {name}")
            _put(root, f"constant/{region}/polyMesh/boundary",
                 "1\n(wall {type wall; nFaces 1; startFace 0;})\n")
    _put(tmp_path, "constant/regionProperties", "regions (fluid (fluid) solid (solid));\n")
    initial = _put(tmp_path, "0/U", "original initial field")
    workspace = CaseWorkspace.create(str(tmp_path), str(tmp_path / ".foamagent"))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*")
              if p.is_file() and ".foamagent" not in p.parts and "polyMesh" in p.parts}

    def native(argv, **kwargs):
        if argv[0] == "foamDictionary":
            return subprocess.CompletedProcess(argv, 0,
                "" if "-includes" in argv else "regions (fluid (fluid) solid (solid));\n", "")
        if argv[0] == "reconstructParMesh":
            for region in ("fluid", "solid"):
                for name in ("points", "faces", "owner", "neighbour", "boundary"):
                    original = before[Path(f"constant/{region}/polyMesh/{name}")]
                    _put(tmp_path, f"constant/{region}/polyMesh/{name}",
                         original.decode() if name == "boundary" else "reconstructed mesh")
                if failure == "native":
                    return subprocess.CompletedProcess(argv, 1, "", "reconstruction failed")
        return subprocess.CompletedProcess(argv, 0, "Mesh OK.\n", "")

    monkeypatch.setattr(meshing, "run_openfoam_utility", native)
    monkeypatch.setattr(meshing, "read_patch_bounds", lambda *a, **kw:
                        {f"{region}/wall": [0, 1, 0, 1, 0, 1] for region in ("fluid", "solid")})
    rename = Path.rename
    failed = False

    def fail_second_region(path, target):
        nonlocal failed
        if failure == "handoff" and not failed and Path(target) == workspace.source_dir / "constant/solid/polyMesh":
            failed = True
            raise OSError("second-region handoff failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_second_region)
    pending = dict(file="0/U", stage="solver", changes="Retain the pending physical correction")
    state = _state(tmp_path, case_input_source_dir=str(workspace.source_dir),
                   rewrite_plan=dict(target_files=[pending]), final_reconstruction=dict(
                       sources=[f"processor0/constant/{region}/polyMesh" for region in ("fluid", "solid")],
                       argv=["reconstructParMesh", "-constant", "-allRegions"]))

    result = meshing.execute_meshing(state, lambda _: {"error_logs": []})

    assert result["error_logs"] and result["mesh_info"] is None
    assert result["mesh_execution"]["final_reconstruction"]["status"] == "rolled_back"
    assert {**state, **result}["rewrite_plan"]["target_files"] == [pending]
    assert all((tmp_path / path).read_bytes() == content for path, content in before.items())
    assert all((workspace.source_dir / path).read_bytes() == content
               for path, content in before.items() if path.parts[0] == "constant")
    assert (workspace.source_dir / ".foamagent/mesh-sources/processor0/constant/fluid/polyMesh/points").is_file()
    assert initial.read_text() == (workspace.source_dir / "0/U").read_text() == "original initial field"
    assert not (workspace.source_dir / ".foamagent/mesh-handoff").exists()


@pytest.mark.parametrize("stop_confirmed", [True, False])
def test_hpc_monitor_failure_cancels_and_confirms_before_returning(
        tmp_path, monkeypatch, stop_confirmed):
    """Lost monitoring never resubmits a known job or permits an unconfirmed stop."""
    import re
    import shlex
    from services import run_hpc

    cluster = dict(cluster_name="cluster", account_number="account", partition="test",
                   nodes=1, ntasks_per_node=1, time_limit=1, memory=1)

    def invoke(user, system):
        if "Extract cluster information" in user:
            return json.dumps(cluster)
        wrapper = re.search(r"using the (.+?) script", user)[1]
        return "#!/bin/bash\nbash " + shlex.quote(wrapper) + "\n"

    state = dict(case_dir=str(tmp_path), config=Config(max_loop=3),
                 user_requirement="Run on the supplied cluster", loop_count=0,
                 llm_service=SimpleNamespace(invoke=invoke))
    script = _put(tmp_path, "Allrun", "true\n")
    real_run, calls = subprocess.run, []
    monkeypatch.setenv("WM_PROJECT_VERSION", "10")
    monkeypatch.setenv("FOAM_CONTROLDICT", "InfoSwitches {}")

    def scheduler(argv, **kwargs):
        calls.append(argv[0])
        if argv[0] == "sbatch":
            assert 0 < kwargs["timeout"] <= 7
            real_run(["bash", argv[-1]], check=True, capture_output=True, text=True)
            return subprocess.CompletedProcess(argv, 0, "Submitted batch job 123 on cluster clusterA\n", "")
        assert argv[argv.index("--clusters") + 1] == "clusterA"
        if argv[0] == "squeue":
            assert argv[argv.index("-j") + 1] == "123"
            if not stop_confirmed or "scancel" not in calls:
                raise subprocess.CalledProcessError(1, argv, stderr="monitor unavailable")
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert argv[0] == "scancel" and argv[-1] == "123"
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(run_hpc.subprocess, "run", scheduler)
    if stop_confirmed:
        result = run_hpc.run_stage_hpc(state, str(script), "solver", 7)
        assert result["returncode"] == 1 and not result["executed"]
        assert result["cancellation_status"] == "COMPLETED"
    else:
        with pytest.raises(run_hpc.HpcWorkflowStop) as stop:
            run_hpc.run_stage_hpc(state, str(script), "solver", 7)
        assert stop.value.reason == "hpc_job_state_unconfirmed"
    assert calls.count("sbatch") == calls.count("scancel") == 1
    assert calls.count("squeue") == 2


def test_cancelled_mcp_run_keeps_case_locked_until_native_work_returns(tmp_path, monkeypatch):
    """Cancelling a client wait must not allow a second tool to overwrite its case."""
    server = importlib.import_module("src.mcp.fastmcp_server")
    entered, release, second_entered = Event(), Event(), Event()
    calls = []

    def execute(**kwargs):
        calls.append(kwargs["case_dir"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=3)
        else:
            second_entered.set()
        return []

    monkeypatch.setattr(server, "run_allrun_and_collect_errors", execute)
    _put(tmp_path, "Allrun", "true\n")
    run = getattr(server.run, "fn", server.run)

    async def log(*args, **kwargs):
        pass

    async def scenario():
        request = server.RunSimulationRequest(case_dir=str(tmp_path), timeout=2)
        ctx = SimpleNamespace(info=log, error=log)
        first = asyncio.create_task(run(request, ctx))
        following = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            following = asyncio.create_task(run(request, ctx))
            assert not await asyncio.to_thread(second_entered.wait, 0.1)
        finally:
            release.set()
            if following is not None:
                response = await asyncio.wait_for(following, 2)
                assert response.status == "success"
            elif not first.done():
                await asyncio.wait_for(first, 2)
        assert second_entered.is_set() and calls == [str(tmp_path), str(tmp_path)]

    asyncio.run(scenario())


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


@pytest.fixture
def reported_workflow(tmp_path, monkeypatch):
    """Use the real graph and preparation with deterministic offline stage results."""
    app = importlib.import_module("main")
    preparation = importlib.import_module("services.preparation")
    from services.review_history import ReviewHistory

    monkeypatch.setattr(preparation, "load_agent_resources", lambda _: (
        SimpleNamespace(print_statistics=lambda: None), {}))
    monkeypatch.setattr(importlib.import_module("router_func"), "llm_requires_hpc", lambda _: False)
    monkeypatch.setattr(app, "meshing_planner_node", lambda _: dict(
        mesh_type="standard_mesh", mesh_operations=_operations("blockMesh", stage="mesh"),
        mesh_boundary_plan=[dict(patch="inlet", role="inlet", location="x=0",
                                 basis="user", evidence="Requested inlet at x=0")],
        mesh_reference=dict(case="tutorial/cavity", reason="Compatible mesh topology")))
    monkeypatch.setattr(app, "standard_meshing_node", lambda _: dict(
        error_logs=[], mesh_info={"cell_count": 100}, mesh_execution={"executed": True}))
    monkeypatch.setattr(app, "solver_planner_node", lambda _: dict(
        case_solver="icoFoam", solver_operations=_operations("icoFoam"), solver_planned=True,
        requires_input_writer=True, requires_hpc=False, requires_visualization=False))

    def writer(state):
        edits = dict(updated_files=["0/U"] if state["loop_count"] else [],
                     file_diffs=[dict(file="0/U", diff="- dimensions invalid\n+ dimensions [0 1 -1 0 0 0 0]")]
                     if state["loop_count"] else [])
        if state["loop_count"]:
            ReviewHistory(state["case_dir"], state.get("case_import_report_dir")).append(
                "file_repair", stage="solver", **edits)
        return dict(error_logs=[], rewrite_plan=None, **edits)

    def runner(state):
        errors = [] if state["loop_count"] else [{"file": "log.icoFoam", "error_content": "Invalid dimensions"}]
        return dict(error_logs=errors, solver_execution={"executed": True, "error_logs": errors},
                    workflow_status="running" if errors else "success")

    monkeypatch.setattr(app, "input_writer_node", writer)
    monkeypatch.setattr(app, "local_runner_node", runner)
    monkeypatch.setattr(app, "solver_reviewer_node", lambda state: dict(
        loop_count=state["loop_count"] + 1, repairing_mesh=False,
        rewrite_plan={"target_files": [dict(file="0/U", stage="solver", changes="Correct dimensions")]},
        review_analysis="Correct dimensions using the native field convention"))
    yield app, Config(case_dir=str(tmp_path / "output"), execution_mode="fast")
    # Imported original/ trees are deliberately read-only.
    for path in tmp_path.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)


@pytest.mark.parametrize("imported", [False, True], ids=["generated", "imported"])
def test_workflow_report_preserves_decisions_retries_and_evidence(reported_workflow, tmp_path, imported):
    """A browser-readable report follows the shared workflow, including a repaired failure."""
    from html import unescape
    import re
    from urllib.parse import unquote

    app, config = reported_workflow
    source = tmp_path / "source"
    if imported:
        _put(source, "system/controlDict", "/* OpenFOAM: https://openfoam.org\nVersion: 10\n*/\napplication icoFoam;\n")
    result = app.main("Inlet at x=0 <script>alert('input')</script>", config,
                      case_path=str(source) if imported else None)
    assert result["workflow_status"] == "success" and result["loop_count"] == 1
    directory = Path(config.case_dir) / ("report" if imported else ".foamagent")
    content = (directory / "workflow_report.html").read_text()
    timeline = content.split('<ol class="timeline">', 1)[1].split('</ol>', 1)[0]
    assert timeline.index("Plan mesh") < timeline.index("Plan physics and solver")
    assert "Run solver locally · attempt 2" in timeline
    assert "Invalid dimensions" in content and "Correct dimensions" in content
    assert "- dimensions invalid" in content and "+ dimensions [0 1 -1 0 0 0 0]" in content
    assert "Requested inlet at x=0" in content and "Compatible mesh topology" in content
    assert "Not assessed by this report" in content
    assert "&lt;script&gt;" in content and "<script>" not in content
    for href in re.findall(r'href="([^"]+)"', content):
        if not href.startswith("#"):
            assert (directory / unquote(unescape(href))).is_file()
    if imported:
        assert not list((Path(config.case_dir) / "original").rglob("workflow_report.html"))
        assert not list(source.rglob("workflow_report.html"))


@pytest.mark.parametrize("failure", ["exception", "visualization"])
def test_workflow_report_retains_failure_without_changing_its_result(reported_workflow, failure, monkeypatch):
    app, config = reported_workflow
    if failure == "exception":
        def fail(_):
            raise ValueError("Planning unavailable")
        monkeypatch.setattr(app, "solver_planner_node", fail)
        with pytest.raises(ValueError, match="Planning unavailable"):
            app.main("Run this case", config)
        expected = "Planning unavailable"
    else:
        planner = app.solver_planner_node
        monkeypatch.setattr(app, "solver_planner_node", lambda state: {**planner(state), "requires_visualization": True})
        node = importlib.import_module("nodes.visualization_node")
        monkeypatch.setattr(node, "visualize_case", lambda *a, **kw: dict(
            pyvista_visualization={"success": False, "error": "Rendering unavailable"}))
        result = app.main("Run this case", config)
        assert result["workflow_status"] == "partial_success" and app.workflow_exit_code(result) != 0
        expected = "Rendering unavailable"
    content = (Path(config.case_dir) / ".foamagent/workflow_report.html").read_text()
    assert expected in content and "Not assessed by this report" in content
    assert ("Not reached" if failure == "exception" else "Completed") in content


def test_workflow_report_respects_output_ownership_and_write_failures(reported_workflow, tmp_path, monkeypatch, capsys):
    app, config = reported_workflow
    # Rejection of an existing output must leave it entirely untouched.
    sentinel = _put(Path(config.case_dir), "keep.txt", "valuable input")
    result = app.main("Run this case", config)
    assert result["workflow_status"] == "failed"
    assert list(Path(config.case_dir).iterdir()) == [sentinel]
    sentinel.unlink()
    # A report path pointing elsewhere must neither be followed nor fail the simulation.
    outside = _put(tmp_path, "outside.txt", "preserve this file")
    planner = app.solver_planner_node

    def plan(state):
        (Path(config.case_dir) / ".foamagent/workflow_report.html").symlink_to(outside)
        return planner(state)

    monkeypatch.setattr(app, "solver_planner_node", plan)
    result = app.main("Run this case", config)
    assert result["workflow_status"] == "success"
    assert outside.read_text() == "preserve this file"
    assert "workflow_report_warning" in capsys.readouterr().err


@pytest.mark.parametrize("runner", ["local", "hpc"])
def test_workflow_report_marks_skipped_solver_execution(reported_workflow, monkeypatch, runner):
    """Returning to earlier failed inputs records a skip with no new execution evidence."""
    app, config = reported_workflow
    config.max_loop = 4
    planner = app.solver_planner_node
    monkeypatch.setattr(app, "solver_planner_node", lambda state: {
        **planner(state), "requires_hpc": runner == "hpc"})

    def writer(state):
        source = Path(state["case_input_source_dir"])
        _put(source, "system/controlDict", "application icoFoam;\n")
        _put(source, "Allrun", "icoFoam\n")
        # Two distinct failed inputs, then an actual edit back to the first.
        _put(source, "0/U", f"internalField uniform ({state['loop_count'] % 2} 0 0);\n")
        return dict(error_logs=[], rewrite_plan=None)

    monkeypatch.setattr(app, "input_writer_node", writer)
    node = importlib.import_module(f"nodes.{runner}_runner_node")
    calls = []
    errors = [{"file": "log.icoFoam", "error_content": "Previous native failure"}]
    if runner == "local":
        monkeypatch.setattr(node, "check_stage_inputs", lambda *a: [])
        monkeypatch.setattr(node, "check_stage_script", lambda *a: [])
        monkeypatch.setattr(node, "run_allrun_and_collect_errors", lambda *a, **kw: calls.append(1) or errors)
    else:
        def execute(state):
            calls.append(1)
            return dict(error_logs=errors, job_id="previous-job", solver_execution={
                "executed": True, "job_id": "previous-job", "error_logs": errors})
        monkeypatch.setattr(node, "_run_hpc", execute)
    monkeypatch.setattr(app, f"{runner}_runner_node", getattr(node, f"{runner}_runner_node"))
    result = app.main("Preserve the physics", config)
    assert len(calls) == 2 and result["termination_reason"] == "repair_made_no_progress"
    # Reporting must leave the workflow's original retained outcome intact.
    assert result["solver_execution"]["executed"] is True
    content = (Path(config.case_dir) / ".foamagent/workflow_report.html").read_text()
    skipped = content.split('<ol class="timeline">', 1)[1].split('</ol>', 1)[0].rsplit('<li id="step-', 1)[1]
    assert "Skip solver execution" in skipped and "Skipped" in skipped
    assert "<pre>False</pre>" in skipped and "<pre>True</pre>" not in skipped
    assert "Previous execution errors" in skipped and "Previous native failure" in skipped
    assert "previous-job" not in skipped


def test_workflow_report_keeps_each_planner_reference(reported_workflow, monkeypatch):
    """A solver replan preserves both reference snapshots even though the source file changes."""
    from services import plan

    app, config = reported_workflow
    metadata = dict(case_name="case", case_domain="fluid", case_category="incompressible", case_solver="icoFoam")
    monkeypatch.setattr(plan, "parse_requirement_to_case_info", lambda *a, **kw: metadata)
    references = iter(["FIRST_TUTORIAL", "SECOND_TUTORIAL"])

    def retrieve(**kwargs):
        reference = next(references)
        return reference, "", "", reference + "_SCRIPT", plan.SimilarCaseAdviceModel(solver="icoFoam")

    monkeypatch.setattr(plan, "retrieve_references", retrieve)
    planner = app.solver_planner_node
    visits = []

    def replan(state):
        visits.append(1)
        return {**planner(state), **plan.prepare_solver_context(state),
                "repairing_mesh": len(visits) == 1, "replan_solver": len(visits) == 1}

    monkeypatch.setattr(app, "solver_planner_node", replan)
    result = app.main("Preserve the physics", config)
    assert result["workflow_status"] == "success" and len(visits) == 2
    assert (Path(config.case_dir) / "similar_case.txt").read_text() == "SECOND_TUTORIAL\n\nSECOND_TUTORIAL_SCRIPT"
    content = (Path(config.case_dir) / ".foamagent/workflow_report.html").read_text()
    timeline = content.split('<ol class="timeline">', 1)[1].split('</ol>', 1)[0]
    plans = [step for step in timeline.split('<li id="step-') if "Plan physics and solver" in step]
    assert len(plans) == 2
    assert "FIRST_TUTORIAL" in plans[0] and "FIRST_TUTORIAL_SCRIPT" in plans[0]
    assert "SECOND_TUTORIAL" not in plans[0]
    assert "SECOND_TUTORIAL" in plans[1] and "SECOND_TUTORIAL_SCRIPT" in plans[1]
    assert "FIRST_TUTORIAL" not in plans[1]


@pytest.mark.parametrize("scenario", ["write_warning", "status_notice", "original_error"])
def test_report_diagnostics_preserve_workflow_outcome(reported_workflow, monkeypatch, scenario):
    """Report I/O and its diagnostics may fail without changing a result or exception."""
    import errno
    from logger import FoamAgentLogger

    app, config = reported_workflow
    write_log = FoamAgentLogger.write

    def full_report_log(self, text, **kwargs):
        if text.startswith("<workflow_report"):
            raise OSError(errno.ENOSPC, "No space left for report diagnostics")
        return write_log(self, text, **kwargs)

    monkeypatch.setattr(FoamAgentLogger, "write", full_report_log)
    stream = sys.stderr
    diagnostics = []

    class FailingReportStream:
        def write(self, text):
            if text.startswith("<workflow_report"):
                diagnostics.append(text)
                raise OSError(errno.ENOSPC, "Diagnostic stream unavailable")
            return stream.write(text)

        def flush(self):
            stream.flush()

    monkeypatch.setattr(sys, "stderr", FailingReportStream())
    write_file = Path.write_text

    def write_report(path, *args, **kwargs):
        if path.name == "workflow_report.html" and scenario != "status_notice":
            raise OSError(errno.ENOSPC, "No space left for report")
        return write_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_report)
    if scenario == "original_error":
        def fail(_):
            raise ValueError("Original planning failure")
        monkeypatch.setattr(app, "solver_planner_node", fail)
        with pytest.raises(ValueError, match="Original planning failure"):
            app.main("Preserve the physics", config)
    else:
        result = app.main("Preserve the physics", config)
        assert result["workflow_status"] == "success" and app.workflow_exit_code(result) == 0
    assert diagnostics
    assert "<workflow_report" not in (Path(config.case_dir) / "workflow.log").read_text()
    if scenario == "status_notice":
        assert (Path(config.case_dir) / ".foamagent/workflow_report.html").is_file()
