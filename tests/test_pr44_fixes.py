"""Focused coverage for changes after PR44; no LLM, OpenFOAM or Slurm required."""

import ast
import asyncio
import gzip
import importlib
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_meshing_workflow import ROOT, _poly_mesh, _put, modules


def _tool_fn(tool):
    """Access the function when FastMCP wraps a registered tool."""
    return getattr(tool, "fn", tool)


def _state(case, **updates):
    from config import Config
    return {"case_dir": str(case), "config": Config(openfoam_target="foundation-v10", max_loop=3),
            "user_requirement": "Run the case", "llm_service": None, **updates}


def _shell(script, out, err, case, *args, **kwargs):
    result = subprocess.run(["bash", script], cwd=case, capture_output=True, text=True)
    Path(out).write_text(result.stdout)
    Path(err).write_text(result.stderr)
    return {"returncode": result.returncode}


@pytest.mark.parametrize("start,overwrite", [(1, False), (7, True)])
def test_preparation_allocates_independent_default_runs(modules, tmp_path, monkeypatch, start, overwrite):
    from config import Config
    from services.output_safety import output_is_owned

    config = Config(run_directory=tmp_path / "runs", run_times=start, overwrite_case_dir=overwrite)
    llm = object()
    monkeypatch.setattr(modules.preparation, "load_agent_resources", lambda _: (llm, {}))
    state = {"config": config, "workflow_mode": "prompt", "requires_hpc": False}
    first = modules.preparation.prepare_inputs({**state, "user_requirement": "Run cavity flow"})
    first_dir = config.run_directory / f"run_{start}"
    assert first["workflow_status"] == "planning" and Path(first["case_dir"]) == first_dir
    _put(first_dir, "0/U", "existing initial field")

    second = modules.preparation.prepare_inputs({**state, "user_requirement": "Run cylinder flow"})
    second_dir = config.run_directory / f"run_{start + 1}"
    assert second["workflow_status"] == "planning" and Path(second["case_dir"]) == second_dir
    assert first["case_name"] == first_dir.name and second["case_name"] == second_dir.name
    assert (first_dir / "0/U").read_text() == "existing initial field"
    assert output_is_owned(first_dir) and output_is_owned(second_dir)
    assert config.case_dir == "" and config.run_times == start


def test_default_runs_skip_occupied_names(modules, tmp_path):
    from config import Config

    runs = tmp_path / "runs"
    _put(runs, "run_7/Allrun", "previous run")
    _put(runs, "run_8", "unrelated file")
    (runs / "run_9").symlink_to(tmp_path / "missing")
    config = Config(run_directory=runs, run_times=7, overwrite_case_dir=True)
    case = modules.preparation._prepare_generated_case_dir(config, None)
    assert case == runs / "run_10"
    assert (runs / "run_7/Allrun").read_text() == "previous run"
    assert (runs / "run_8").read_text() == "unrelated file"
    assert (runs / "run_9").is_symlink() and not (tmp_path / "missing").exists()


def test_concurrent_default_runs_reserve_different_directories(modules, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from config import Config
    from services.output_safety import output_is_owned

    config = Config(run_directory=tmp_path / "runs")
    barrier = Barrier(2)
    validate = modules.preparation.validate_output_path

    def validate_together(path, **kwargs):
        result = validate(path, **kwargs)
        if Path(path).name == "run_1":
            barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(modules.preparation, "validate_output_path", validate_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        cases = list(pool.map(lambda _: modules.preparation._prepare_generated_case_dir(config, None), range(2)))
    assert set(cases) == {config.run_directory / "run_1", config.run_directory / "run_2"}
    assert all(output_is_owned(case) for case in cases)


def test_explicit_output_keeps_existing_overwrite_rules(modules, tmp_path):
    from config import Config
    from services.output_safety import OutputDirectorySafetyError, OUTPUT_MARKER_NAME

    output = tmp_path / "chosen-output"
    config = Config(case_dir=str(output), run_directory=tmp_path / "unused-runs")
    assert modules.preparation._prepare_generated_case_dir(config, None) == output
    _put(output, "0/U", "existing field")
    with pytest.raises(FileExistsError):
        modules.preparation._prepare_generated_case_dir(config, None)
    assert (output / "0/U").read_text() == "existing field"
    config.overwrite_case_dir = True
    assert modules.preparation._prepare_generated_case_dir(config, None) == output
    assert not (output / "0/U").exists()
    (output / OUTPUT_MARKER_NAME).unlink()
    _put(output, "valuable-input", "preserve me")
    with pytest.raises(OutputDirectorySafetyError):
        modules.preparation._prepare_generated_case_dir(config, None)
    assert (output / "valuable-input").read_text() == "preserve me"
    assert not config.run_directory.exists()


@pytest.mark.parametrize("unsafe", ["symlink_parent", "whitespace", "source_directory", "parent_file"])
def test_default_runs_retain_output_path_checks(modules, tmp_path, unsafe):
    from config import Config

    source = None
    if unsafe == "symlink_parent":
        target = tmp_path / "target"
        target.mkdir()
        runs = tmp_path / "runs"
        runs.symlink_to(target, target_is_directory=True)
    elif unsafe == "whitespace":
        runs = tmp_path / "runs with spaces"
    elif unsafe == "source_directory":
        source = tmp_path / "source"
        source.mkdir()
        runs = source / "runs"
    else:
        runs = tmp_path / "file"
        runs.write_text("preserve me")
    with pytest.raises((OSError, ValueError)):
        modules.preparation._prepare_generated_case_dir(Config(run_directory=runs), source)
    assert not (runs / "run_1").exists()


def test_mesh_planner_uses_current_sources_and_early_hpc_routing(modules, tmp_path, monkeypatch):
    _poly_mesh(tmp_path / "processor0")
    _put(tmp_path, "duct.msh", "mesh")
    _put(tmp_path, "constant/triSurface/duct.stl", "solid duct")
    _put(tmp_path, ".foamagent/mesh-logs/old.msh", "archive")
    reconstruction = {"sources": ["processor0/constant/polyMesh"],
                      "argv": ["reconstructParMesh", "-constant"]}

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj is None:
            return "hpc_run"
        files = ast.literal_eval(re.search(r"<case_files>(.*?)</case_files>", user, re.S)[1])
        assert "constant/triSurface/duct.stl" in files
        assert not any(".foamagent" in name for name in files)
        assert "processor0/constant/polyMesh" in user
        return pydantic_obj(mesh_type="custom_mesh", mesh_source=next(
            name for name in files if name.endswith(".msh")), final_reconstruction=reconstruction)

    llm = SimpleNamespace(invoke=invoke)
    monkeypatch.setattr(modules.preparation, "prepare_imported_case", lambda _: {
        "case_dir": str(tmp_path), "case_context": {"platform": "foundation-v10"}, "llm_service": llm})
    state = _state(tmp_path, workflow_mode="imported_case")
    state.update(modules.preparation.prepare_inputs(state))
    assert state["requires_hpc"] is True
    for name in ("duct.msh", "replacement.msh"):
        plan = modules.meshing.plan_meshing(state)
        assert plan["mesh_source"] == str(tmp_path / name)
        assert plan["final_reconstruction"]["sources"] == reconstruction["sources"]
        if name == "duct.msh":
            (tmp_path / name).rename(tmp_path / "replacement.msh")


def test_custom_mesh_source_retries_switches_and_same_file(modules, tmp_path):
    a, b, case = tmp_path / "a.msh", tmp_path / "b.msh", tmp_path / "case"
    a.write_text("A")
    b.write_text("B")
    converted = []

    def convert(argv, **kwargs):
        assert argv == ["gmshToFoam", "geometry.msh"]
        converted.append((case / "geometry.msh").read_text())
        _poly_mesh(case)

    for source, retry in ((a, False), (a, True), (b, True), (case / "geometry.msh", False)):
        result = modules.mesh.copy_custom_mesh(str(source), str(case), retry=retry, utility_runner=convert)
        assert not result["error_logs"]
        if len(converted) == 1:
            (case / "geometry.msh").write_text("repaired A")
    assert converted == ["A", "repaired A", "B", "B"]


def test_mesh_handoff_and_cleanup_preserve_required_inputs(modules, tmp_path):
    _poly_mesh(tmp_path)
    _put(tmp_path, "0/U", "initial field")
    validate = modules.meshing.validate_mesh_handoff
    for relative in ("0/polyMesh", "1/polyMesh", "processor0/constant/polyMesh"):
        _put(tmp_path, relative + "/points", "unresolved mesh")
        assert validate(tmp_path)
        shutil.rmtree(tmp_path / relative)
    _put(tmp_path, "0/polyMesh/cellMap", "auxiliary")
    assert not validate(tmp_path)
    mesh = tmp_path / "constant/polyMesh"
    (mesh / "points").unlink()
    assert validate(tmp_path)
    _poly_mesh(tmp_path)
    for path in mesh.iterdir():
        Path(str(path) + ".gz").write_bytes(gzip.compress(path.read_bytes()))
        path.unlink()
    result = modules.meshing.execute_meshing(_state(tmp_path), lambda _: {"error_logs": []})
    assert not result["error_logs"] and "inlet" in result["mesh_info"]["boundary_text"]
    shutil.copytree(mesh, tmp_path / "1/polyMesh")
    _put(tmp_path, "2/U", "old solver output")
    modules.run_local._cleanup_run_artifacts(str(tmp_path), preserve_mesh=True)
    assert (tmp_path / "1/polyMesh/points.gz").exists()
    assert not (tmp_path / "2").exists()
    assert (tmp_path / "0/U").read_text() == "initial field"


@pytest.mark.parametrize("outcome", ["missing_source", "failed", "incomplete", "success"])
def test_final_reconstruction_requires_complete_sources_and_new_output(modules, tmp_path, monkeypatch, outcome):
    _poly_mesh(tmp_path)
    source = "processor0/constant/polyMesh"
    _poly_mesh(tmp_path / "processor0")
    old = tmp_path / "constant/polyMesh/points"
    old.write_text("old serial mesh")
    if outcome == "missing_source":
        (tmp_path / source / "points").unlink()

    def reconstruct(argv, **kwargs):
        assert argv == ["reconstructParMesh", "-constant"] and not old.exists()
        if outcome == "failed":
            raise subprocess.CalledProcessError(1, argv, stderr="reconstruction failed")
        _poly_mesh(tmp_path)
        if outcome == "incomplete":
            old.unlink()
        return subprocess.CompletedProcess(argv, 0, "reconstructed", "")

    monkeypatch.setattr(modules.meshing, "run_openfoam_utility", reconstruct)
    state = _state(tmp_path, final_reconstruction={"sources": [source], "argv": ["reconstructParMesh", "-constant"]})
    result = modules.meshing.execute_meshing(state, lambda _: {"error_logs": []})
    assert bool(result["error_logs"]) == (outcome != "success")
    if outcome == "missing_source":
        assert old.read_text() == "old serial mesh"
    elif outcome == "success":
        assert old.read_text() != "old serial mesh" and (tmp_path / source).exists()
        modules.run_local._cleanup_run_artifacts(str(tmp_path))
        assert old.exists() and not (tmp_path / "processor0").exists()


def test_explicit_allmesh_and_mixed_file_context(modules, tmp_path, monkeypatch):
    from utils import load_case_files
    _poly_mesh(tmp_path)
    _put(tmp_path, "system/controlDict", "application simpleFoam; endTime 237;")
    script = "#!/bin/sh\nprintf executed > mesh-result\n"

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj:
            assert "endTime 237" in user and "meshScale 0.0037" in user
            return pydantic_obj(list_foamfile=[dict(folder_name="system", file_name="controlDict",
                                                  content="application simpleFoam; endTime 237; writeInterval 2;")])
        return script if "<file_name>Allmesh</file_name>" in system else "meshScale 0.0037;"

    state = _state(tmp_path, llm_service=SimpleNamespace(invoke=invoke), mesh_type="standard_mesh",
        mesh_commands=["must not replace the explicit script"], mesh_subtasks=[
            dict(folder_name="system", file_name="meshSettings", changes="Set meshScale 0.0037"),
            dict(folder_name="", file_name="Allmesh", changes="Execute the prepared mesh script")],
        rewrite_plan={"target_files": [dict(file="system/controlDict", stage="mesh", changes="Set writeInterval 2")]},
        review_analysis="Preserve existing settings")
    state.update(load_case_files(state))
    monkeypatch.setattr(modules.meshing, "run_command", _shell)
    monkeypatch.setattr(modules.meshing, "build_allrun", lambda **_: pytest.fail("Explicit Allmesh replaced"))
    result = modules.meshing.execute_meshing(state, modules.meshing.standard_meshing)
    assert not result["error_logs"] and (tmp_path / "mesh-result").read_text() == "executed"
    assert "endTime 237; writeInterval 2" in (tmp_path / "system/controlDict").read_text()


def test_mesh_retry_rotates_tool_logs_and_preserves_inputs(modules, tmp_path, monkeypatch):
    inputs = {"input/material.log": "material", "constant/triSurface/logo.stl": "geometry",
              "workflow.log": "ERROR: historical workflow", "review.log": "ERROR: historical review"}
    for name, content in inputs.items():
        _put(tmp_path, name, content)
    _put(tmp_path, "log.tool", "ERROR: old native failure")
    monkeypatch.setattr(modules.meshing, "run_command", _shell)
    for message, failed in (("ERROR: current native failure", True), ("native success", False)):
        _put(tmp_path, "Allmesh", "set -e\ntest ! -e log.tool\nprintf ran > log.tool\n"
             f"printf '{message}' > native.out\n")
        result = modules.meshing._run_mesh_commands(_state(tmp_path, explicit_allmesh=True), [])
        assert result["mesh_execution"]["command_result"]["returncode"] == 0
        assert bool(result["error_logs"]) is failed
        assert result["mesh_execution"]["logs"]["native.out"] == message
        assert "workflow.log" not in result["mesh_execution"]["logs"]
        assert all((tmp_path / name).read_text() == content for name, content in inputs.items())
    assert len(list((tmp_path / ".foamagent/mesh-logs").glob("*/log.tool"))) == 2


def _solver_state(case, llm):
    from config import Config
    from utils import load_case_files

    state = {
        "case_dir": str(case), "config": Config(openfoam_target="foundation-v10"),
        "case_origin": "imported", "llm_service": llm, "user_requirement": "Repair the case",
        "loop_count": 0, "solver_planned": True, "solver_allrun": "",
        "input_writer_mode": "initial", "requires_input_writer": True,
        "requires_hpc": False, "review_analysis": "Apply the explicit file changes",
        "rewrite_plan": None, "tutorial_reference": "", "case_solver": "icoFoam",
    }
    return {**state, **load_case_files(state)}


@pytest.mark.parametrize("action,mode", [
    ("create", "sequential_dependency"), ("create", "parallel_no_context"),
    ("modify", "sequential_dependency"),
])
def test_explicit_solver_script_preserves_instructions_and_shell(modules, tmp_path, monkeypatch, action, mode):
    writer = importlib.import_module("nodes.input_writer_node")
    script = "#!/bin/sh\ncat <<'EOF'\n$(literal) `unchanged`\nEOF\nicoFoam\n"
    calls = []

    def invoke(user, system, pydantic_obj=None):
        calls.append(user)
        if pydantic_obj:
            return pydantic_obj(list_foamfile=[{"folder_name": "", "file_name": "Allrun", "content": script}])
        return "```sh\n" + script + "```"

    monkeypatch.setattr(writer, "_build_solver_allrun", lambda _: pytest.fail("Explicit Allrun overwritten"))
    state = _solver_state(tmp_path, SimpleNamespace(invoke=invoke))
    state["config"].input_writer_generation_mode = mode
    task = {"folder_name": "", "file_name": "Allrun", "stage": "solver",
            "action": action, "changes": "Retain the quoted heredoc before icoFoam"}
    if action == "create":
        state.update(subtasks=[task], pending_solver_subtasks=[task])
    else:
        state.update(input_writer_mode="rewrite", rewrite_plan={"target_files": [
            {"file": "Allrun", "stage": "solver", "changes": task["changes"]},
        ]})
    result = writer.input_writer_node(state)
    assert len(calls) == 1 and task["changes"] in calls[0]
    assert (tmp_path / "Allrun").read_text() == script
    assert not result.get("error_logs") and result["solver_allrun"] is None
    if action == "create":
        assert result["pending_solver_subtasks"] == result["subtasks"] == []


def test_pending_solver_tasks_survive_retries_once_with_distinct_edits(modules, tmp_path, monkeypatch):
    review = importlib.import_module("services.review")
    writer = importlib.import_module("nodes.input_writer_node")
    monkeypatch.setattr(review, "log_review", lambda *args: None)
    monkeypatch.setattr(writer, "_build_solver_allrun", lambda _: pytest.fail("Explicit Allrun overwritten"))
    _poly_mesh(tmp_path)
    (tmp_path / "system").mkdir()
    (tmp_path / "system/fvSolution").write_text("original settings")
    edits = [
        {"file": "system/fvSolution", "stage": "solver", "changes": "Correct the pressure tolerance"},
        {"file": "./system/fvSolution", "stage": "solver", "changes": "Preserve velocity relaxation"},
    ]
    plans, creates, rewrites = [], [], []
    script = "#!/bin/sh\nicoFoam -noFunctionObjects\n"

    def invoke(user, system, pydantic_obj=None):
        if pydantic_obj is modules.meshing.MeshingPlan:
            plans.append(user)
            return pydantic_obj(mesh_type="custom_mesh", mesh_subtasks=[{
                "folder_name": "" if len(plans) == 1 else ".", "file_name": "Allrun",
                "stage": "solver", "action": "create", "changes": f"Use solver plan {len(plans)}",
            }])
        if pydantic_obj is review.MeshRepairPlan:
            return pydantic_obj(target_files=state["rewrite_plan"]["target_files"],
                                repair_domain="mesh", action="continue")
        if pydantic_obj is modules.meshing.MeshRepairImpact:
            return pydantic_obj(replan_solver=False, reason="Preserve the physical plan")
        if pydantic_obj:
            rewrites.append(user)
            return pydantic_obj(list_foamfile=[{
                "folder_name": "system", "file_name": "fvSolution", "content": "corrected settings",
            }])
        if "<file_task>" in user:
            creates.append(user)
            return script
        return "Retry the mesh while preserving pending solver tasks"

    state = _solver_state(tmp_path, SimpleNamespace(invoke=invoke))
    state.update(repairing_mesh=True, rewrite_plan={"target_files": edits})
    for attempt in range(3):
        state.update(modules.meshing.plan_meshing(state))
        state.update(modules.meshing.execute_meshing(state, lambda _: {"error_logs": [
            {"file": "Allmesh", "error_content": "failed mesh command"},
        ] if attempt < 2 else []}))
        assert len(state["rewrite_plan"]["target_files"]) == 2
        if attempt < 2:
            state.update(review.review_case(state))
    assert [t["changes"] for t in state["rewrite_plan"]["target_files"]] == [t["changes"] for t in edits]
    state.update(writer.input_writer_node(state))
    assert len(creates) == len(rewrites) == 1 and "Use solver plan 3" in creates[0]
    assert all(t["changes"] in rewrites[0] for t in edits)
    assert (tmp_path / "Allrun").read_text() == script
    assert (tmp_path / "system/fvSolution").read_text() == "corrected settings"
    assert state["pending_solver_subtasks"] == [] and not state["error_logs"]


def test_mcp_plan_writer_preserves_root_paths_and_task_metadata(modules, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    server = importlib.import_module("src.mcp.fastmcp_server")
    database = tmp_path / "database"
    (database / "raw").mkdir(parents=True)
    (database / "raw/openfoam_case_stats.json").write_text("{}")
    monkeypatch.setattr(server, "require_target_corpus", lambda *_: None)
    monkeypatch.setattr(server, "database_path_for_config", lambda *_: database)
    monkeypatch.setattr(server, "global_config", server.Config(openfoam_target="foundation-v10"))
    monkeypatch.setattr(server, "resolve_case_dir", lambda **_: str(tmp_path / "case"))
    monkeypatch.setattr(server, "retrieve_references", lambda **_: ("", "", "", "", None))
    tasks = [
        {"file_name": "Allrun", "folder_name": "", "stage": "solver", "action": "create", "changes": "Use icoFoam"},
        {"file_name": "blockMeshDict", "folder_name": "system", "stage": "mesh", "action": "modify", "changes": "Keep patches"},
        {"file_name": "U", "folder_name": "0"},
    ]
    metadata = dict(case_name="case", case_solver="icoFoam", case_domain="incompressible", case_category="laminar")
    monkeypatch.setattr(server, "generate_simulation_plan", lambda **_: {**metadata, "subtasks": tasks})
    captured = []

    def generate(**kwargs):
        captured.append(kwargs["subtasks"])
        return {"foamfiles": server.FoamPydantic(list_foamfile=[])}

    async def log(message):
        pass

    monkeypatch.setattr(server, "initial_write", generate)
    context = SimpleNamespace(info=log, error=log)
    response = asyncio.run(_tool_fn(server.plan)(server.PlanRequest(user_requirement="Run the case"), context))
    assert response.subtasks[0] == {"file": "Allrun", "folder": "", "stage": "solver", "action": "create", "changes": "Use icoFoam"}
    # Accept both the plan's legacy keys and the service's native path keys.
    for subtasks in (response.subtasks, tasks):
        asyncio.run(_tool_fn(server.input_writer)(server.GenerateFilesRequest(
            **metadata, user_requirement="Run the case", subtasks=subtasks, overwrite_output=True,
        ), context))
    assert captured == [tasks, tasks]


@pytest.mark.parametrize("mode,through_mcp", [
    ("sequential_dependency", False), ("parallel_no_context", False),
    ("sequential_dependency", True),
])
@pytest.mark.parametrize("folder", ["", ".", None, "tools"])
def test_writer_preserves_explicit_allrun_and_automatic_fallback(
    modules, tmp_path, monkeypatch, mode, through_mcp, folder,
):
    writer = modules.input_writer
    case, database = tmp_path / "case", tmp_path / "database"
    _put(database, "raw/openfoam_commands.txt", "icoFoam\n")
    explicit = "#!/bin/sh\nicoFoam -noFunctionObjects\n"
    automatic = "#!/bin/sh\nicoFoam\n"
    changes = "Preserve -noFunctionObjects"
    prompts, progress = [], []

    def invoke(user, system, pydantic_obj=None):
        prompts.append((user, system))
        if pydantic_obj:
            return pydantic_obj(commands=["icoFoam"])
        if "<file_name>Allrun</file_name>" in system:
            assert changes in user
            return explicit
        if "<file_name>controlDict</file_name>" in system:
            return "application icoFoam;"
        return "```sh\n" + automatic + "```"

    monkeypatch.setattr(writer, "retrieve_faiss", lambda *args, **kwargs: [{"full_content": "icoFoam help"}])
    monkeypatch.setattr(writer, "database_path_for_config", lambda *_: database)
    llm = SimpleNamespace(invoke=invoke)
    tasks = [{"folder_name": "system", "file_name": "controlDict"}]
    if folder is not None:
        tasks.append({"folder_name": folder, "file_name": "Allrun", "stage": "solver",
                      "action": "create", "changes": changes})

    if through_mcp:
        monkeypatch.syspath_prepend(str(ROOT))
        server = importlib.import_module("src.mcp.fastmcp_server")
        monkeypatch.setattr(server, "require_target_corpus", lambda *_: None)
        monkeypatch.setattr(server, "database_path_for_config", lambda *_: database)
        monkeypatch.setattr(server, "resolve_case_dir", lambda **_: str(case))
        monkeypatch.setattr(server, "retrieve_references", lambda **_: ("", "", "", "", None))
        monkeypatch.setattr(writer, "global_llm_service", llm)

        async def log(message):
            pass

        async def report_progress(current, total, message):
            progress.append((current, total))

        context = SimpleNamespace(info=log, error=log, report_progress=report_progress)
        # Exercise the public MCP keys and the actual generation service.
        response = asyncio.run(_tool_fn(server.input_writer)(server.GenerateFilesRequest(
            case_name="case", case_solver="icoFoam", case_domain="incompressible",
            case_category="laminar", user_requirement="Run the case",
            subtasks=[{"file": t["file_name"], "folder": t["folder_name"],
                       **{key: t[key] for key in ("stage", "action", "changes") if key in t}}
                      for t in tasks],
        ), context))
        assert response.allrun_script == str(case / "Allrun")
    else:
        result = writer.initial_write(
            case_dir=str(case), subtasks=tasks, user_requirement="Run the case",
            tutorial_reference="", case_solver="icoFoam", generation_mode=mode,
            database_path=str(database), llm_service=llm,
            progress_callback=lambda current, total, message: progress.append((current, total)),
        )
        root_scripts = [f for f in result["foamfiles"].list_foamfile
                        if f.file_name == "Allrun" and f.folder_name == ""]
        assert len(root_scripts) == 1
        assert root_scripts[0].content == (explicit if folder in ("", ".") else automatic)

    expected = explicit if folder in ("", ".") else automatic
    assert (case / "Allrun").read_text() == expected
    if folder == "tools":
        assert (case / "tools/Allrun").read_text() == explicit
    assert sum("Available OpenFOAM commands" in user for user, _ in prompts) == (folder not in ("", "."))
    assert progress[-1][0] == progress[-1][1]


class _Scheduler:
    """Fake Slurm and elapsed time; execute stage wrappers with real bash."""

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


@pytest.fixture
def hpc_case(modules, tmp_path, monkeypatch):
    from config import Config

    hpc = importlib.import_module("services.run_hpc")
    node = importlib.import_module("nodes.hpc_runner_node")
    review = importlib.import_module("services.review")
    prompts = []
    options = SimpleNamespace(partition="normal", reviews=0)

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

    state = dict(case_dir=str(tmp_path), user_requirement="Run on HPC", loop_count=0,
                 config=Config(openfoam_target="foundation-v10", max_loop=3, max_time_limit=1),
                 llm_service=SimpleNamespace(invoke=invoke))
    script = tmp_path / "Allrun"
    script.write_text("exit 0\n")
    monkeypatch.setenv("WM_PROJECT_VERSION", "10")
    return SimpleNamespace(hpc=hpc, node=node, review=review, state=state, script=script,
                           scheduler=_Scheduler(monkeypatch, hpc), prompts=prompts, options=options)


@pytest.mark.parametrize("exit_code", [0, 7, None])
def test_hpc_requires_current_attempt_exit_result(hpc_case, tmp_path, exit_code):
    c = hpc_case
    c.script.write_text(f"exit {exit_code or 0}\n")
    c.scheduler.execute = exit_code is not None
    old = tmp_path / ".foamagent/solver-jobs/old"
    old.mkdir(parents=True)
    (old / "exit-code").write_text("0")
    result = c.hpc.run_stage_hpc(c.state, str(c.script), "solver", 1)
    assert (result["returncode"] == 0) == (exit_code == 0)
    assert result["job_id"] and Path(result["script"]).parent != old
    if exit_code is None:
        assert "No exit result" in result["diagnostic"]


def test_hpc_refreshes_resources_between_stages(hpc_case, tmp_path):
    c = hpc_case
    dictionary = tmp_path / "system/decomposeParDict"
    dictionary.parent.mkdir()
    c.state["cluster_info"] = {"ntasks_per_node": 99}
    for stage, count in [("mesh", 2), ("solver", 16)]:
        dictionary.write_text(f"numberOfSubdomains {count};")
        result = c.hpc.run_stage_hpc(c.state, str(c.script), stage, 1)
        assert result["returncode"] == 0
        assert c.state["cluster_info"]["ntasks_per_node"] == count
        assert f"#SBATCH -n {count}\n" in Path(result["script"]).read_text()


@pytest.mark.parametrize("budget", [2, 3])
def test_hpc_rejection_uses_shared_budget_and_error_context(hpc_case, budget):
    c = hpc_case
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


@pytest.mark.parametrize("failure", ["unknown", "active_job"])
def test_hpc_uncertain_jobs_stop_without_resubmission(hpc_case, failure):
    c = hpc_case
    c.scheduler.submissions = ["unknown" if failure == "unknown" else "accepted"]
    c.scheduler.queue_until, c.scheduler.confirm_cancel = float("inf"), False
    result = c.node.hpc_runner_node(c.state)
    assert result["workflow_status"] == "failed"
    assert result["termination_reason"] == (
        "hpc_submission_unconfirmed" if failure == "unknown" else "hpc_job_state_unconfirmed")
    from langgraph.graph import END
    from router_func import route_after_runner
    assert route_after_runner(result) == END
    assert sum(a[0] == "sbatch" for a in c.scheduler.calls) == 1
    if failure == "active_job":
        assert result["job_id"] and c.scheduler.clock == 61
        assert sum(a[0] == "scancel" for a in c.scheduler.calls) == 1


def test_hpc_deadline_includes_scheduler_query(hpc_case):
    c = hpc_case
    c.scheduler.query_seconds = 10
    status, ok, error = c.hpc.wait_for_job("123", max_wait_time=1)
    assert status == "TIMEOUT" and not ok and "Monitoring exceeded" in error
    assert c.scheduler.clock == 1


def test_native_mesh_waits_on_slurm_using_configured_deadline(hpc_case, modules):
    c = hpc_case
    c.state["requires_hpc"], c.state["config"].max_time_limit = True, 120
    c.scheduler.queue_until = 60
    runner = modules.meshing._native_mesh_runners(c.state)["utility_runner"]
    result = runner(["bash", "-c", "printf mesh-ran"], timeout=10)
    assert result.returncode == 0 and result.stdout == "mesh-ran"
    assert c.scheduler.clock == 60 and c.state["mesh_jobs"][-1]["returncode"] == 0
    assert sum(a[0] == "sbatch" for a in c.scheduler.calls) == 1


@pytest.mark.parametrize("backend", ["custom", "gmsh"])
def test_uncertain_hpc_job_escapes_mesh_retries(modules, tmp_path, monkeypatch, backend):
    from services.run_hpc import HpcWorkflowStop
    from langgraph.graph import END
    from router_func import route_after_meshing
    calls = []

    def stop(*args, **kwargs):
        calls.append(args)
        raise HpcWorkflowStop("hpc_job_state_unconfirmed", "job still active", {"job_id": "123"})

    monkeypatch.setattr(modules.meshing, "_native_mesh_runners", lambda _: {
        "utility_runner": stop, "python_runner": stop})
    monkeypatch.setattr(modules.mesh, "extract_boundary_names_from_requirements", lambda *a, **kw: [])
    llm = SimpleNamespace(invoke=lambda *a, **kw: modules.mesh.GMSHPythonCode(
        python_code="pass", mesh_type="2D", geometry_type="test"))
    _put(tmp_path, "source.msh", "source")
    state = _state(tmp_path, llm_service=llm, mesh_source=str(tmp_path / "source.msh"))
    execute = modules.meshing.imported_meshing if backend == "custom" else modules.meshing.gmsh_meshing
    result = modules.meshing.execute_meshing(state, execute)
    assert len(calls) == 1 and result["termination_reason"] == "hpc_job_state_unconfirmed"
    assert route_after_meshing(result) == END


@pytest.mark.parametrize("change,exit_code", [(None, 7), (None, None), ("input", 7), ("submission", 7)])
def test_hpc_review_ignores_attempt_ids_but_keeps_real_changes(hpc_case, tmp_path, change, exit_code):
    c = hpc_case
    c.scheduler.execute = exit_code is not None
    c.script.write_text("printf 'specific failure' >&2\nexit 7\n")
    dictionary = tmp_path / "system/controlDict"
    dictionary.parent.mkdir()
    dictionary.write_text("deltaT 0.1;\n")
    c.state.update(c.node.hpc_runner_node(c.state))
    errors = c.state["error_logs"]
    c.state.update(c.review.review_case(c.state))
    if change == "input":
        dictionary.write_text("deltaT 0.01;\n")
    elif change == "submission":
        c.options.partition = "debug"
    c.state.update(c.node.hpc_runner_node(c.state))
    result = c.review.review_case(c.state)
    assert errors == c.state["error_logs"] and "attempt-" not in str(errors)
    assert (result.get("termination_reason") == "repair_made_no_progress") == (change is None)
    assert c.options.reviews == (1 if change is None else 2)
    jobs = c.state["solver_jobs"]
    assert jobs[0]["job_id"] != jobs[1]["job_id"] and jobs[0]["script"] != jobs[1]["script"]
