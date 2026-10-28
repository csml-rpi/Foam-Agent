"""Feature coverage for Preparation -> Meshing -> Solver Planner and shared repair.

Run: python -m pytest tests/test_meshing_workflow.py -q

One parameterized core covers both native targets. LangGraph and services are
real; external LLM, Gmsh and OpenFOAM calls are mocked. Small local shell scripts
exercise Allmesh output capture. Existing import/translator regressions remain
in their own files.
"""

import ast
import importlib
import re
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
    _put(root, "constant/polyMesh/boundary", "2\n(\ninlet { type patch; nFaces 1; startFace 0; }\nwall { type wall; nFaces 2; startFace 1; }\n)\n")


@pytest.fixture
def modules(monkeypatch, tmp_path):
    # Optional provider imports need no credentials or installed SDKs in this test.
    providers = {
        "langchain.chat_models": "init_chat_model",
        "langchain_community.vectorstores": "FAISS",
        "langchain_openai.embeddings": "OpenAIEmbeddings",
        "langchain_aws": "ChatBedrockConverse",
        "langchain_anthropic": "ChatAnthropic",
        "langchain_ollama": "ChatOllama",
        "botocore.exceptions": "ClientError",
        "tracking_aws": None,
        "tiktoken": None,
    }
    for name, symbol in providers.items():
        try:
            importlib.import_module(name)
        except ImportError:
            for size in range(1, len(name.split(".")) + 1):
                part = ".".join(name.split(".")[:size])
                if part not in sys.modules:
                    module = ModuleType(part)
                    module.__path__ = []
                    if part == "langchain":
                        module.debug = module.verbose = False
                        module.llm_cache = None
                    monkeypatch.setitem(sys.modules, part, module)
            if symbol:
                setattr(sys.modules[name], symbol, type(symbol, (Exception,) if symbol == "ClientError" else (), {}))
    utils = importlib.import_module("utils")
    monkeypatch.setattr(utils, "LLMService", lambda *_: None)
    names = ("main", "nodes.solver_planner_node", "nodes.local_runner_node",
             "services.meshing", "services.mesh", "services.input_writer", "services.run_local", "services.plan",
             "services.preparation")
    result = SimpleNamespace(**{name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in names})
    monkeypatch.setattr(result.preparation, "setup_logging", lambda *_: None)
    yield result
    # The existing importer makes original/ immutable; allow pytest teardown.
    for path in tmp_path.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)


@pytest.mark.parametrize("scenario", [
    # Inputs and Preparation: CLI, working copies, combined sources and targets.
    "cli_mesh", "cli_case_mesh", "standard", "gmsh", "custom", "case_poly",
    "case_upload", "case_prompt", "target_mismatch", "standalone_plan",
    # Planning: LLM-selected operations, explicit file stages and initial edits.
    "upload_generate", "case_initial_regenerate", "standard_snappy", "case_no_commands",
    "case_external_switch", "case_incomplete", "case_first_edit", "solver_named_mesh",
    "solver_requests_mesh", "case_deferred_physics", "case_command_mesh", "allmesh_first_edit",
    # Execution: native failures, metadata, startup files and ordered commands.
    "external_failure", "missing_mesh",
    "case_unparsed_boundary", "case_unparsed_native_error",
    "utility_lifecycle", "mesh_log_capture", "mesh_log_timeout", "mesh_log_read_error",
    "custom_repeated", "custom_retry_copy",
    # Repair: shared Reviewer, stop/budget decisions and pending physical tasks.
    "mesh_retry", "case_regenerate", "solver_retry", "solver_mixed_retry", "solver_mixed_replan",
    "mesh_repeat_stop", "mesh_repeat_budget", "solver_repeat_unchanged",
    "allmesh_repair_without_commands",
    "pending_physics", "pending_physics_create_failure", "pending_physics_edit_failure",
    "pending_physics_mesh_failure",
])
@pytest.mark.parametrize("target", ["foundation-v10", "esi-v2006"])
def test_mesh_before_solver_workflow(scenario, target, modules, monkeypatch, tmp_path):
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
                assert pydantic_obj is m.plan.OpenFOAMPlanModel
                assert "stage as mesh or solver" in system
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
    if scenario.startswith("allmesh_"):
        from config import Config
        import subprocess
        case = tmp_path / "case"
        _put(case, "Allmesh", "#!/bin/sh\nexit 9\n")
        edited = "#!/bin/sh\nprintf 'planned script' > mesh-result\n"
        def edit(**kwargs):
            assert kwargs["rewrite_plan"]["target_files"][0]["file"] == "Allmesh"
            _put(case, "Allmesh", edited)
            return {}
        def run(script, stdout, stderr, working_dir, *args, **kwargs):
            process = subprocess.run(["bash", script], cwd=working_dir, capture_output=True, text=True)
            Path(stdout).write_text(process.stdout)
            Path(stderr).write_text(process.stderr)
            return {"returncode": process.returncode}
        monkeypatch.setattr(m.meshing, "rewrite_files", edit)
        monkeypatch.setattr(m.meshing, "build_allrun", lambda **kw: pytest.fail("Do not replace the explicitly edited Allmesh"))
        monkeypatch.setattr(m.meshing, "run_command", run)
        state = {"case_dir": str(case), "config": Config(openfoam_target=target),
                 "user_requirement": "Execute the planned mesh script", "llm_service": None,
                 "mesh_type": "standard_mesh", "repairing_mesh": scenario == "allmesh_repair_without_commands",
                 "mesh_commands": [] if scenario == "allmesh_repair_without_commands" else ["plannedOperation"],
                 "rewrite_plan": {"target_files": [{"file": "Allmesh", "stage": "mesh", "changes": "Run the planned operation"}]}}
        _poly_mesh(case)
        result = m.meshing.execute_meshing(state, m.meshing.standard_meshing)
        assert not result["error_logs"]
        assert (case / "Allmesh").read_text() == edited
        assert (case / "mesh-result").read_text() == "planned script"
        return

    if scenario == "custom_retry_copy":
        case = tmp_path / "case"
        source = tmp_path / "source.msh"
        source.write_text("original uploaded mesh")
        observed = []
        def convert(args, **kwargs):
            observed.append((case / "geometry.msh").read_text())
            _poly_mesh(case)
            return SimpleNamespace(stdout="", stderr="")
        monkeypatch.setattr(m.mesh, "run_openfoam_utility", convert)
        for retry in (False, True, True):
            result = m.mesh.copy_custom_mesh(str(source), str(case),
                                            openfoam_target=target, retry=retry)
            assert not result["error_logs"]
            if len(observed) == 1:
                (case / "geometry.msh").write_text("repaired working mesh")
            elif len(observed) == 2:
                (case / "geometry.msh").unlink()
        assert observed == ["original uploaded mesh", "repaired working mesh", "original uploaded mesh"]
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
                assert "executor does not supply default dictionaries" in system
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

    planned, prepared, materialized = [], [], []
    original_planner = m.main.solver_planner_node
    original_prepare = m.preparation.prepare_inputs
    original_materialize = m.preparation.prepare_output_directory
    unparsed = scenario in {"case_unparsed_boundary", "case_unparsed_native_error"}

    def observe_planning(state):
        info = state["mesh_info"]
        assert (Path(info["poly_mesh_path"]) / "boundary").is_file()
        assert info["boundary_text"] == (Path(info["poly_mesh_path"]) / "boundary").read_text()
        planned.append(state)
        return original_planner(state)

    def observe_preparation(state):
        prepared.append(state)
        return original_prepare(state)

    def observe_materialization(*args, **kwargs):
        materialized.append(args)
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(m.main, "solver_planner_node", observe_planning)
    monkeypatch.setattr(m.main, "preparation_node", observe_preparation)
    monkeypatch.setattr(m.preparation, "prepare_output_directory", observe_materialization)

    if scenario == "target_mismatch":
        from config import Config
        source = tmp_path / "source"
        _put(source, "system/controlDict",
             "// Website: https://openfoam.org\n// Version: 10\napplication icoFoam;\n")
        config = Config(case_dir=str(tmp_path / "output"), openfoam_target="esi-v2006")
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
        assert not meshes and not planned and len(prepared) == 1 and not materialized

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
        assert not meshes and not planned and len(prepared) == 2 and not materialized
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
                assert "<case_contents>" in user and "<mesh_facts>" not in user
                assert "<mesh_state>" not in user
                if imported and uploaded:
                    assert str(tmp_path / "output/work/input" / custom_path.name) in user
                    assert "original physical field" in user
                if scenario in {"case_prompt", "case_external_switch"}:
                    assert "<user_requirement>Keep the supplied physics and use the available mesh materials.</user_requirement>" in user
                assert not any(model and model.__name__ in {"OpenFOAMPlanModel", "ExistingCasePlan"} for _, _, model in calls[:-1]) or repair or scenario in {"solver_requests_mesh", "case_command_mesh"}
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
                assert "planned separately by meshing_planner" in user
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
                assert "only the solver" in system
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
            if "minimal controlDict" in system or "basic controlDict" in system:
                return "application icoFoam;\nendTime 1;\n"
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
    assert "patches" not in result["mesh_info"]
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
    assert len(prepared) == 1
    assert len(selections) == len(retrievals) == (2 if scenario in {"solver_mixed_replan", "solver_requests_mesh", "case_command_mesh"} else 1)
    if scenario == "standard_snappy":
        assert "blockMesh" not in mesh_runs[0]
    if scenario in {"upload_generate", "case_initial_regenerate"}:
        assert "blockMesh" in mesh_runs[0]
    if imported:
        assert (source / "Allrun").read_text(encoding="utf-8") == original_script
        assert (tmp_path / "output/original/Allrun").read_text(encoding="utf-8") == original_script
        assert result["mesh_info"]["poly_mesh_path"]
        assert "mesh_state" not in result["case_import_manifest"].to_dict()
        assert "mesh_state" not in result["case_context"] and "mesh_facts" not in result["case_context"]
        if scenario == "case_incomplete":
            assert not (source / "system/fvSchemes").exists()
            assert "Euler" in (Path(result["case_dir"]) / "system/fvSchemes").read_text()
        assert (Path(result["case_dir"]) / "0/U").read_text(encoding="utf-8") == ("fixed physical field\n" if scenario in {"solver_retry", "solver_mixed_retry", "solver_mixed_replan", "case_unparsed_native_error"} else "original physical field\n")
        assert (Path(result["case_dir"]) / "system/controlDict").read_text(encoding="utf-8") == (source / "system/controlDict").read_text(encoding="utf-8")
    if scenario == "case_no_commands" or unparsed:
        assert mesh_runs == []
    if uploaded:
        assert len(mesh_runs) == 1 and mesh_runs[0].count("checkMesh") == (3 if scenario == "custom_repeated" else 1)
    if scenario in {"mesh_retry", "case_regenerate", "external_failure", "case_external_switch"}:
        assert visits[2:6] == ["meshing", "reviewer", "meshing_planner", "meshing"]
    if scenario == "mesh_retry":
        assert len(mesh_runs) == 2 and visits.count("reviewer") == 1 and len(solver_runs) == 1
    if scenario == "case_regenerate":
        assert len(mesh_runs) == 2 and len(solver_runs) == 1
        assert "blockMesh" not in mesh_runs[0] and "blockMesh" in mesh_runs[1]
        assert sum(model is not None and model.__name__ == "MeshingPlan" for _, _, model in calls) == 2
        assert (Path(result["case_dir"]) / "system/blockMeshDict").read_text() == "existing mesh dictionary\n"
    if scenario in {"solver_retry", "solver_mixed_retry", "solver_mixed_replan", "case_unparsed_native_error"}:
        assert len(solver_runs) == 2 and visits.count("reviewer") == 1
    if not imported:
        field_prompt = next(prompt for folder, name, prompt in write_calls if folder == "0" and name == "U")
        assert "inlet" in field_prompt and "nFaces" in field_prompt
    assert result["solver_planned"] is True
    assert "prepared_inputs" not in result and "repair_impact" not in result
    assert "<actual_mesh>" not in result["user_requirement"]
    assert visits.count("solver_planner") == len(planned) == len(selections)
    assert len(materialized) == (0 if imported else 1)
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
