"""One core feature test: real LangGraph/services, mocked external LLM and tools."""

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
    names = ("main", "nodes.solver_planner_node", "nodes.imported_case_node", "nodes.local_runner_node",
             "services.meshing", "services.mesh", "services.input_writer", "services.run_local", "services.plan")
    result = SimpleNamespace(**{name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in names})
    monkeypatch.setattr(result.plan, "setup_logging", lambda *_: None)
    monkeypatch.setattr(result.imported_case_node, "setup_logging", lambda *_: None)
    yield result
    # The existing importer makes original/ immutable; allow pytest teardown.
    for path in tmp_path.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)


@pytest.mark.parametrize("scenario", [
    "standard", "gmsh", "custom", "case_poly", "mesh_retry",
    "case_regenerate", "solver_retry", "solver_mixed_retry", "target_mismatch",
])
def test_mesh_before_solver_workflow(scenario, modules, monkeypatch, tmp_path):
    m = modules
    planned, prepared, materialized = [], [], []
    original_planner = m.main.solver_planner_node
    original_prepare = m.plan.prepare_case_context
    original_materialize = m.plan.prepare_output_directory

    def observe_planning(state):
        info = state["mesh_info"]
        assert (Path(info["poly_mesh_path"]) / "boundary").is_file()
        assert [patch["name"] for patch in info["patches"]] == ["inlet", "wall"]
        planned.append(state)
        return original_planner(state)

    def observe_preparation(*args, **kwargs):
        prepared.append(kwargs)
        return original_prepare(*args, **kwargs)

    def observe_materialization(*args, **kwargs):
        materialized.append(args)
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(m.main, "solver_planner_node", observe_planning)
    monkeypatch.setattr(m.plan, "prepare_case_context", observe_preparation)
    monkeypatch.setattr(m.plan, "prepare_output_directory", observe_materialization)

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
        monkeypatch.setattr(m.imported_case_node, "load_agent_resources",
                            lambda _: pytest.fail("A target mismatch must stop before loading resources"))
        state = m.main.initialize_state(
            "Run the imported case", config, workflow_mode="imported_case",
            case_import_path=str(source), requires_visualization=False,
        )
        result = m.main.create_foam_agent_graph().compile().invoke(state)
        assert result["workflow_status"] == "failed"
        assert result["termination_reason"] == "case_target_mismatch"
        assert not meshes and not planned and not prepared and not materialized
        return

    imported = scenario.startswith("case_") or scenario.startswith("solver_")
    uploaded = scenario == "custom"
    has_poly = imported
    solver_script = "#!/bin/sh\n. ${WM_PROJECT_DIR}/bin/tools/RunFunctions\nrestore0Dir\nrunApplication icoFoam -case .\n"
    original_script = solver_script.replace("restore0Dir", "runApplication blockMesh\nrunApplication checkMesh\nrestore0Dir")
    source = tmp_path / "source"
    if imported:
        _put(source, "system/controlDict", "// Website: https://openfoam.org\n// Version: 10\napplication icoFoam;\nendTime 1;\n")
        _put(source, "system/fvSchemes", "ddtSchemes {}\n")
        _put(source, "system/fvSolution", "solvers {}\n")
        _put(source, "constant/transportProperties", "nu 1e-5;\n")
        _put(source, "0/U", "original physical field\n")
        _put(source, "Allrun", original_script)
        if has_poly:
            _poly_mesh(source)
        if scenario == "case_regenerate":
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
                kind = "gmsh_mesh" if scenario == "gmsh" else "custom_mesh" if uploaded or has_poly and not repair else "standard_mesh"
                tasks = []
                case_root = tmp_path / "output" / ("work" if imported else "")
                if kind == "standard_mesh" and not (case_root / "system/blockMeshDict").is_file():
                    tasks = [{"folder_name": "system", "file_name": "blockMeshDict"}]
                commands = ["blockMesh -dict system/blockMeshDict", "checkMesh"] if kind == "standard_mesh" else ["checkMesh"] if has_poly and not uploaded and kind != "gmsh_mesh" else []
                return pydantic_obj(mesh_type=kind, reuse_mesh=has_poly and not repair and not uploaded,
                                    mesh_subtasks=tasks, mesh_commands=commands,
                                    solver_allrun=solver_script if imported else "")
            if name == "ExistingCasePlan":
                return pydantic_obj(status="ready")
            if name == "OpenFOAMPlanModel":
                assert "planned separately by meshing_planner" in system
                return pydantic_obj(subtasks=[
                    {"folder_name": "system", "file_name": "controlDict"},
                    {"folder_name": "0", "file_name": "U"},
                    {"folder_name": "system", "file_name": "blockMeshDict"},
                    {"folder_name": "", "file_name": "Allrun"},
                ])
            if name == "GMSHPythonCode":
                return pydantic_obj(python_code="# mock external Gmsh execution", mesh_type="3D", geometry_type="channel")
            if name == "CommandsPydantic":
                assert "only the solver" in system
                return pydantic_obj(commands=["icoFoam -case ."])
            if name == "RewritePlan":
                if scenario == "case_regenerate":
                    return pydantic_obj(target_files=[])
                target = "system/blockMeshDict" if scenario == "mesh_retry" else "0/U" if scenario == "solver_retry" else "Allmesh"
                targets = [{"file": target, "changes": "Apply the diagnosed repair"}]
                if scenario == "solver_mixed_retry":
                    targets.append({"file": "0/U", "changes": "Apply the field correction"})
                return pydantic_obj(target_files=targets)
            if name == "FoamPydantic":
                plan = ast.literal_eval(re.search(r"<rewrite_plan>(.*?)</rewrite_plan>", user, re.S)[1])
                files = []
                for target in plan["target_files"]:
                    path = Path(target["file"])
                    content = "fixed mesh dictionary\n" if path.name == "blockMeshDict" else "fixed physical field\n" if path.name == "U" else "#!/bin/sh\nblockMesh -dict system/blockMeshDict\ncheckMesh\n"
                    files.append({"folder_name": "" if path.parent == Path(".") else path.parent.as_posix(), "file_name": path.name, "content": content})
                return pydantic_obj(list_foamfile=files)
            if "underlying issues" in system:
                if scenario == "case_regenerate":
                    return "Regenerate the mesh using the existing blockMeshDict. <repair_domain>mesh</repair_domain>"
                domain = "mesh" if scenario in {"mesh_retry", "solver_mixed_retry"} else "solver"
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
                return "application icoFoam;\nendTime 1;\n" if file == "controlDict" else f"generated {file}\n"
            if "minimal controlDict" in system or "basic controlDict" in system:
                return "application icoFoam;\nendTime 1;\n"
            if "extract boundary names" in system:
                return "inlet,wall"
            if "polyMesh boundary file" in system:
                return "2\n(\ninlet { type patch; nFaces 1; startFace 0; }\nwall { type wall; nFaces 2; startFace 1; }\n)\n"
            raise AssertionError(f"Unexpected LLM call: {system[:120]}")

    llm = LLM()
    monkeypatch.setattr(m.main, "load_agent_resources", lambda _: (llm, {}))
    monkeypatch.setattr(m.imported_case_node, "load_agent_resources", lambda _: (llm, {}))
    from config import Config
    config = Config(case_dir=str(tmp_path / "output"), database_path=str(tmp_path / "db"), max_loop=5)
    _put(tmp_path / "db", "foundation-v10/raw/openfoam_commands.txt", "blockMesh\ncheckMesh\nicoFoam\n")
    monkeypatch.setattr(m.input_writer, "retrieve_faiss", lambda *_, **__: [{"full_content": "command help"}])

    monkeypatch.setattr(m.plan, "parse_requirement_to_case_info", lambda *_, **__: {
        "case_name": "case", "case_domain": "incompressible", "case_category": "laminar", "case_solver": "icoFoam",
    })
    monkeypatch.setattr(m.plan, "retrieve_references", lambda **_: ("reference", {}, "", original_script, {}))

    def run_script(script, stdout, stderr, case_dir, *_, **__):
        case = Path(case_dir)
        content = Path(script).read_text(encoding="utf-8")
        Path(stdout).write_text("", encoding="utf-8")
        Path(stderr).write_text("", encoding="utf-8")
        if Path(script).name == "Allmesh":
            mesh_runs.append(content)
            assert "icoFoam" not in content and "restore0Dir" not in content
            if scenario == "mesh_retry" and len(mesh_runs) == 1:
                Path(stderr).write_text("bad mesh dictionary", encoding="utf-8")
                return {"returncode": 1}
            if scenario == "case_regenerate" and "blockMesh" not in content:
                Path(stderr).write_text("existing mesh requires regeneration", encoding="utf-8")
                return {"returncode": 1}
            if "blockMesh" in content:
                if scenario == "mesh_retry":
                    assert "fixed" in (case / "system/blockMeshDict").read_text(encoding="utf-8")
                _poly_mesh(case)
        else:
            solver_runs.append(content)
            assert all(token not in content for token in ("blockMesh", "checkMesh", "gmshToFoam", "Allmesh"))
            assert (case / "constant/polyMesh/boundary").is_file()
            assert "restore0Dir" in content and "icoFoam -case ." in content
            if scenario in {"solver_retry", "solver_mixed_retry"} and len(solver_runs) == 1:
                Path(stderr).write_text("solver detected an input error", encoding="utf-8")
                return {"returncode": 1}
        return {"returncode": 0}
    monkeypatch.setattr(m.meshing, "run_command", run_script)
    monkeypatch.setattr(m.run_local, "run_command", run_script)

    conversions, gmsh_calls = [], []
    def convert(args, **kwargs):
        conversions.append(args)
        if args[0] == "gmshToFoam":
            _poly_mesh(Path(kwargs["working_dir"]))
        return SimpleNamespace(stdout="Mesh OK")
    monkeypatch.setattr(m.mesh, "run_openfoam_utility", convert)
    def gmsh(args, **kwargs):
        gmsh_calls.append(kwargs)
        _put(Path(kwargs["cwd"]), "geometry.msh", "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        return SimpleNamespace(stderr="")
    monkeypatch.setattr(m.mesh.subprocess, "run", gmsh)

    state = m.main.initialize_state(
        "Run the case", config, str(custom_path) if uploaded else None,
        workflow_mode="imported_case" if imported else "prompt",
        case_import_path=str(source) if imported else None,
        user_requirement_explicit=not imported,
        requires_visualization=False,
    )
    graph = m.main.create_foam_agent_graph().compile()
    visits, result = [], None
    for mode, update in graph.stream(state, stream_mode=["updates", "values"]):
        if mode == "updates":
            visits.extend(update)
        else:
            result = update
    assert result["workflow_status"] == "success" and not result["error_logs"]
    assert result["mesh_info"]["patches"] == [
        {"name": "inlet", "type": "patch", "n_faces": 1},
        {"name": "wall", "type": "wall", "n_faces": 2},
    ]
    assert visits.index("meshing_planner") < visits.index("meshing") < visits.index("solver_planner") < visits.index("input_writer") < visits.index("local_runner")
    assert visits.count("meshing_planner") == visits.count("meshing") == (2 if scenario == "solver_mixed_retry" else 1)
    assert result["solver_allrun"] is None
    assert len(gmsh_calls) == (1 if scenario == "gmsh" else 0)
    assert conversions == ([["gmshToFoam", "geometry.msh"], ["checkMesh"]] if scenario == "gmsh" else [["gmshToFoam", "geometry.msh"]] if uploaded else [])
    if imported:
        assert (source / "Allrun").read_text(encoding="utf-8") == original_script
        assert (tmp_path / "output/original/Allrun").read_text(encoding="utf-8") == original_script
        assert result["case_context"]["mesh_state"] == "existing-polyMesh"
        assert result["case_import_manifest"].mesh_state == "existing-polyMesh"
        assert (Path(result["case_dir"]) / "0/U").read_text(encoding="utf-8") == ("fixed physical field\n" if scenario in {"solver_retry", "solver_mixed_retry"} else "original physical field\n")
        assert (Path(result["case_dir"]) / "system/controlDict").read_text(encoding="utf-8") == (source / "system/controlDict").read_text(encoding="utf-8")
    if uploaded:
        assert len(mesh_runs) == 1 and mesh_runs[0].count("checkMesh") == 1
    if scenario == "mesh_retry":
        assert len(mesh_runs) == 2 and visits.count("reviewer") == 0 and len(solver_runs) == 1
    if scenario == "case_regenerate":
        assert len(mesh_runs) == 2 and len(solver_runs) == 1
        assert "blockMesh" not in mesh_runs[0] and "blockMesh" in mesh_runs[1]
        assert sum(model is not None and model.__name__ == "MeshingPlan" for _, _, model in calls) == 2
        assert (Path(result["case_dir"]) / "system/blockMeshDict").read_text() == "existing mesh dictionary\n"
    if scenario in {"solver_retry", "solver_mixed_retry"}:
        assert len(solver_runs) == 2 and visits.count("reviewer") == 1
    if not imported:
        field_prompt = next(prompt for folder, name, prompt in write_calls if folder == "0" and name == "U")
        assert "inlet" in field_prompt and "n_faces" in field_prompt
    assert result["solver_planned"] is True
    assert visits.count("solver_planner") == len(planned) == 1
    assert len(prepared) == len(materialized) == (0 if imported else 1)
    planning_prompts = [
        user for user, _, model in calls
        if model and model.__name__ in {"OpenFOAMPlanModel", "ExistingCasePlan"}
    ]
    if not imported:
        assert len(planning_prompts) == 1
    for prompt in planning_prompts:
        assert "<actual_mesh>" in prompt and "inlet" in prompt and "n_faces" in prompt
    if scenario == "solver_mixed_retry":
        assert result["rewrite_plan"]["target_files"] == [
            {"file": "0/U", "changes": "Apply the field correction"},
        ]
