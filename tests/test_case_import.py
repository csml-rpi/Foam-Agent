"""Focused coverage for the unified existing-case agent workflow."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import foambench_main  # noqa: E402
import main  # noqa: E402
from models import ExistingCasePlan  # noqa: E402
from nodes.input_writer_node import input_writer_node  # noqa: E402
from nodes.local_runner_node import local_runner_node  # noqa: E402
from router_func import (  # noqa: E402
    route_after_planner,
    route_after_input_writer,
    route_after_runner,
)
from services.case_import import import_case  # noqa: E402
from services.plan import (  # noqa: E402
    DEFAULT_IMPORTED_REQUIREMENT,
    plan_imported_case,
)
from services.input_writer import rewrite_files  # noqa: E402
from utils import FoamPydantic, FoamfilePydantic, read_case_foamfiles, scan_case_directory  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_writable_tmp_tree(tmp_path: Path):
    """Allow pytest to remove immutable original/ trees after each test."""
    yield
    paths = [tmp_path, *tmp_path.rglob("*")]
    for path in paths:
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode | 0o700)
        except OSError:
            pass


def _make_case(root: Path, target: str = "foundation-v10") -> Path:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "0").mkdir()
    if target == "esi-v2006":
        header = "Website: https://www.openfoam.com\nVersion: v2006\n"
    else:
        header = "Website: https://openfoam.org\nVersion: 10\n"
    (root / "system" / "controlDict").write_text(
        header + "application icoFoam;\nstartTime 0;\nendTime 1;\n",
        encoding="utf-8",
    )
    (root / "system" / "fvSchemes").write_text("ddtSchemes {}\n", encoding="utf-8")
    (root / "system" / "fvSolution").write_text("solvers {}\n", encoding="utf-8")
    (root / "constant" / "polyMesh").mkdir()
    (root / "0" / "U").write_text("internalField uniform (0 0 0);\n", encoding="utf-8")
    (root / "Allrun").write_text("#!/bin/sh\nicoFoam\n", encoding="utf-8")
    return root


class _FakeLLM:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def invoke(self, user_prompt, system_prompt, pydantic_obj=None):
        self.calls.append((user_prompt, system_prompt, pydantic_obj))
        return self.response


def test_replanning_uses_current_files_instead_of_import_snapshot(tmp_path):
    case = _make_case(tmp_path / "case")
    llm = _FakeLLM(ExistingCasePlan())
    directory = scan_case_directory(str(case))
    result = plan_imported_case({
        "config": SimpleNamespace(openfoam_target="foundation-v10"),
        "case_dir": str(case),
        "case_context": {
            "platform": "foundation-v10", "application": "obsoleteSolver",
            "has_allrun": True, "issues": [],
        },
        "user_requirement_explicit": False,
        "loop_count": 1,
        "error_logs": ["Reconsider the setup"],
        "dir_structure": directory,
        "foamfiles": read_case_foamfiles(str(case), directory),
        "llm_service": llm,
    })
    assert result["workflow_status"] == "running"
    prompt = llm.calls[0][0]
    assert "application icoFoam" in prompt
    assert "fvSolution" in prompt
    assert "obsoleteSolver" not in prompt


def test_rewrite_reads_solver_from_current_control_dict(tmp_path):
    from services.input_writer import rewrite_files

    case = _make_case(tmp_path / "case")
    llm = _FakeLLM(FoamPydantic(list_foamfile=[]))
    rewrite_files(
        case_dir=str(case), llm_service=llm,
        foamfiles=read_case_foamfiles(str(case)), error_logs=[],
        review_analysis="Update settings",
        rewrite_plan={"target_files": [{"file": "system/controlDict", "changes": "Update settings"}]},
        user_requirement="Update settings", openfoam_fork="foundation",
        case_solver="obsoleteSolver",
    )
    prompts = llm.calls[0][:2]
    assert "<case_solver>icoFoam</case_solver>" in prompts[0]
    assert "obsoleteSolver" not in "".join(prompts)


@pytest.mark.parametrize("target", ["foundation-v10", "esi-v2006"])
def test_import_materialises_original_and_work_and_detects_target(tmp_path: Path, target: str) -> None:
    source = _make_case(tmp_path / "source", target)
    output = tmp_path / "task"

    manifest = import_case(source, output)

    assert manifest.platform == target
    assert manifest.application == "icoFoam"
    assert (output / "original" / "system" / "controlDict").is_file()
    assert (output / "work" / "system" / "controlDict").is_file()
    assert not os.access(output / "original" / "system" / "controlDict", os.W_OK)
    assert (output / "report" / "case_context.json").is_file()


def test_case_only_complete_case_goes_directly_to_runner_without_llm() -> None:
    llm = _FakeLLM(response=AssertionError("LLM must not be called"))
    state = {
        "case_context": {
            "platform": "foundation-v10",
            "has_allrun": True,
            "issues": [],
            "candidate_cases": [],
        },
        "case_dir": "/task/work",
        "user_requirement": DEFAULT_IMPORTED_REQUIREMENT,
        "user_requirement_explicit": False,
        "llm_service": llm,
    }

    result = plan_imported_case(state)

    assert result["workflow_status"] == "running"
    assert not result["requires_input_writer"] and not result["requires_meshing"]
    assert llm.calls == []
    assert route_after_planner({**state, **result, "case_origin": "imported"}) == "local_runner"


@pytest.mark.parametrize("target", ["foundation-v10", "esi-v2006"])
def test_import_node_loads_resources_for_the_detected_target_when_planning_is_needed(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    imported_module = importlib.import_module("nodes.imported_case_node")
    source = _make_case(tmp_path / "source", target)
    observed = []
    monkeypatch.setattr(imported_module, "setup_logging", lambda _path: None)
    monkeypatch.setattr(
        imported_module,
        "load_agent_resources",
        lambda config: (observed.append(config.openfoam_target) or object(), {"target": target}),
    )
    config = SimpleNamespace(
        case_dir=str(tmp_path / "task"),
        overwrite_case_dir=False,
        openfoam_target="",
    )

    result = imported_module.case_import_node(
        {
            "config": config,
            "case_import_path": str(source),
            "case_import_subdir": None,
            "user_requirement_explicit": True,
            "requires_visualization": False,
        }
    )

    assert observed == [target]
    assert config.openfoam_target == target
    assert result["case_context"]["platform"] == target
    assert result["case_stats"] == {"target": target}


def test_explicit_change_creates_targeted_writer_then_run_plan() -> None:
    plan = ExistingCasePlan.model_validate(
        {'status': 'ready', 'requires_meshing': False, 'requires_input_writer': True, 'target_files': [{'file': 'system/controlDict', 'changes': 'Set endTime to 5'}], 'reason': 'Change the requested end time.'}
    )
    llm = _FakeLLM(plan)
    state = {
        "case_context": {
            "platform": "foundation-v10",
            "has_allrun": True,
            "issues": [],
            "candidate_cases": [],
        },
        "case_dir": "/task/work",
        "user_requirement": "Set endTime to 5, then run.",
        "user_requirement_explicit": True,
        "llm_service": llm,
    }

    result = plan_imported_case(state)

    assert result["requires_input_writer"] and not result["requires_meshing"]
    assert result["input_writer_mode"] == "rewrite"
    assert result["rewrite_plan"]["target_files"] == [
        {"file": "system/controlDict", "changes": "Set endTime to 5"}
    ]


def test_missing_files_and_unknown_physics_use_structured_planner_outcomes() -> None:
    missing_file_plan = ExistingCasePlan.model_validate(
        {'status': 'ready', 'requires_meshing': False, 'requires_input_writer': True, 'target_files': [], 'reason': 'Create the missing dictionary.'}
    )
    state = {
        "case_context": {
            "platform": "foundation-v10",
            "has_allrun": False,
            "issues": ["system/fvSolution is missing"],
            "candidate_cases": [],
        },
        "case_dir": "/task/work",
        "user_requirement": DEFAULT_IMPORTED_REQUIREMENT,
        "user_requirement_explicit": False,
        "llm_service": _FakeLLM(missing_file_plan),
    }
    repair = plan_imported_case(state)
    assert repair["workflow_status"] == "failed"
    assert not repair["requires_input_writer"]
    assert "target_files" in repair["termination_reason"]

    failed_plan = ExistingCasePlan.model_validate(
        {'status': 'failed', 'reason': 'The inlet condition cannot be inferred.', 'requires_meshing': False, 'requires_input_writer': False}
    )
    failed = plan_imported_case({**state, "llm_service": _FakeLLM(failed_plan)})
    assert failed["workflow_status"] == "failed"
    assert failed["termination_reason"] == "The inlet condition cannot be inferred."


def test_modify_existing_applies_only_planned_target_files(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    control_before = (case / "system" / "controlDict").read_text(encoding="utf-8")
    solution_before = (case / "system" / "fvSolution").read_text(encoding="utf-8")
    response = FoamPydantic(
        list_foamfile=[
            FoamfilePydantic(
                folder_name="system",
                file_name="controlDict",
                content=control_before.replace("endTime 1", "endTime 5"),
            ),
            FoamfilePydantic(
                folder_name="system",
                file_name="fvSolution",
                content="solvers { changed yes; }\n",
            ),
            FoamfilePydantic(folder_name="0", file_name="p", content="internalField uniform 0;\n"),
        ]
    )

    result = rewrite_files(
        case_dir=str(case),
        error_logs=[],
        review_analysis="Only update the requested end time.",
        rewrite_plan={
            "target_files": [
                {"file": "system/controlDict", "changes": "Set endTime to 5"}
            ]
        },
        user_requirement="Set endTime to 5.",
        foamfiles=read_case_foamfiles(str(case), scan_case_directory(str(case))),
        dir_structure=scan_case_directory(str(case)),
        llm_service=_FakeLLM(response),
    )

    assert "endTime 5" in (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert (case / "system" / "fvSolution").read_text(encoding="utf-8") == solution_before
    assert not (case / "0" / "p").exists()
    assert result["updated_files"] == ["system/controlDict"]


def test_rewrite_uses_planned_files_and_records_actual_diffs(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    control_before = (case / "system" / "controlDict").read_text(encoding="utf-8")
    solution_before = (case / "system" / "fvSolution").read_text(encoding="utf-8")
    response = FoamPydantic(
        list_foamfile=[
            FoamfilePydantic(
                folder_name="system",
                file_name="controlDict",
                content=control_before.replace("endTime 1", "endTime 5"),
            ),
            FoamfilePydantic(
                folder_name="system",
                file_name="fvSolution",
                content=solution_before.replace("solvers {}", "solvers { p PCG; }"),
            ),
        ]
    )
    directory = scan_case_directory(str(case))
    state = {'case_origin': 'imported', 'case_dir': str(case), 'input_writer_mode': 'rewrite', 'review_analysis': 'Apply the planned changes to fix the reported setup error.', 'rewrite_plan': {'target_files': [{'file': 'system/controlDict', 'changes': 'Set endTime to 5'}, {'file': 'system/fvSolution', 'changes': 'Update solver settings'}]}, 'error_logs': [{'file': 'Allrun', 'error_content': 'setup error'}], 'user_requirement': DEFAULT_IMPORTED_REQUIREMENT, 'foamfiles': read_case_foamfiles(str(case), directory), 'dir_structure': directory, 'case_solver': 'icoFoam', 'config': SimpleNamespace(openfoam_target='foundation-v10'), 'llm_service': _FakeLLM(response), 'loop_count': 1}

    result = input_writer_node(state)

    assert result["updated_files"] == ["system/controlDict", "system/fvSolution"]
    assert len(result["file_diffs"]) == 2
    assert route_after_input_writer({**state, **result, "requires_hpc": False}) == "local_runner"


@pytest.mark.parametrize(
    ("target", "target_version"),
    [("", "foundation-v10"), ("foundation-v10", "foundation-v10"), ("esi-v2006", "esi-v2006")],
)
def test_runner_uses_shared_cleanup_and_selected_runtime(
    monkeypatch, target: str, target_version: str
) -> None:
    runner_module = importlib.import_module("nodes.local_runner_node")
    observed = {}

    def fake_run(case_dir, timeout, **kwargs):
        observed.update(case_dir=case_dir, timeout=timeout, **kwargs)
        return []

    monkeypatch.setattr(runner_module, "run_allrun_and_collect_errors", fake_run)
    state = {'case_origin': 'imported', 'case_dir': '/task/work', 'config': SimpleNamespace(max_time_limit=9, openfoam_target=target)}

    result = local_runner_node(state)

    assert observed["openfoam_target"] == target_version
    assert result["workflow_status"] == "success"


def test_failed_run_returns_errors_without_restoring_old_results(
    tmp_path: Path, monkeypatch
) -> None:
    runner_module = importlib.import_module("nodes.local_runner_node")
    original = _make_case(tmp_path / "original")
    work = _make_case(tmp_path / "work")
    (original / "10").mkdir()
    (original / "10" / "U").write_text("restart", encoding="utf-8")
    (work / "10").mkdir()
    (work / "10" / "U").write_text("restart", encoding="utf-8")

    def failed_run(case_dir, _timeout, **_kwargs):
        import shutil

        shutil.rmtree(Path(case_dir) / "10")
        (Path(case_dir) / "20").mkdir()
        return [{"file": "Allrun", "error_content": "solver failed"}]

    monkeypatch.setattr(runner_module, "run_allrun_and_collect_errors", failed_run)
    state = {'case_origin': 'imported', 'case_dir': str(work), 'config': SimpleNamespace(max_time_limit=9, openfoam_target='foundation-v10')}

    result = local_runner_node(state)

    assert result["error_logs"] == [{"file": "Allrun", "error_content": "solver failed"}]
    assert not (work / "10").exists()
    assert (original / "10" / "U").read_text(encoding="utf-8") == "restart"
    assert (work / "20").is_dir()


def test_cli_accepts_case_and_prompt_together(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "foambench_main.py",
            "--case_path",
            "./case",
            "--prompt_path",
            "./requirements.txt",
            "--openfoam_target",
            "esi-v2006",
        ],
    )

    args = foambench_main.parse_args()

    assert args.case_path == "./case"
    assert args.prompt_path == "./requirements.txt"
    assert args.openfoam_target == "esi-v2006"


def test_graph_routes_success_to_visualization_or_result() -> None:
    base = {"case_origin": "imported", "error_logs": [], "llm_service": _FakeLLM()}
    assert route_after_runner({**base, "requires_visualization": False}) == "__end__"
    assert route_after_runner({**base, "requires_visualization": True}) == "visualization"


def test_langgraph_case_only_path_skips_writer_and_returns_shared_result(monkeypatch) -> None:
    writer_calls = []

    def fake_import(_state):
        return {
            "case_origin": "imported",
            "case_dir": "/task/work",
            "case_context": {
                "platform": "foundation-v10",
                "has_allrun": True,
                "issues": [],
                "candidate_cases": [],
            },
            "workflow_status": "planning",
            "error_logs": [],
        }

    def fake_runner(state):
        return {'workflow_status': 'success', 'error_logs': []}

    monkeypatch.setattr(main, "case_import_node", fake_import)
    monkeypatch.setattr(main, "local_runner_node", fake_runner)
    monkeypatch.setattr(
        main,
        "input_writer_node",
        lambda state: writer_calls.append(state) or state,
    )
    config = SimpleNamespace(openfoam_target="foundation-v10", recursion_limit=20)
    state = main.initialize_state(
        DEFAULT_IMPORTED_REQUIREMENT,
        config,
        workflow_mode="imported_case",
        case_import_path="/input/case",
        requires_visualization=False,
        user_requirement_explicit=False,
    )

    result = main.create_foam_agent_graph().compile().invoke(
        state, config={"recursion_limit": 20}
    )

    assert result["workflow_status"] == "success"
    assert writer_calls == []


@pytest.mark.parametrize(
    ('mesh', 'writer', 'hpc', 'expected'),
    [
        (False, False, False, ['local_runner']),
        (False, True, False, ['input_writer', 'local_runner']),
        (True, False, False, ['meshing', 'local_runner']),
        (True, True, False, ['meshing', 'input_writer', 'local_runner']),
        (True, True, True, ['meshing', 'input_writer', 'hpc_runner']),
    ],
)
def test_imported_graph_routes_without_action_bookkeeping(monkeypatch, mesh, writer, hpc, expected):
    visited = []
    monkeypatch.setattr(main, 'case_import_node', lambda state: {})
    monkeypatch.setattr(main, 'planner_node', lambda state: {
        'workflow_status': 'running',
        'requires_meshing': mesh,
        'requires_input_writer': writer,
        'requires_hpc': hpc,
    })
    def node(name):
        def execute(state):
            visited.append(name)
            return {'error_logs': []}
        return execute
    for name in ('meshing', 'input_writer', 'local_runner', 'hpc_runner'):
        monkeypatch.setattr(main, name + '_node', node(name))
    main.create_foam_agent_graph().compile().invoke({
        'workflow_mode': 'imported_case', 'case_origin': 'imported',
        'requires_visualization': False, 'error_logs': [],
    })
    assert visited == expected


def test_imported_graph_repairs_then_runs_without_repeating_mesh(monkeypatch):
    visited = []
    monkeypatch.setattr(main, 'case_import_node', lambda state: {})
    monkeypatch.setattr(main, 'planner_node', lambda state: {
        'workflow_status': 'running', 'requires_meshing': True,
        'requires_input_writer': False, 'requires_hpc': False,
    })
    monkeypatch.setattr(main, 'meshing_node', lambda state: visited.append('mesh') or {'error_logs': []})
    def runner(state):
        visited.append('run')
        return {'error_logs': ['solver error'] if visited.count('run') == 1 else []}
    monkeypatch.setattr(main, 'local_runner_node', runner)
    monkeypatch.setattr(main, 'reviewer_node', lambda state: visited.append('review') or {
        'loop_count': 1, 'input_writer_mode': 'rewrite',
    })
    monkeypatch.setattr(main, 'input_writer_node', lambda state: visited.append('write') or {'error_logs': []})
    main.create_foam_agent_graph().compile().invoke({
        'workflow_mode': 'imported_case', 'case_origin': 'imported',
        'requires_visualization': False, 'error_logs': [],
        'config': SimpleNamespace(max_loop=3),
    })
    assert visited == ['mesh', 'run', 'review', 'write', 'run']


def test_runner_has_no_allrun_preflight_or_completion_marker_gate(
    tmp_path: Path, monkeypatch
) -> None:
    run_service = importlib.import_module("services.run_local")
    case = tmp_path / "case"
    case.mkdir()
    (case / "Allrun").write_text(
        "#!/bin/sh\nwmake libso\ncustomSolver --user-option\n",
        encoding="utf-8",
    )
    calls = []

    def successful_run(*args, **kwargs):
        calls.append((args, kwargs))
        (case / "log.customSolver").write_text(
            "Failed 2 mesh checks\nsolver output without a completion marker\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, timed_out=False)

    monkeypatch.setattr(run_service, "run_command", successful_run)

    assert run_service.run_allrun_and_collect_errors(str(case)) == []
    assert len(calls) == 1


def test_custom_mesh_conversion_requires_polymesh_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    mesh_service = importlib.import_module("services.mesh")
    source = tmp_path / "uploaded.msh"
    source.write_text("mesh", encoding="utf-8")
    case = tmp_path / "case"
    monkeypatch.setattr(mesh_service, "run_openfoam_utility", lambda *_args, **_kwargs: None)

    result = mesh_service.copy_custom_mesh(
        str(source),
        "Use the uploaded mesh.",
        str(case),
        llm_service=_FakeLLM("application customSolver;"),
    )

    assert result["error_logs"] == ["polyMesh directory not created"]
    assert not (case / "constant" / "polyMesh").exists()
    assert result["mesh_commands"] == []


def test_allrun_generation_uses_command_help(tmp_path: Path, monkeypatch) -> None:
    writer_service = importlib.import_module("services.input_writer")
    case = tmp_path / "case"
    case.mkdir()
    database = tmp_path / "database"
    (database / "raw").mkdir(parents=True)
    (database / "raw" / "openfoam_commands.txt").write_text("simpleFoam\n", encoding="utf-8")
    calls = []
    retrievals = []
    config = SimpleNamespace(openfoam_target="esi-v2006")
    monkeypatch.setattr(writer_service, "database_path_for_config", lambda selected: database)

    def invoke(user_prompt, system_prompt, pydantic_obj=None):
        calls.append((user_prompt, system_prompt))
        if pydantic_obj is not None:
            return pydantic_obj(commands=["simpleFoam"])
        return "```sh\n#!/bin/sh\nsimpleFoam\n```"

    def retrieve(database_name, command, topk, *, config):
        retrievals.append((database_name, command, topk, config))
        return [{"full_content": "simpleFoam command help"}]

    monkeypatch.setattr(writer_service, "retrieve_faiss", retrieve)
    result = writer_service.build_allrun(
        case_dir=str(case), database_path=str(database), searchdocs=2,
        dir_structure={"system": ["controlDict"]},
        case_info="case solver: simpleFoam", allrun_reference="",
        mesh_type="custom_mesh", mesh_commands=[],
        llm_service=SimpleNamespace(invoke=invoke), config=config,
        openfoam_fork="esi-v2006",
    )
    assert result["commands"] == ["simpleFoam"]
    assert len(calls) == 2
    assert retrievals == [("openfoam_command_help", "simpleFoam", 2, config)]
    assert "simpleFoam command help" in calls[1][1]
    assert all("ESI/OpenCFD OpenFOAM v2006" in system for _, system in calls)
    assert (case / "Allrun").read_text(encoding="utf-8") == "#!/bin/sh\nsimpleFoam"
