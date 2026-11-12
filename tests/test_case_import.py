"""Focused coverage for the unified existing-case agent workflow."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess
import gzip
import zipfile

import pytest


@pytest.mark.parametrize('with_properties', [False, True])
@pytest.mark.parametrize('time_dir', ['0', '0.0', '0e0'])
@pytest.mark.parametrize('compressed', [False, True])
def test_regional_case_context_keeps_inputs_without_bulk_mesh(tmp_path, with_properties, time_dir, compressed):
    from utils import load_case_files, read_case_foamfiles, scan_case_directory

    def put(relative, content):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(content.encode()) if path.suffix == '.gz' else content.encode())

    inputs = {
        'constant/regionProperties': 'regions (fluid (air) solid (wall));',
        'system/controlDict': 'application chtMultiRegionFoam;',
        'system/air/fvSolution': 'solvers { p_rgh { solver GAMG; } }',
        'system/wall/fvSchemes': 'ddtSchemes { default Euler; }',
        'constant/air/thermophysicalProperties': 'thermoType { type heRhoThermo; }',
        'constant/wall/thermophysicalProperties': 'thermoType { type heSolidThermo; }',
        f'{time_dir}/air/T': 'internalField uniform 300;',
        f'{time_dir}/wall/T': 'internalField uniform 400;',
    }
    if not with_properties:
        inputs.pop('constant/air/thermophysicalProperties')
        inputs.pop('constant/wall/thermophysicalProperties')
    boundaries = {
        'constant/air/polyMesh/boundary': 'airWall { type wall; }',
        'constant/wall/polyMesh/boundary': 'solidWall { type wall; }',
    }
    if compressed:
        inputs = {path + ('.gz' if path != 'system/controlDict' else ''): content
                  for path, content in inputs.items()}
        boundaries = {path + '.gz': content for path, content in boundaries.items()}
    for relative, content in {**inputs, **boundaries}.items():
        put(relative, content)
    put('constant/air/polyMesh/points', 'bulk mesh coordinates')
    put('constant/geometry/surface.stl', 'bulk geometry coordinates')
    put('constant/.foamagent/reconstruction/old/points', 'archived coordinates')
    put('1/air/T', 'old result field')
    put('processors2/0/air/T', 'decomposed field')
    directory = scan_case_directory(str(tmp_path))
    assert set(directory) == {str(Path(path).parent) for path in inputs}
    assert set(f'{f.folder_name}/{f.file_name}' for f in
               read_case_foamfiles(str(tmp_path), directory).list_foamfile) == set(inputs)
    loaded = load_case_files({'case_dir': str(tmp_path)})
    actual = {f'{f.folder_name}/{f.file_name}': f.content for f in loaded['foamfiles'].list_foamfile}
    assert actual == {**inputs, **boundaries}


@pytest.mark.parametrize('target,returned,existing,actual', [
    ('U', 'U.gz', ['U.gz'], 'U.gz'),
    ('U.gz', 'U.gz', ['U', 'U.gz'], 'U.gz'),
    ('U', 'U', ['U', 'U.gz'], 'U'),
])
def test_rewrite_preserves_selected_compression_and_native_file_precedence(
    tmp_path, target, returned, existing, actual,
):
    from services.input_writer import rewrite_files
    from utils import FoamPydantic, load_case_files

    field_dir = tmp_path / '0.0'
    field_dir.mkdir()
    for name in existing:
        text = 'internalField uniform ' + ('1;' if name == 'U' else '2;')
        (field_dir / name).write_bytes(gzip.compress(text.encode()) if name.endswith('.gz') else text.encode())
    initial = load_case_files({'case_dir': str(tmp_path)})
    current = initial['foamfiles'].list_foamfile
    assert len(current) == 1 and current[0].file_name == ('U' if 'U' in existing else 'U.gz')
    content = 'internalField uniform 3;'
    llm = SimpleNamespace(invoke=lambda *a, **kw: FoamPydantic(list_foamfile=[dict(
        folder_name='0.0', file_name=returned, content=content)]))
    result = rewrite_files(
        str(tmp_path), [], 'Correct the requested initial value',
        {'target_files': [dict(file='0.0/' + target, stage='solver', changes='Set the requested value')]},
        'Preserve all other physical inputs', llm_service=llm,
        foamfiles=initial['foamfiles'], dir_structure=initial['dir_structure'],
        omitted_files=initial['input_omissions'],
    )
    output = field_dir / actual
    assert output.is_file() and sorted(p.name for p in field_dir.iterdir()) == [actual]
    if actual.endswith('.gz'):
        assert output.read_bytes().startswith(b'\x1f\x8b')
        assert gzip.decompress(output.read_bytes()).decode() == content
    else:
        assert output.read_text() == content
    assert result['written_files'] == result['updated_files'] == ['0.0/' + actual]
    assert result['rewrite_plan'] == {'target_files': []} and not result['error_logs']
    assert result['dir_structure'] == {'0.0': [actual]}
    assert [f.file_name for f in result['foamfiles'].list_foamfile] == [actual]
    assert load_case_files({'case_dir': str(tmp_path)})['foamfiles'].list_foamfile[0].content == content


@pytest.mark.parametrize('mode', ['sequential_dependency', 'parallel_no_context'])
def test_initial_writer_creates_compressed_fields_with_actual_receipts(tmp_path, mode):
    from services.input_writer import initial_write
    from services.stage_plan import pending_file_creates

    task = dict(folder_name='0.0', file_name='U.gz', stage='solver', action='create', changes='Create velocity')
    content = 'FoamFile { class volVectorField; object U; }\ninternalField uniform (1 0 0);\n'
    result = initial_write(str(tmp_path), [task], 'Use the requested velocity', '', 'icoFoam',
                           generation_mode=mode, llm_service=SimpleNamespace(invoke=lambda *a: content))
    assert gzip.decompress((tmp_path / '0.0/U.gz').read_bytes()).decode() == content
    assert result['written_files'] == ['0.0/U.gz']
    assert result['dir_structure'] == {'0.0': ['U.gz']}
    assert not pending_file_creates([dict(task, file_name='U')], result['written_files'])


def test_case_file_variants_reject_symlinks_before_writing(tmp_path):
    from utils import save_case_text

    outside = tmp_path / 'outside'
    outside.write_text('preserve external input')
    field = tmp_path / '0/U'
    field.parent.mkdir()
    field.write_text('original input')
    (field.parent / 'U.gz').symlink_to(outside)
    with pytest.raises(ValueError, match='symbolic links'):
        save_case_text(field, 'replacement input')
    assert outside.read_text() == 'preserve external input' and field.read_text() == 'original input'


def test_truncated_gzip_context_is_reported_and_explicit_repair_restores_it(tmp_path, capsys):
    from services.case_import import _read_text
    from services.input_writer import rewrite_files
    from utils import FoamPydantic, load_case_files

    content = 'FoamFile { class volVectorField; object U; }\ninternalField uniform (1 0 0);\n'
    field = tmp_path / '0.0/U.gz'
    field.parent.mkdir()
    truncated = gzip.compress(content.encode())[:-4]
    field.write_bytes(truncated)
    with pytest.raises(EOFError):
        gzip.decompress(truncated)
    valid = tmp_path / 'constant/transportProperties.gz'
    valid.parent.mkdir()
    valid_bytes = gzip.compress(b'nu 1e-05;\n')
    valid.write_bytes(valid_bytes)
    initial = load_case_files({'case_dir': str(tmp_path)})
    assert [(f.folder_name, f.file_name) for f in initial['foamfiles'].list_foamfile] == [
        ('constant', 'transportProperties.gz')]
    assert '0.0/U.gz' in initial['input_omissions']
    assert _read_text(field) == ''
    llm = SimpleNamespace(invoke=lambda *a, **kw: FoamPydantic(list_foamfile=[dict(
        folder_name='0.0', file_name='U.gz', content=content)]))
    result = rewrite_files(
        str(tmp_path), [], 'Replace the corrupt field from the supplied physical definition',
        {'target_files': [dict(file='0.0/U.gz', stage='solver', changes='Restore supplied field definition')]},
        'Keep the supplied physical definition', llm_service=llm,
        foamfiles=initial['foamfiles'], dir_structure=initial['dir_structure'],
        omitted_files=initial['input_omissions'],
    )
    assert gzip.decompress(field.read_bytes()).decode() == content
    assert result['written_files'] == result['updated_files'] == ['0.0/U.gz']
    assert not result['error_logs'] and result['rewrite_plan'] == {'target_files': []}
    assert valid.read_bytes() == valid_bytes
    assert any(f.file_name == 'U.gz' and f.content == content for f in
               load_case_files({'case_dir': str(tmp_path)})['foamfiles'].list_foamfile)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import foambench_main  # noqa: E402
from models import ExistingCasePlan  # noqa: E402
from nodes.input_writer_node import input_writer_node  # noqa: E402
from nodes.local_runner_node import local_runner_node  # noqa: E402
from router_func import (  # noqa: E402
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
def native_dictionary(monkeypatch):
    from services import solver_execution
    monkeypatch.setattr(solver_execution, 'run_openfoam_utility', lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, 'application icoFoam;\n', ''))


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


def _solver_operations():
    return [dict(stage="solver", command="#!/bin/sh\nicoFoam\n", purpose="Execute the imported case's solver",
                 executor="script", required_inputs=[], generated_inputs=[])]


def _make_case(root: Path, target: str = "foundation-v10") -> Path:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "0").mkdir()
    if target == "esi-v2006":
        header = "/* OpenFOAM: The Open Source CFD Toolbox\nWebsite: https://www.openfoam.com\nVersion: v2006\n*/\n"
    else:
        header = "/* OpenFOAM: The Open Source CFD Toolbox\nWebsite: https://openfoam.org\nVersion: 10\n*/\n"
    (root / "system" / "controlDict").write_text(
        header + "application icoFoam;\nstartTime 0;\nendTime 1;\n",
        encoding="utf-8",
    )
    (root / "system" / "fvSchemes").write_text("ddtSchemes {}\n", encoding="utf-8")
    (root / "system" / "fvSolution").write_text("solvers {}\n", encoding="utf-8")
    (root / "constant" / "polyMesh").mkdir()
    (root / "0" / "U").write_text("internalField uniform (0 0 0);\n", encoding="utf-8")
    (root / "Allrun").write_text("#!/bin/sh\nicoFoam\n", encoding="utf-8")
    helper = root / "prepare-geometry"
    helper.write_text("#!/bin/sh\necho prepared\n", encoding="utf-8")
    helper.chmod(0o755)
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
    llm = _FakeLLM(ExistingCasePlan(operations=[]))
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
@pytest.mark.parametrize("archive_input", [False, True])
def test_import_materialises_original_and_work_and_detects_target(tmp_path: Path, target: str, archive_input: bool) -> None:
    source = _make_case(tmp_path / "source", target)
    original = (source / "system" / "controlDict").read_bytes()
    if archive_input:
        archive = tmp_path / "case.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            for path in source.rglob("*"):
                zipped.write(path, path.relative_to(source))
        source = archive
    output = tmp_path / "task"

    manifest = import_case(source, output)

    assert manifest.platform == target
    assert manifest.application == "icoFoam"
    assert (output / "original" / "system" / "controlDict").is_file()
    assert (output / "work" / "system" / "controlDict").is_file()
    assert not os.access(output / "original" / "system" / "controlDict", os.W_OK)
    assert (output / "report" / "case_context.json").is_file()
    assert (tmp_path / "source" / "system" / "controlDict").read_bytes() == original
    assert (output / "work" / "system" / "controlDict").stat().st_mode & 0o200
    for directory in ("original", "work"):
        assert (output / directory / "prepare-geometry").stat().st_mode & 0o111


@pytest.mark.parametrize("target", ["foundation-v10", "esi-v2006"])
def test_import_node_loads_resources_for_the_detected_target_when_planning_is_needed(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    imported_module = importlib.import_module("services.preparation")
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

    result = imported_module.prepare_imported_case(
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
        {'operations': [], 'status': 'ready', 'requires_meshing': False, 'requires_input_writer': True, 'target_files': [{'file': 'system/controlDict', 'changes': 'Set endTime to 5', 'stage': 'solver'}], 'reason': 'Change the requested end time.'}
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
        {"file": "system/controlDict", "changes": "Set endTime to 5", "stage": "solver"}
    ]


def test_missing_files_and_unknown_physics_use_structured_planner_outcomes() -> None:
    missing_file_plan = ExistingCasePlan.model_validate(
        {'operations': [], 'status': 'ready', 'requires_meshing': False, 'requires_input_writer': True, 'target_files': [], 'reason': 'Create the missing dictionary.'}
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
        {'operations': [], 'status': 'failed', 'reason': 'The inlet condition cannot be inferred.', 'requires_meshing': False, 'requires_input_writer': False}
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
                {"file": "system/controlDict", "changes": "Set endTime to 5", "stage": "solver"}
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
    state = {'pending_solver_operations': [], 'solver_operations': _solver_operations(), 'solver_review_history': [], 'solver_script_dirty': False, 'case_origin': 'imported', 'case_dir': str(case), 'input_writer_mode': 'rewrite', 'review_analysis': 'Apply the planned changes to fix the reported setup error.', 'rewrite_plan': {'target_files': [{'file': 'system/controlDict', 'changes': 'Set endTime to 5', 'stage': 'solver'}, {'file': 'system/fvSolution', 'changes': 'Update solver settings'}]}, 'error_logs': [{'file': 'Allrun', 'error_content': 'setup error'}], 'user_requirement': DEFAULT_IMPORTED_REQUIREMENT, 'foamfiles': read_case_foamfiles(str(case), directory), 'dir_structure': directory, 'case_solver': 'icoFoam', 'config': SimpleNamespace(openfoam_target='foundation-v10'), 'llm_service': _FakeLLM(response), 'loop_count': 1}

    result = input_writer_node(state)

    assert result["updated_files"] == ["system/controlDict", "system/fvSolution"]
    assert len(result["file_diffs"]) == 2
    assert route_after_input_writer({**state, **result, "requires_hpc": False}) == "local_runner"


@pytest.mark.parametrize(
    ("target", "target_version"),
    [("", "foundation-v10"), ("foundation-v10", "foundation-v10"), ("esi-v2006", "esi-v2006")],
)
def test_runner_uses_shared_cleanup_and_selected_runtime(
    tmp_path: Path, monkeypatch, target: str, target_version: str
) -> None:
    runner_module = importlib.import_module("nodes.local_runner_node")
    observed = {}

    def fake_run(case_dir, timeout, **kwargs):
        observed.update(case_dir=case_dir, timeout=timeout, **kwargs)
        return []

    monkeypatch.setattr(runner_module, "run_allrun_and_collect_errors", fake_run)
    (tmp_path / 'Allrun').write_text('#!/bin/sh\nicoFoam\n')
    state = {'pending_solver_operations': [], 'solver_operations': _solver_operations(), 'solver_review_history': [], 'solver_script_dirty': False, 'case_origin': 'imported', 'case_dir': str(tmp_path), 'config': SimpleNamespace(max_time_limit=9, openfoam_target=target)}

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
    state = {'pending_solver_operations': [], 'solver_operations': _solver_operations(), 'solver_review_history': [], 'solver_script_dirty': False, 'case_origin': 'imported', 'case_dir': str(work), 'config': SimpleNamespace(max_time_limit=9, openfoam_target='foundation-v10')}

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


def test_runner_reports_missing_native_solver_completion(
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
        return {'returncode': 0, 'timed_out': False}

    monkeypatch.setattr(run_service, "run_command", successful_run)

    errors = run_service.run_allrun_and_collect_errors(str(case))
    assert errors
    assert any('No normal completion evidence' in error['error_content'] for error in errors)
    assert len(calls) == 1
    assert calls[0][1]["fail_fast"] is True


def test_custom_mesh_conversion_requires_polymesh_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    mesh_service = importlib.import_module("services.mesh")
    source = tmp_path / "uploaded.msh"
    source.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    case = tmp_path / "case"
    monkeypatch.setattr(mesh_service, "run_openfoam_utility", lambda args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout='', stderr=''))

    result = mesh_service.copy_custom_mesh(
        str(source),
        str(case),
        timeout=3600,
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
    assert (case / "Allrun").read_text(encoding="utf-8") == "#!/bin/sh\nsimpleFoam\n"


def test_invalid_import_keeps_previous_output(tmp_path):
    from models import CaseImportError
    from services.output_safety import prepare_output_directory

    output = prepare_output_directory(tmp_path / "output", overwrite=False)
    previous = output / "work/0/U"
    previous.parent.mkdir(parents=True)
    previous.write_text("previous valid field")
    archive = tmp_path / "broken.zip"
    archive.write_text("invalid archive")

    with pytest.raises(CaseImportError):
        import_case(archive, output, overwrite=True)

    assert previous.read_text() == "previous valid field"
