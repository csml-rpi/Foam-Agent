"""Focused regression tests for InputWriter compatibility and Allrun contracts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import FoamPydantic, FoamfilePydantic  # noqa: E402
from services import input_writer as writer_service  # noqa: E402


class _FakeLLM:
    def __init__(self, text_response: str = "FoamFile { version 2.0; }") -> None:
        self.text_response = text_response
        self.calls = []

    def invoke(self, user_prompt, system_prompt, pydantic_obj=None):
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "pydantic_obj": pydantic_obj,
            }
        )
        if pydantic_obj is not None:
            return pydantic_obj(commands=["blockMesh", "simpleFoam"])
        return self.text_response


def test_initial_node_uses_selected_solver_fork_and_requirement(monkeypatch, tmp_path):
    writer_node = importlib.import_module("nodes.input_writer_node")
    captured = {}

    def fake_initial_write(**kwargs):
        captured["initial"] = kwargs
        return {
            "dir_structure": {"system": ["blockMeshDict"]},
            "foamfiles": FoamPydantic(list_foamfile=[]),
        }

    def fake_build_allrun(**kwargs):
        captured["allrun"] = kwargs
        return {"allrun_path": str(tmp_path / "Allrun"), "allrun_script": "", "commands": []}

    monkeypatch.setattr(writer_node, "initial_write", fake_initial_write)
    monkeypatch.setattr(writer_node, "build_allrun", fake_build_allrun)
    monkeypatch.setattr(writer_node, "convert_case_to_esi_if_needed", lambda *_: None)
    monkeypatch.setattr(writer_node, "scan_case_directory", lambda *_: {"system": ["blockMeshDict"]})
    monkeypatch.setattr(writer_node, "read_case_foamfiles", lambda *_: FoamPydantic(list_foamfile=[]))

    class Config:
        input_writer_generation_mode = "sequential_dependency"
        reuse_generated_dir = ""
        openfoam_fork = "foundation"
        database_path = str(tmp_path / "database")
        searchdocs = 3

    requirement = "Generate and validate a transient flow case."
    state = {
        "input_writer_mode": "initial",
        "config": Config(),
        "case_dir": str(tmp_path / "case"),
        "subtasks": [],
        "user_requirement": requirement,
        "tutorial_reference": "",
        "case_solver": "simpleFoam",
        "case_stats": {"case_solver": ["icoFoam", "simpleFoam"]},
        "similar_case_advice": None,
        "case_info": "case solver: simpleFoam",
        "allrun_reference": "",
        "mesh_type": "standard_mesh",
        "mesh_commands": [],
    }

    writer_node.input_writer_node(state)

    assert captured["initial"]["case_solver"] == "simpleFoam"
    assert captured["initial"]["openfoam_fork"] == "foundation"
    assert captured["allrun"]["user_requirement"] == requirement


def test_rewrite_node_passes_selected_solver_and_fork(monkeypatch, tmp_path):
    writer_node = importlib.import_module("nodes.input_writer_node")
    captured = {}

    def fake_rewrite_files(**kwargs):
        captured.update(kwargs)
        return {"dir_structure": {}, "foamfiles": FoamPydantic(list_foamfile=[])}

    monkeypatch.setattr(writer_node, "rewrite_files", fake_rewrite_files)
    monkeypatch.setattr(writer_node, "convert_case_to_esi_if_needed", lambda *_: None)
    monkeypatch.setattr(writer_node, "scan_case_directory", lambda *_: {})
    monkeypatch.setattr(writer_node, "read_case_foamfiles", lambda *_: FoamPydantic(list_foamfile=[]))

    class Config:
        openfoam_fork = "foundation"

    writer_node.input_writer_node(
        {
            "input_writer_mode": "rewrite",
            "config": Config(),
            "case_dir": str(tmp_path / "case"),
            "case_solver": "simpleFoam",
            "review_analysis": "Use the correct dictionary.",
            "rewrite_plan": {"target_files": []},
            "error_logs": [],
            "user_requirement": "Use transient flow.",
            "foamfiles": FoamPydantic(list_foamfile=[]),
            "dir_structure": {},
        }
    )

    assert captured["openfoam_fork"] == "foundation"
    assert captured["case_solver"] == "simpleFoam"


def test_rewrite_keeps_solver_and_foundation_contract(monkeypatch, tmp_path):
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    original = FoamfilePydantic(
        file_name="controlDict",
        folder_name="system",
        content="FoamFile { version 2.0; }",
    )

    class RewriteLLM(_FakeLLM):
        def invoke(self, user_prompt, system_prompt, pydantic_obj=None):
            self.calls.append(
                {
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "pydantic_obj": pydantic_obj,
                }
            )
            return FoamPydantic(
                list_foamfile=[
                    FoamfilePydantic(
                        file_name="controlDict",
                        folder_name="system",
                        content="FoamFile { version 2.0; }\napplication simpleFoam;",
                    )
                ]
            )

    fake_llm = RewriteLLM()
    monkeypatch.setattr(writer_service, "global_llm_service", fake_llm)

    result = writer_service.rewrite_files(
        case_dir=str(case_dir),
        error_logs=["missing application"],
        review_analysis="Set the solver application using Foundation conventions.",
        rewrite_plan={"target_files": [{"file": "system/controlDict"}]},
        user_requirement="Use steady flow.",
        foamfiles=FoamPydantic(list_foamfile=[original]),
        dir_structure={"system": ["controlDict"]},
        openfoam_fork="foundation",
        case_solver="simpleFoam",
    )

    system_prompt = fake_llm.calls[0]["system_prompt"]
    user_prompt = fake_llm.calls[0]["user_prompt"]
    assert "Foundation OpenFOAM v10" in system_prompt
    assert "selected solver is simpleFoam" in system_prompt
    assert "<openfoam_fork>foundation</openfoam_fork>" in user_prompt
    assert "<case_solver>simpleFoam</case_solver>" in user_prompt
    assert result["updated_files"] == ["system/controlDict"]
    assert (case_dir / "system" / "controlDict").read_text(encoding="utf-8").endswith(
        "application simpleFoam;"
    )


@pytest.mark.parametrize(
    ("folder_name", "file_name"),
    [
        ("../outside", "controlDict"),
        ("system", "../outside"),
        ("C:/outside", "controlDict"),
    ],
)
def test_rewrite_rejects_llm_paths_that_escape_case_root(
    monkeypatch,
    tmp_path,
    folder_name: str,
    file_name: str,
):
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    class UnsafeRewriteLLM:
        def invoke(self, *_args, **_kwargs):
            return FoamPydantic(
                list_foamfile=[
                    FoamfilePydantic(
                        file_name=file_name,
                        folder_name=folder_name,
                        content="attempted escape",
                    )
                ]
            )

    monkeypatch.setattr(writer_service, "global_llm_service", UnsafeRewriteLLM())

    with pytest.raises(ValueError, match="unsafe rewrite path"):
        writer_service.rewrite_files(
            case_dir=str(case_dir),
            error_logs=["dictionary error"],
            review_analysis="Repair only system/controlDict.",
            rewrite_plan={"target_files": [{"file": "system/controlDict"}]},
            user_requirement="Keep the case inside its output directory.",
            foamfiles=FoamPydantic(list_foamfile=[]),
            dir_structure={},
        )

    assert not (tmp_path / "outside").exists()
    assert not (case_dir / "outside").exists()


def test_build_allrun_command_selection_sees_requirement_and_normalizes_order(
    monkeypatch, tmp_path
):
    database = tmp_path / "database"
    raw = database / "raw"
    raw.mkdir(parents=True)
    (raw / "openfoam_commands.txt").write_text(
        "blockMesh\ncheckMesh\nsimpleFoam\n", encoding="utf-8"
    )

    fake_llm = _FakeLLM(
        """```sh
#!/bin/sh
. $WM_PROJECT_DIR/bin/tools/RunFunctions
runApplication blockMesh
runApplication simpleFoam
runApplication checkMesh
```"""
    )
    monkeypatch.setattr(writer_service, "global_llm_service", fake_llm)
    monkeypatch.setattr(
        writer_service,
        "retrieve_faiss",
        lambda *_args, **_kwargs: [{"full_content": "command help"}],
    )

    requirement = "Generate the mesh, check its quality, then run the flow solver."
    result = writer_service.build_allrun(
        case_dir=str(tmp_path / "case"),
        database_path=str(database),
        searchdocs=1,
        dir_structure={"system": ["blockMeshDict", "controlDict"]},
        case_info="case solver: simpleFoam",
        allrun_reference="",
        mesh_type="standard_mesh",
        mesh_commands=[],
        user_requirement=requirement,
    )

    assert requirement in fake_llm.calls[0]["user_prompt"]
    lines = result["allrun_script"].splitlines()
    assert lines[0] == "#!/bin/sh"
    assert "sh" not in {line.strip() for line in lines}
    block_index = lines.index("runApplication blockMesh")
    check_index = lines.index("runApplication checkMesh")
    solver_index = lines.index("runApplication simpleFoam")
    assert block_index < check_index < solver_index
    assert sum("checkMesh" in line for line in lines) == 1


def test_custom_mesh_does_not_regenerate_uploaded_mesh():
    script = """#!/bin/sh
. $WM_PROJECT_DIR/bin/tools/RunFunctions
runApplication blockMesh
runApplication simpleFoam
"""

    normalized = writer_service._ensure_checkmesh_before_solver(
        script,
        case_info="case solver: simpleFoam",
        dir_structure={"system": ["blockMeshDict", "controlDict"]},
        mesh_type="custom_mesh",
        mesh_commands=["checkMesh"],
    )

    assert "blockMesh" not in normalized
    assert normalized.index("checkMesh") < normalized.index("simpleFoam")


def test_mesh_utility_application_runs_checkmesh_after_its_mesh_command():
    """Mesh tutorials use blockMesh as the application, not a flow solver."""
    normalized = writer_service._ensure_checkmesh_before_solver(
        "#!/bin/sh\n"
        ". $WM_PROJECT_DIR/bin/tools/RunFunctions\n"
        "runApplication blockMesh\n",
        case_info="case solver: blockMesh",
        dir_structure={"system": ["blockMeshDict", "controlDict"]},
        mesh_type="standard_mesh",
        mesh_commands=[],
    )

    lines = normalized.splitlines()
    assert lines.count("runApplication blockMesh") == 1
    assert lines.count("runApplication checkMesh") == 1
    assert lines.index("runApplication blockMesh") < lines.index("runApplication checkMesh")


def test_direct_allrun_commands_are_wrapped_for_retained_logs():
    script = """#!/bin/sh
blockMesh
simpleFoam
"""

    normalized = writer_service._ensure_checkmesh_before_solver(
        script,
        case_info="case solver: simpleFoam",
        dir_structure={"system": ["blockMeshDict", "controlDict"]},
        mesh_type="standard_mesh",
        mesh_commands=[],
    )

    assert ". $WM_PROJECT_DIR/bin/tools/RunFunctions" in normalized
    assert "runApplication blockMesh" in normalized
    assert "runApplication checkMesh" in normalized
    assert "runApplication simpleFoam" in normalized


def test_initial_write_passes_the_injected_llm_to_allrun(monkeypatch, tmp_path):
    """All generation steps must share the workflow's configured LLM client."""
    captured = {}

    def fake_build_allrun(*_args, **kwargs):
        captured.update(kwargs)
        return {"allrun_script": "#!/bin/sh\n"}

    injected_llm = _FakeLLM()
    monkeypatch.setattr(writer_service, "build_allrun", fake_build_allrun)

    writer_service.initial_write(
        case_dir=str(tmp_path / "case"),
        subtasks=[],
        user_requirement="Generate a case.",
        tutorial_reference="",
        case_solver="icoFoam",
        case_info="case solver: icoFoam",
        database_path=str(tmp_path / "database"),
        llm_service=injected_llm,
    )

    assert captured["llm_service"] is injected_llm


def test_initial_write_rejects_duplicate_generation_targets(tmp_path):
    with pytest.raises(ValueError, match="Duplicate generated subtask target: system/controlDict"):
        writer_service.initial_write(
            case_dir=str(tmp_path / "case"),
            subtasks=[
                {"folder_name": "system", "file_name": "controlDict"},
                {"folder_name": "system", "file_name": "controlDict"},
            ],
            user_requirement="Generate a case.",
            tutorial_reference="",
            case_solver="icoFoam",
            llm_service=_FakeLLM(),
        )


def test_initial_write_records_allrun_as_a_case_relative_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        writer_service,
        "build_allrun",
        lambda *_args, **_kwargs: {"allrun_script": "#!/bin/sh\n"},
    )

    result = writer_service.initial_write(
        case_dir=str(tmp_path / "case"),
        subtasks=[],
        user_requirement="Generate a case.",
        tutorial_reference="",
        case_solver="icoFoam",
        case_info="case solver: icoFoam",
        database_path=str(tmp_path / "database"),
        llm_service=_FakeLLM(),
    )

    allrun = next(
        item
        for item in result["foamfiles"].list_foamfile
        if item.file_name == "Allrun"
    )
    assert allrun.folder_name == ""
