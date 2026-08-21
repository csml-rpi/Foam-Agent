"""Regression tests for the thin meshing-node adapter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_gmsh_route_preserves_case_context_and_injected_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The adapter must use the service signature without replacing planner output."""
    meshing_node = importlib.import_module("nodes.meshing_node")
    injected_llm = object()
    captured: dict[str, object] = {}
    expected = {"mesh_info": {"mesh_file_type": "gmsh"}, "error_logs": []}

    def fake_handle_gmsh_mesh(
        user_requirement: str,
        case_dir: str,
        max_loop: int,
        *,
        llm_service: object,
    ) -> dict[str, object]:
        captured.update(
            user_requirement=user_requirement,
            case_dir=case_dir,
            max_loop=max_loop,
            llm_service=llm_service,
        )
        return expected

    monkeypatch.setattr(meshing_node, "service_handle_gmsh_mesh", fake_handle_gmsh_mesh)
    state = {
        "config": SimpleNamespace(max_loop=4),
        "user_requirement": "Create a Gmsh channel mesh.",
        "case_dir": str(tmp_path / "planned-case"),
        "llm_service": injected_llm,
        "mesh_type": "gmsh_mesh",
    }

    assert meshing_node.meshing_node(state) == expected
    assert captured == {
        "user_requirement": "Create a Gmsh channel mesh.",
        "case_dir": str(tmp_path / "planned-case"),
        "max_loop": 4,
        "llm_service": injected_llm,
    }


def test_gmsh_service_returns_the_same_complete_contract_after_refactoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mesh orchestration keeps one stable result shape for graph consumers."""
    mesh = importlib.import_module("services.mesh")
    case_dir = tmp_path / "case"
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"

    class FakeLLM:
        def invoke(self, *_args: object, pydantic_obj: object = None, **_kwargs: object):
            if pydantic_obj is None:
                return "inlet,outlet"
            return pydantic_obj(
                python_code="print('generated')",
                mesh_type="3D",
                geometry_type="channel",
            )

    def fake_run_python(case_dir_arg: str, _python_file: str, _python_code: str) -> str:
        Path(case_dir_arg, "geometry.msh").write_text("mesh", encoding="utf-8")
        return ""

    def fake_convert(case_dir_arg: str, _requirement: str, _llm: object) -> str:
        target = Path(case_dir_arg) / "constant" / "polyMesh" / "boundary"
        target.parent.mkdir(parents=True)
        target.write_text("inlet\n{\n}\noutlet\n{\n}\n", encoding="utf-8")
        return str(target)

    monkeypatch.setattr(mesh, "_run_gmsh_python", fake_run_python)
    monkeypatch.setattr(mesh, "_convert_gmsh_mesh", fake_convert)
    monkeypatch.setattr(mesh, "run_checkmesh_and_correct", lambda *_args, **_kwargs: (True, False, ""))
    monkeypatch.setattr(mesh, "_update_boundary_file", lambda *_args, **_kwargs: None)

    result = mesh.handle_gmsh_mesh(
        "Create a channel with inlet and outlet.",
        str(case_dir),
        max_loop=1,
        llm_service=FakeLLM(),
    )

    assert result["custom_mesh_used"] is True
    assert result["mesh_file_destination"] == str(case_dir / "geometry.msh")
    assert result["mesh_info"]["mesh_description"] == "GMSH generated channel mesh"
    assert result["error_logs"] == []
    assert boundary_file.exists()


def test_gmsh_boundary_validation_accepts_extra_valid_patches(
    tmp_path: Path,
) -> None:
    """Requested patch names are required, but OpenFOAM may have extra patches."""
    mesh = importlib.import_module("services.mesh")
    boundary_file = tmp_path / "boundary"
    boundary_file.write_text(
        "inlet\n{\n}\noutlet\n{\n}\nwalls\n{\n}\nfrontAndBack\n{\n}\n",
        encoding="utf-8",
    )

    mismatch, corrected = mesh._correct_boundary_mismatch(
        str(boundary_file),
        ["inlet", "outlet"],
        user_requirement="Channel with inlet and outlet",
        python_code="print('mesh')",
        llm_client=object(),
    )

    assert mismatch is False
    assert corrected is None


def test_gmsh_attempt_cleanup_removes_stale_mesh_artifacts(tmp_path: Path) -> None:
    """A retry cannot convert a mesh produced by an earlier failed attempt."""
    mesh = importlib.import_module("services.mesh")
    msh_file = tmp_path / "geometry.msh"
    boundary_file = tmp_path / "constant" / "polyMesh" / "boundary"
    msh_file.write_text("stale mesh", encoding="utf-8")
    boundary_file.parent.mkdir(parents=True)
    boundary_file.write_text("stale boundary", encoding="utf-8")

    mesh._clear_gmsh_attempt_outputs(str(tmp_path))

    assert not msh_file.exists()
    assert not boundary_file.parent.exists()
