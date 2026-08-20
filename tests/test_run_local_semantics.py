"""Unit tests for local OpenFOAM execution and semantic validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from services import run_local  # noqa: E402
from services.allrun_commands import command_positions  # noqa: E402
from utils import check_foam_errors, run_command  # noqa: E402


def _make_case(tmp_path: Path, allrun: str) -> Path:
    case = tmp_path / "case"
    (case / "constant").mkdir(parents=True)
    (case / "system").mkdir()
    (case / "0").mkdir()
    (case / "Allrun").write_text(allrun, encoding="utf-8")
    return case


def test_check_foam_errors_rejects_failed_mesh_checks_with_end(tmp_path: Path) -> None:
    (tmp_path / "log.checkMesh").write_text(
        "Failed 2 mesh checks.\nEnd\n",
        encoding="utf-8",
    )

    errors = check_foam_errors(str(tmp_path))

    assert len(errors) == 1
    assert "failed mesh checks" in errors[0]["error_content"]


def test_check_foam_errors_accepts_normal_continuity_error_text_silently(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "log.icoFoam").write_text(
        "time step continuity errors : sum local = 0, global = 0, cumulative = 0\nEnd\n",
        encoding="utf-8",
    )

    assert check_foam_errors(str(tmp_path)) == []
    assert capsys.readouterr().out == ""


def test_preflight_requires_check_mesh_after_generated_mesh(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "# runApplication checkMesh\n"
        "runApplication icoFoam\n",
    )

    errors = run_local.validate_openfoam_case_preflight(str(case))

    assert len(errors) == 1
    assert "does not run checkMesh after mesh generation" in errors[0]["error_content"]


def test_preflight_rejects_non_coplanar_symmetry_plane_patch(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "runApplication checkMesh\n"
        "runApplication icoFoam\n",
    )
    (case / "system" / "blockMeshDict").write_text(
        "vertices\n(\n"
        "(0 0 0) (1 0 0) (1 1 0) (0 1 0)\n"
        "(0 0 1) (1 0 1) (1 1 1) (0 1 1)\n"
        ");\n"
        "boundary\n(\n"
        "thinSides\n{\n"
        "type symmetryPlane;\n"
        "faces\n(\n(0 3 7 4)\n(1 2 6 5)\n);\n"
        "}\n"
        ");\n",
        encoding="utf-8",
    )

    errors = run_local.validate_openfoam_case_preflight(str(case))

    assert len(errors) == 1
    assert "different planes" in errors[0]["error_content"]


def test_preflight_requires_simulation_type_in_momentum_transport(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "runApplication checkMesh\n"
        "runApplication icoFoam\n",
    )
    (case / "constant" / "momentumTransport.air").write_text(
        "FoamFile {}\nlaminar\n", encoding="utf-8"
    )

    errors = run_local.validate_openfoam_case_preflight(str(case))

    assert len(errors) == 1
    assert "missing the required Foundation OpenFOAM simulationType" in errors[0]["error_content"]


def test_preflight_does_not_accept_a_commented_simulation_type(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "runApplication checkMesh\n"
        "runApplication icoFoam\n",
    )
    (case / "constant" / "momentumTransport").write_text(
        "// simulationType laminar;\nFoamFile {}\n",
        encoding="utf-8",
    )

    errors = run_local.validate_openfoam_case_preflight(str(case))

    assert len(errors) == 1
    assert "missing the required Foundation OpenFOAM simulationType" in errors[0]["error_content"]


def test_preflight_accepts_multi_plane_generic_symmetry_patch(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "runApplication checkMesh\n"
        "runApplication icoFoam\n",
    )
    (case / "system" / "blockMeshDict").write_text(
        "vertices\n(\n"
        "(0 0 0) (1 0 0) (1 1 0) (0 1 0)\n"
        "(0 0 1) (1 0 1) (1 1 1) (0 1 1)\n"
        ");\n"
        "boundary\n(\n"
        "thinSides\n{\n"
        "type symmetry;\n"
        "faces\n(\n(0 3 7 4)\n(1 2 6 5)\n);\n"
        "}\n"
        ");\n",
        encoding="utf-8",
    )

    assert run_local.validate_openfoam_case_preflight(str(case)) == []


def test_command_detection_ignores_shell_assignment_values() -> None:
    assert command_positions("application=icoFoam\n", "icoFoam") == []


def test_preflight_requires_check_mesh_before_configured_solver(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "runApplication icoFoam\n"
        "runApplication checkMesh\n",
    )
    (case / "system" / "controlDict").write_text(
        "application icoFoam;\n",
        encoding="utf-8",
    )

    errors = run_local.validate_openfoam_case_preflight(str(case))

    assert len(errors) == 1
    assert "before the configured solver icoFoam" in errors[0]["error_content"]


def test_preflight_accepts_mesh_utility_application_after_mesh_check(
    tmp_path: Path,
) -> None:
    """A mesh tutorial can legitimately declare blockMesh as its application."""
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "runApplication checkMesh\n",
    )
    (case / "system" / "controlDict").write_text(
        "application blockMesh;\n",
        encoding="utf-8",
    )

    assert run_local.validate_openfoam_case_preflight(str(case)) == []


def test_preflight_orders_custom_mesh_check_before_solver(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication icoFoam\n"
        "runApplication checkMesh\n",
    )
    (case / "system" / "controlDict").write_text(
        "application icoFoam;\n",
        encoding="utf-8",
    )

    errors = run_local.validate_openfoam_case_preflight(str(case))

    assert len(errors) == 1
    assert "checkMesh must run before" in errors[0]["error_content"]


def test_postflight_validates_explicit_custom_mesh_check(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        "runApplication checkMesh\n"
        "runApplication icoFoam\n",
    )
    (case / "system" / "controlDict").write_text(
        "application icoFoam;\n",
        encoding="utf-8",
    )
    (case / "log.checkMesh").write_text(
        "Failed 1 mesh checks.\nEnd\n",
        encoding="utf-8",
    )
    # The postflight contract now also verifies that a declared solver
    # produced a complete log. Keep this fixture focused on the mesh-check
    # failure being asserted below.
    (case / "log.icoFoam").write_text("Time = 1\nEnd\n", encoding="utf-8")

    errors = run_local.validate_openfoam_case_postflight(str(case))

    assert len(errors) == 1
    assert "did not report 'Mesh OK'" in errors[0]["error_content"]


def test_postflight_accepts_direct_command_evidence_in_allrun_output(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path,
        "checkMesh\n"
        "icoFoam\n",
    )
    (case / "system" / "controlDict").write_text(
        "application icoFoam;\n",
        encoding="utf-8",
    )
    (case / "Allrun.out").write_text(
        "Checking geometry...\nMesh OK.\nTime = 1\nEnd\n",
        encoding="utf-8",
    )

    assert run_local.validate_openfoam_case_postflight(str(case)) == []


def test_flow_solver_does_not_require_lagrangian_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(
        tmp_path,
        "runApplication blockMesh\n"
        "runApplication checkMesh\n"
        "runApplication icoFoam\n",
    )
    (case / "system" / "controlDict").write_text(
        "application icoFoam;\n",
        encoding="utf-8",
    )

    def successful_cavity_run(*args, **kwargs):
        (case / "log.blockMesh").write_text("End\n", encoding="utf-8")
        (case / "log.checkMesh").write_text("Mesh OK.\nEnd\n", encoding="utf-8")
        (case / "log.icoFoam").write_text("Time = 1\nEnd\n", encoding="utf-8")
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(run_local, "run_command", successful_cavity_run)

    assert run_local.run_allrun_and_collect_errors(str(case)) == []


def test_nonzero_exit_code_fails_even_when_solver_log_ends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, "runApplication icoFoam\n")

    def failed_run(*args, **kwargs):
        (case / "log.icoFoam").write_text("End\n", encoding="utf-8")
        return {"returncode": 7, "timed_out": False}

    monkeypatch.setattr(run_local, "run_command", failed_run)

    errors = run_local.run_allrun_and_collect_errors(str(case))

    assert any("return code 7" in str(error) for error in errors)


def test_cleanup_run_artifacts_removes_only_disposable_solver_outputs(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case"
    case.mkdir()
    for directory in (
        "0",
        "0.5",
        "12",
        "processor0",
        "processor12",
        "postProcessing",
        "VTK",
        "system",
        "processorNotes",
    ):
        (case / directory).mkdir()
    (case / "log.icoFoam").write_text("old log", encoding="utf-8")
    (case / "Allrun.out").write_text("old output", encoding="utf-8")
    (case / "Allrun.err").write_text("old error", encoding="utf-8")
    (case / "keep.txt").write_text("input", encoding="utf-8")
    external = tmp_path / "external-log-target"
    external.write_text("must survive", encoding="utf-8")
    (case / "log.external").symlink_to(external)

    run_local._cleanup_run_artifacts(str(case))

    for removed in (
        "0.5",
        "12",
        "processor0",
        "processor12",
        "postProcessing",
        "VTK",
        "log.icoFoam",
        "Allrun.out",
        "Allrun.err",
        "log.external",
    ):
        assert not (case / removed).exists()
    for retained in ("0", "system", "processorNotes", "keep.txt"):
        assert (case / retained).exists()
    assert external.read_text(encoding="utf-8") == "must survive"


def test_run_command_exposes_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openfoam_root = tmp_path / "openfoam"
    (openfoam_root / "etc").mkdir(parents=True)
    (openfoam_root / "etc" / "bashrc").write_text("", encoding="utf-8")
    monkeypatch.setenv("WM_PROJECT_DIR", str(openfoam_root))

    script = tmp_path / "Allrun"
    script.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    result = run_command(
        str(script),
        str(tmp_path / "Allrun.out"),
        str(tmp_path / "Allrun.err"),
        str(tmp_path),
        5,
    )

    assert result == {"returncode": 7, "timed_out": False}


def test_run_command_exposes_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openfoam_root = tmp_path / "openfoam"
    (openfoam_root / "etc").mkdir(parents=True)
    (openfoam_root / "etc" / "bashrc").write_text("", encoding="utf-8")
    monkeypatch.setenv("WM_PROJECT_DIR", str(openfoam_root))

    script = tmp_path / "Allrun"
    script.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    result = run_command(
        str(script),
        str(tmp_path / "Allrun.out"),
        str(tmp_path / "Allrun.err"),
        str(tmp_path),
        0.05,
    )

    assert result["timed_out"] is True
    assert result["returncode"] != 0
