"""Offline contracts for deterministic visualization execution."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from services.visualization import (  # noqa: E402
    ensure_foam_file,
    generate_deterministic_pyvista_script,
    run_pyvista_script,
    visualize_case,
)
from services import visualization as visualization_service  # noqa: E402


def test_ensure_foam_file_creates_case_named_marker(tmp_path: Path) -> None:
    case_dir = tmp_path / "cavity"
    case_dir.mkdir()

    foam_file = ensure_foam_file(str(case_dir))

    assert foam_file == "cavity.foam"
    assert (case_dir / foam_file).is_file()


def test_visualization_runner_requires_the_declared_artifact(tmp_path: Path) -> None:
    ok, image, errors = run_pyvista_script(
        str(tmp_path),
        "from pathlib import Path\nPath('result.png').write_bytes(b'png')\n",
        expected_png="result.png",
        timeout_s=5,
    )

    assert ok is True
    assert image == str(tmp_path / "result.png")
    assert errors == []
    assert (tmp_path / "visualization.py").is_file()


def test_visualization_runner_rejects_success_without_expected_artifact(
    tmp_path: Path,
) -> None:
    ok, image, errors = run_pyvista_script(
        str(tmp_path),
        "print('completed without writing an image')\n",
        expected_png="missing.png",
        timeout_s=5,
    )

    assert ok is False
    assert image == ""
    assert "expected PNG was not created" in errors[0]


def test_visualization_runner_does_not_accept_a_stale_expected_artifact(
    tmp_path: Path,
) -> None:
    stale_output = tmp_path / "visualization.png"
    stale_output.write_bytes(b"old image")

    ok, image, errors = run_pyvista_script(
        str(tmp_path),
        "print('completed without writing an image')\n",
        expected_png="visualization.png",
        timeout_s=5,
    )

    assert ok is False
    assert image == ""
    assert "expected PNG was not created" in errors[0]
    assert not stale_output.exists()


def test_visualization_runner_rejects_an_artifact_path_outside_the_case(
    tmp_path: Path,
) -> None:
    external_output = tmp_path.parent / "outside.png"

    ok, image, errors = run_pyvista_script(
        str(tmp_path),
        "print('not run because the artifact path is invalid')\n",
        expected_png="../outside.png",
        timeout_s=5,
    )

    assert ok is False
    assert image == ""
    assert "must remain inside the case directory" in errors[0]
    assert not external_output.exists()


def test_deterministic_pyvista_template_has_a_fixed_output_contract() -> None:
    script = generate_deterministic_pyvista_script(
        foam_file="case.foam",
        output_png="visualization.png",
        field_preference="p",
    )

    assert "pv.OpenFOAMReader" in script
    assert "out_png" in script
    assert "'visualization.png'" in script
    assert "'p'" in script
    assert "pv.start_xvfb()" in script
    assert "Headless visualization requires Xvfb" in script
    assert "plotter.show(" not in script


def test_visualize_case_returns_a_complete_failure_contract_without_case_dir() -> None:
    result = visualize_case(None, "Plot the velocity.")

    assert result["plot_outputs"] == []
    assert result["pyvista_visualization"] == {"success": False, "error": "Missing case_dir"}
    assert result["visualization_summary"]["pyvista_success"] is False


def test_visualize_case_prefers_the_deterministic_renderer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = str(tmp_path / "visualization.png")
    monkeypatch.setattr(visualization_service, "ensure_foam_file", lambda _case_dir: "case.foam")
    monkeypatch.setattr(
        visualization_service,
        "run_pyvista_script",
        lambda *_args, **_kwargs: (True, output, []),
    )

    result = visualize_case(str(tmp_path), "Plot pressure.")

    assert result["plot_outputs"] == [output]
    assert result["visualization_summary"]["used"] == "deterministic_template"
    assert result["plot_configs"][0]["field_name"] == "U"


def test_visualize_case_can_disable_the_llm_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Imported cases must not execute an LLM-generated visualization script."""
    monkeypatch.setattr(visualization_service, "ensure_foam_file", lambda _case_dir: "case.foam")
    monkeypatch.setattr(
        visualization_service,
        "run_pyvista_script",
        lambda *_args, **_kwargs: (False, "", ["deterministic renderer unavailable"]),
    )

    def llm_must_not_run(*_args, **_kwargs):
        raise AssertionError("LLM fallback must be disabled")

    monkeypatch.setattr(visualization_service, "generate_pyvista_script", llm_must_not_run)

    result = visualize_case(
        str(tmp_path),
        "Plot pressure.",
        allow_llm_fallback=False,
    )

    assert result["pyvista_visualization"]["success"] is False
    assert "LLM fallback is disabled" in result["pyvista_visualization"]["error"]
