"""Workflow-level contracts for the protected ``--case_path`` graph branch."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import main as workflow_main  # noqa: E402
from config import Config  # noqa: E402
from nodes import imported_case_node  # noqa: E402


def _write_case(root: Path, *, wrong_object_name: bool = False) -> Path:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    object_name = "wrongName" if wrong_object_name else "controlDict"
    (root / "system" / "controlDict").write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM: The Open Source CFD Toolbox
| Website:  https://openfoam.org
| Version:  10
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format ascii;
    class dictionary;
    object %s;
}
application icoFoam;
startTime 0;
endTime 2;
"""
        % object_name,
        encoding="utf-8",
    )
    (root / "system" / "blockMeshDict").write_text(
        "FoamFile { object blockMeshDict; }\n", encoding="utf-8"
    )
    return root


def _import_config(output: Path, *, max_loop: int = 1) -> Config:
    config = Config()
    config.case_dir = str(output)
    config.max_loop = max_loop
    config.max_time_limit = 1
    config.recursion_limit = 20
    return config


def test_imported_case_uses_graph_branch_without_planner_or_llm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _write_case(tmp_path / "source")
    output = tmp_path / "output"
    calls: list[str] = []

    def planner_must_not_run(_state):
        raise AssertionError("Imported cases must not enter Planner")

    def llm_must_not_be_constructed(_config):
        raise AssertionError("Imported cases must not construct an LLM client")

    def controlled_success(*_args, **_kwargs):
        calls.append("run")
        return []

    monkeypatch.setattr(workflow_main, "planner_node", planner_must_not_run)
    monkeypatch.setattr(workflow_main, "LLMService", llm_must_not_be_constructed)
    monkeypatch.setattr(imported_case_node, "execute_imported_case", controlled_success)

    result = workflow_main.main_imported_case(str(source), _import_config(output))

    assert result["status"] == "success"
    assert calls == ["run"]
    assert len(result["attempts"]) == 1
    assert result["manifest"]["application"] == "icoFoam"
    assert (output / "original" / "system" / "controlDict").is_file()
    assert (output / "work" / "system" / "controlDict").is_file()


def test_imported_case_merges_into_the_common_local_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _write_case(tmp_path / "source")
    output = tmp_path / "output"
    common_runner_calls: list[str] = []
    original_runner = workflow_main.local_runner_node

    def controlled_success(*_args, **_kwargs):
        return []

    def common_runner(state):
        common_runner_calls.append(state["execution_policy"])
        return original_runner(state)

    monkeypatch.setattr(imported_case_node, "execute_imported_case", controlled_success)
    monkeypatch.setattr(workflow_main, "local_runner_node", common_runner)

    result = workflow_main.main_imported_case(str(source), _import_config(output))

    assert result["status"] == "success"
    assert common_runner_calls == ["controlled_import"]


def test_imported_case_can_explicitly_use_the_common_visualization_node(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _write_case(tmp_path / "source")
    output = tmp_path / "output"
    visualization_states: list[dict] = []

    def controlled_success(*_args, **_kwargs):
        return []

    def record_visualization(state):
        visualization_states.append(state)
        return {}

    monkeypatch.setattr(imported_case_node, "execute_imported_case", controlled_success)
    monkeypatch.setattr(workflow_main, "visualization_node", record_visualization)

    result = workflow_main.main_imported_case(
        str(source),
        _import_config(output),
        visualize=True,
    )

    assert result["status"] == "success"
    assert len(visualization_states) == 1
    assert visualization_states[0]["case_dir"] == str(output / "work")
    assert visualization_states[0]["requires_visualization"] is True


def test_imported_visualization_failure_is_not_reported_as_case_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A requested deterministic visualization is part of imported-case success."""
    source = _write_case(tmp_path / "source")
    output = tmp_path / "output"

    def controlled_success(*_args, **_kwargs):
        return []

    def failed_visualization(_state):
        return {
            "pyvista_visualization": {"success": False, "error": "Xvfb unavailable"},
            "plot_outputs": [],
            "plot_configs": [],
            "visualization_summary": {"pyvista_success": False},
            "case_import_status": "visualization_failed",
        }

    monkeypatch.setattr(imported_case_node, "execute_imported_case", controlled_success)
    monkeypatch.setattr(workflow_main, "visualization_node", failed_visualization)

    result = workflow_main.main_imported_case(
        str(source),
        _import_config(output),
        visualize=True,
    )

    assert result["status"] == "visualization_failed"


def test_imported_case_graph_retries_only_an_approved_non_numeric_repair(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = _write_case(tmp_path / "source", wrong_object_name=True)
    output = tmp_path / "output"
    calls = 0
    common_reviewer_calls: list[str] = []
    original_reviewer = workflow_main.reviewer_node

    def repaired_on_second_run(work_dir, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        control_dict = Path(work_dir) / "system" / "controlDict"
        if calls == 1:
            return [{"error_content": "FoamFile object mismatch"}]
        assert "object controlDict;" in control_dict.read_text(encoding="utf-8")
        return []

    def common_reviewer(state):
        common_reviewer_calls.append(state["repair_policy"])
        return original_reviewer(state)

    monkeypatch.setattr(imported_case_node, "execute_imported_case", repaired_on_second_run)
    monkeypatch.setattr(workflow_main, "reviewer_node", common_reviewer)

    result = workflow_main.main_imported_case(str(source), _import_config(output))

    assert result["status"] == "success"
    assert calls == 2
    assert common_reviewer_calls == ["numeric_invariant_only"]
    assert result["attempts"][0]["repairs"][0]["status"] == "applied"
    assert "object wrongName;" in (output / "original" / "system" / "controlDict").read_text(
        encoding="utf-8"
    )
    assert "object controlDict;" in (output / "work" / "system" / "controlDict").read_text(
        encoding="utf-8"
    )
