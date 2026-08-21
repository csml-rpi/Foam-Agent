"""Regression tests for workflow failure-state persistence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nodes import planner_node as planner_module  # noqa: E402
from nodes import reviewer_node as reviewer_module  # noqa: E402
from router_func import (  # noqa: E402
    route_after_case_import,
    route_after_meshing,
    route_after_reviewer,
    route_after_runner,
)
from services.output_safety import validate_output_path  # noqa: E402
from main import workflow_exit_code  # noqa: E402


def test_reviewer_persists_max_loop_termination_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        reviewer_module,
        "review_error_logs",
        lambda **_kwargs: ("diagnosis", ["attempt"]),
    )
    monkeypatch.setattr(
        reviewer_module,
        "generate_rewrite_plan",
        lambda **_kwargs: {"target_files": []},
    )
    monkeypatch.setattr(reviewer_module, "log_review", lambda *_args: None)

    class Config:
        max_loop = 1

    result = reviewer_module.reviewer_node(
        {
            "config": Config(),
            "error_logs": [{"file": "Allrun", "error_content": "failed"}],
            "loop_count": 0,
            "history_text": [],
            "tutorial_reference": "",
            "foamfiles": None,
            "user_requirement": "Run a valid case.",
            "similar_case_advice": None,
        }
    )

    assert result["loop_count"] == 1
    assert result["termination_reason"] == "max_review_loop_reached"


def test_nonempty_explicit_output_is_rejected_before_planning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    (output_dir / "prior-result").write_text("keep", encoding="utf-8")

    def planning_must_not_run(**_kwargs):
        raise AssertionError("protected output should fail before planning")

    monkeypatch.setattr(
        planner_module,
        "generate_simulation_plan",
        planning_must_not_run,
    )

    class Config:
        case_dir = str(output_dir)
        overwrite_case_dir = False

    with pytest.raises(FileExistsError, match="not empty"):
        planner_module.planner_node(
            {
                "config": Config(),
                "user_requirement": "Run a case.",
                "case_stats": {},
            }
        )

    assert (output_dir / "prior-result").read_text(encoding="utf-8") == "keep"


def test_overwrite_rejects_repository_root() -> None:
    with pytest.raises(ValueError, match="protected broad path"):
        validate_output_path(ROOT)


def test_runner_route_ends_without_visualization() -> None:
    assert route_after_runner(
        {"error_logs": [], "requires_visualization": False}
    ) == "__end__"


def test_meshing_route_ends_after_a_mesh_failure() -> None:
    assert route_after_meshing(
        {"error_logs": [{"file": "mesh", "error_content": "gmshToFoam failed"}]}
    ) == "__end__"
    assert route_after_meshing({"error_logs": []}) == "input_writer"


def test_mesh_generation_failure_returns_a_nonzero_cli_exit_code() -> None:
    assert workflow_exit_code({"termination_reason": "mesh_generation_failed"}) == 2
    assert workflow_exit_code({"termination_reason": None}) == 0


def test_reviewer_route_ends_at_the_retry_limit_without_visualization() -> None:
    class Config:
        max_loop = 1

    assert route_after_reviewer(
        {
            "config": Config(),
            "loop_count": 1,
            "requires_visualization": False,
        }
    ) == "__end__"


def test_imported_case_uses_common_runner_and_reviewer_routes() -> None:
    assert route_after_case_import({"case_import_status": "ready"}) == "local_runner"
    assert route_after_reviewer(
        {
            "repair_policy": "numeric_invariant_only",
            "case_import_status": "ready",
        }
    ) == "local_runner"
    assert route_after_reviewer(
        {
            "repair_policy": "numeric_invariant_only",
            "case_import_status": "blocked",
        }
    ) == "__end__"
    assert route_after_reviewer({"repair_policy": "unsupported"}) == "__end__"
