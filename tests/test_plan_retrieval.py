"""Focused tests for metadata-safe tutorial retrieval."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from services import plan  # noqa: E402


def _candidate(
    *,
    name: str,
    domain: str,
    solver: str,
    score: float,
    structure: str,
    full_content: str,
    category: str = "None",
) -> dict[str, Any]:
    return {
        "case_name": name,
        "case_domain": domain,
        "case_category": category,
        "case_solver": solver,
        "score": score,
        "dir_structure": structure,
        "full_content": full_content,
    }


def _advice() -> plan.SimilarCaseAdviceModel:
    return plan.SimilarCaseAdviceModel(
        match_level="high",
        use_scope="compatible files",
        advice="Use the compatible reference.",
    )


def test_parse_requirement_rejects_path_traversal_case_name(monkeypatch) -> None:
    class MaliciousPlannerLLM:
        def invoke(self, *_args, **_kwargs):
            return plan.CaseSummaryModel(
                case_name="../escape",
                case_domain="incompressible",
                case_category="tutorial",
                case_solver="icoFoam",
            )

    monkeypatch.setattr(plan, "global_llm_service", MaliciousPlannerLLM())

    with pytest.raises(ValueError, match="Unsafe generated case name"):
        plan.parse_requirement_to_case_info(
            "Generate a cavity case.",
            {
                "case_domain": ["incompressible"],
                "case_category": ["tutorial"],
                "case_solver": ["icoFoam"],
            },
        )


def test_truncation_preserves_tree_file_names_and_small_files() -> None:
    reference = """<index>case name: genericCase</index>
<directory_structure>
<dir>directory name: constant. File names in this directory: [largeData, settings]</dir>
</directory_structure>
<tutorials>
<directory_begin>directory name: constant
<file_begin>file name: largeData
<file_content>HEADER
{large_content}
TAIL</file_content>
</file_end>
<file_begin>file name: settings
<file_content>importantSetting  true;</file_content>
</file_end>
</directory_end>
</tutorials>""".format(large_content="x" * 2_000)

    cropped = plan._truncate_large_reference_files(
        reference,
        per_file_limit=160,
        total_content_limit=1_000,
    )

    assert "<directory_structure>" in cropped
    assert "file name: largeData" in cropped
    assert "file name: settings" in cropped
    assert "importantSetting  true;" in cropped
    assert "reference content omitted" in cropped
    assert "HEADER" not in cropped
    assert "TAIL" not in cropped
    assert "x" * 500 not in cropped


def test_rerank_uses_generic_case_name_terms_before_vector_score() -> None:
    candidates = [
        {
            "case_name": "unrelatedExample",
            "case_solver": "sharedSolver",
            "score": 0.01,
        },
        {
            "case_name": "thermalMixing",
            "case_solver": "sharedSolver",
            "score": 0.5,
        },
    ]

    ranked = plan._rerank_candidates(
        candidates,
        "sharedSolver",
        "Model transient thermal mixing in a vessel.",
    )

    assert ranked[0]["case_name"] == "thermalMixing"


def test_retrieve_references_expands_recall_and_loads_exact_details(monkeypatch) -> None:
    target_structure = (
        "<dir>directory name: 0. File names in this directory: [U]</dir>\n"
        "<dir>directory name: constant. File names in this directory: "
        "[largeData, solverProperties]</dir>"
    )
    other_structure = (
        "<dir>directory name: 0. File names in this directory: [wrongField]</dir>"
    )
    target_details = f"""<index>
case name: compatibleCase
case domain: particles
case category: None
case solver: particleSolver
</index>
<directory_structure>{target_structure}</directory_structure>
<tutorials>
<file_begin>file name: largeData
<file_content>{'p' * 30_000}</file_content>
</file_end>
<file_begin>file name: solverProperties
<file_content>correctSetting true;</file_content>
</file_end>
</tutorials>"""

    structure_candidates = [
        _candidate(
            name="wrongDomain",
            domain="compressible",
            solver="particleSolver",
            score=0.01,
            structure=other_structure,
            full_content="structure-only-wrong-domain",
        ),
        _candidate(
            name="wrongSolver",
            domain="particles",
            solver="otherSolver",
            score=0.02,
            structure=other_structure,
            full_content="structure-only-wrong-solver",
        ),
        _candidate(
            name="compatibleCase",
            domain="particles",
            solver="particleSolver",
            score=5.0,
            structure=target_structure,
            full_content="structure-only-target",
        ),
    ]
    detail_candidates = [
        # Same metadata but a different tree: exact tree must win over score.
        _candidate(
            name="compatibleCase",
            domain="particles",
            solver="particleSolver",
            score=0.01,
            structure=other_structure,
            full_content="wrong duplicate details",
        ),
        _candidate(
            name="compatibleCase",
            domain="particles",
            solver="particleSolver",
            score=10.0,
            structure=target_structure,
            full_content=target_details,
        ),
        _candidate(
            name="anotherCase",
            domain="particles",
            solver="particleSolver",
            score=0.001,
            structure=target_structure,
            full_content="wrong case details",
        ),
    ]
    calls: list[tuple[str, str, int]] = []
    retrieval_config = object()
    received_configs: list[object | None] = []

    def fake_retrieve(database_name: str, query: str, topk: int = 1, **kwargs):
        calls.append((database_name, query, topk))
        received_configs.append(kwargs.get("config"))
        if database_name == "openfoam_tutorials_structure":
            return structure_candidates
        if database_name == "openfoam_tutorials_details":
            return detail_candidates
        if database_name == "openfoam_allrun_scripts":
            return [{"full_content": "runApplication particleSolver"}]
        raise AssertionError(f"Unexpected database: {database_name}")

    monkeypatch.setattr(plan, "retrieve_faiss", fake_retrieve)
    monkeypatch.setattr(plan, "_build_advice", lambda *args, **kwargs: _advice())

    details, structure, counts, allrun, advice = plan.retrieve_references(
        case_name="newCase",
        case_solver="particleSolver",
        case_domain="particles",
        case_category="None",
        searchdocs=2,
        user_requirement="Simulate a generic particle flow.",
        config=retrieval_config,
    )

    assert [call[0] for call in calls] == [
        "openfoam_tutorials_structure",
        "openfoam_tutorials_details",
        "openfoam_allrun_scripts",
    ]
    assert received_configs == [retrieval_config, retrieval_config, retrieval_config]
    assert calls[0][2] >= 200
    assert calls[1][2] >= 50
    assert "generic particle flow" in calls[0][1]
    assert structure == target_structure
    assert "There are 1 files in Directory: 0" in counts
    assert "correctSetting true;" in details
    assert "wrong duplicate details" not in details
    assert "structure-only-target" not in details
    assert "reference content omitted" in details
    assert "p" * 100 not in details
    assert "runApplication particleSolver" in allrun
    assert advice.match_level == "high"


def test_retrieve_references_does_not_fall_back_to_wrong_solver(monkeypatch) -> None:
    structure = "<dir>directory name: 0. File names in this directory: [U]</dir>"
    calls: list[str] = []

    def fake_retrieve(database_name: str, query: str, topk: int = 1):
        calls.append(database_name)
        return [
            _candidate(
                name="wrongSolverCase",
                domain="fluid",
                solver="otherSolver",
                score=0.0,
                structure=structure,
                full_content="wrong solver contents",
            )
        ]

    monkeypatch.setattr(plan, "retrieve_faiss", fake_retrieve)
    monkeypatch.setattr(plan, "_build_advice", lambda *args, **kwargs: _advice())

    result = plan.retrieve_references(
        case_name="newCase",
        case_solver="requiredSolver",
        case_domain="fluid",
        case_category="None",
        user_requirement="Run a fluid case.",
    )

    assert result[:4] == ("", "", "", "")
    assert calls == ["openfoam_tutorials_structure"]


def test_details_require_the_same_directory_structure(monkeypatch) -> None:
    selected_structure = (
        "<dir>directory name: 0. File names in this directory: [U]</dir>"
    )
    different_structure = (
        "<dir>directory name: 0. File names in this directory: [T]</dir>"
    )
    selected = _candidate(
        name="sameMetadata",
        domain="fluid",
        solver="flowSolver",
        score=1.0,
        structure=selected_structure,
        full_content="structure entry",
    )
    wrong_tree_detail = _candidate(
        name="sameMetadata",
        domain="fluid",
        solver="flowSolver",
        score=0.0,
        structure=different_structure,
        full_content="details for another tree",
    )

    monkeypatch.setattr(
        plan,
        "retrieve_faiss",
        lambda *_args, **_kwargs: [wrong_tree_detail],
    )

    assert (
        plan._retrieve_matching_details(selected, selected_structure, recall_k=50)
        is None
    )
