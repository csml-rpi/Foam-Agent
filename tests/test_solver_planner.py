"""Feature2 core regression using the existing real-workflow scenario harness."""

from pathlib import Path

import pytest

from test_meshing_subgraph import _exercise_meshing_scenario, _put, modules


@pytest.mark.parametrize("scenario", [
    "standard", "gmsh", "custom", "case_poly", "case_dict", "case_missing",
    "case_custom", "case_gmsh", "mesh_retry", "solver_retry", "solver_mesh_retry",
    "solver_mixed_retry", "target_mismatch",
])
def test_solver_planner_after_mesh_preserves_preparation_and_repairs(
    scenario, modules, monkeypatch, tmp_path,
):
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

    run = _exercise_meshing_scenario(scenario, m, monkeypatch, tmp_path)
    assert run.result["solver_planned"] is True
    assert run.visits.count("solver_planner") == len(planned) == 1
    imported = scenario.startswith("case_") or scenario.startswith("solver_")
    assert len(prepared) == len(materialized) == (0 if imported else 1)

    planning_prompts = [
        user for user, _, model in run.calls
        if model and model.__name__ in {"OpenFOAMPlanModel", "ExistingCasePlan"}
    ]
    if not imported:
        assert len(planning_prompts) == 1
    for prompt in planning_prompts:
        assert "<actual_mesh>" in prompt and "inlet" in prompt and "n_faces" in prompt

    if scenario in {"solver_mesh_retry", "solver_mixed_retry"}:
        assert run.visits.count("meshing_planner") == 2
        assert len(run.solver_runs) == 2
    if scenario == "solver_mixed_retry":
        assert run.result["rewrite_plan"]["target_files"] == [
            {"file": "0/U", "changes": "Apply the field correction"},
        ]
        assert (Path(run.result["case_dir"]) / "0/U").read_text() == "fixed physical field\n"
