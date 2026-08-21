# visualization_node.py
"""Thin LangGraph adapter for visualization."""

from services.visualization import visualize_case

def visualization_node(state):
    """Delegate visualization work and merge its result into graph state."""
    print("<visualization>")
    result = visualize_case(
        state.get("case_dir"),
        state.get("user_requirement", ""),
        llm_service=state.get("llm_service"),
        allow_llm_fallback=state.get("execution_policy") != "controlled_import",
    )
    if not result["pyvista_visualization"]["success"]:
        print(f"<visualization_error>{result['pyvista_visualization']['error']}</visualization_error>")
    print("</visualization>")
    if (
        state.get("execution_policy") == "controlled_import"
        and not result["pyvista_visualization"]["success"]
    ):
        # Imported cases promise a deterministic, non-LLM visualization path.
        # Treat an unavailable renderer as a failed requested operation instead
        # of reporting the simulation as fully successful.
        return {
            **state,
            **result,
            "case_import_status": "visualization_failed",
            "termination_reason": "imported_case_visualization_failed",
        }
    return {**state, **result}
