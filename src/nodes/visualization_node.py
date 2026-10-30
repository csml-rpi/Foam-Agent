# visualization_node.py
"""Thin LangGraph adapter for visualization."""

from services.visualization import visualize_case

def visualization_node(state):
    """Delegate visualization work and merge its result into graph state."""
    print("<visualization>")
    try:
        result = visualize_case(
            state.get("case_dir"),
            state.get("user_requirement", ""),
            llm_service=state.get("llm_service"),
            max_loop=getattr(state.get("config"), "max_loop", 2),
        )
    except Exception as exc:
        result = {"pyvista_visualization": {"success": False, "error": str(exc)}}
    if not result["pyvista_visualization"]["success"]:
        error = result["pyvista_visualization"]["error"]
        print(f"<visualization_error>{error}</visualization_error>")
        result["visualization_error"] = error
        if state.get("workflow_status") == "success" and not state.get("termination_reason"):
            result["workflow_status"] = "partial_success"
            result["termination_reason"] = "visualization_failed"
            result["workflow_message"] = "Simulation completed successfully, but visualization failed."
    print("</visualization>")
    return {**state, **result}
