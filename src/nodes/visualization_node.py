# visualization_node.py
"""Thin LangGraph adapter for visualization."""
from logger import print

from services.visualization import visualize_case
from services.execution_logs import declared_startup_inputs

def visualization_node(state):
    """Delegate visualization work and merge its result into graph state."""
    print("<visualization>")
    try:
        result = visualize_case(
            state.get("case_dir"),
            state.get("user_requirement", ""),
            llm_service=state.get("llm_service"),
            config=state.get('config'),
            max_loop=getattr(state.get("config"), "max_loop", 2),
            protected_inputs=declared_startup_inputs([
                *state.get('mesh_operations', []), *state.get('solver_operations', []),
                *state.get('pending_solver_operations', []),
            ]),
            source_input_dir=state.get('case_input_source_dir'),
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
