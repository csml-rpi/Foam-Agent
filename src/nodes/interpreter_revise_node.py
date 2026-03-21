"""Revise user requirement from interpreter feedback and re-enter planner (max N times)."""

from __future__ import annotations


def interpreter_revise_node(state):
    cfg = state["config"]
    interp = state.get("interpreter_report") or {}

    print("============================== Interpreter requirement revision ==============================")

    try:
        from interpreter.llm_factory import create_interpreter_llm
        from interpreter.rerun_requirement import revise_requirement_after_interpreter

        llm = create_interpreter_llm(cfg)
        rev = revise_requirement_after_interpreter(llm, state["user_requirement"], interp)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "interpreter_revise_applied": False,
            "interpreter_report": {
                **interp,
                "rerun_required": False,
                "rerun_reason": f"Revision LLM failed: {e}",
            },
        }

    next_req = str(rev.get("requirement", "") or "").strip()
    if not rev.get("valid", False) or not next_req:
        return {
            "interpreter_revise_applied": False,
            "interpreter_report": {
                **interp,
                "rerun_required": False,
                "rerun_reason": "Requirement revision invalid or empty; stopping interpreter rerun loop.",
            },
        }

    updates = list(state.get("interpreter_requirement_updates") or [])
    updates.append(
        {
            "attempt": int(state.get("interpreter_rerun_count") or 0),
            "feedback": rev.get("feedback", []),
            "valid": bool(rev.get("valid", False)),
            "requirement": next_req,
        }
    )

    return {
        "interpreter_revise_applied": True,
        "user_requirement": next_req,
        "interpreter_rerun_count": int(state.get("interpreter_rerun_count") or 0) + 1,
        "loop_count": 0,
        "error_logs": [],
        "interpreter_requirement_updates": updates,
        "flow_analysis_text": None,
        "flow_analysis_bundle": None,
    }
