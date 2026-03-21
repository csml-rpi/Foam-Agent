"""Post-run interpreter: viz_creator plus VLM verdict."""

from __future__ import annotations

import json
from pathlib import Path


def interpreter_node(state):
    cfg = state["config"]
    if not getattr(cfg, "enable_post_run_interpreter", True):
        return {}

    case_dir = state.get("case_dir")
    print("============================== Interpreter ==============================")

    if not case_dir:
        return {
            "interpreter_revise_applied": None,
            "interpreter_report": {
                "rerun_required": False,
                "summary": "No case_dir; skipped interpreter.",
            },
        }

    case_path = Path(case_dir)
    if not case_path.is_dir():
        return {
            "interpreter_revise_applied": None,
            "interpreter_report": {
                "rerun_required": False,
                "summary": "case_dir not found",
            },
        }

    prompt_path = Path(__file__).resolve().parent.parent / "interpreter" / "prompts_results_interpreter.json"
    try:
        prompts = json.loads(prompt_path.read_text(encoding="utf-8"))["ResultsInterpreterAgent"]
    except Exception as e:
        return {
            "interpreter_revise_applied": None,
            "interpreter_report": {"rerun_required": False, "interpreter_error": str(e)},
        }

    try:
        from interpreter.interpreter_agent import ResultsInterpreterAgent
        from interpreter.llm_factory import create_interpreter_llm

        llm = create_interpreter_llm(cfg)
        agent = ResultsInterpreterAgent(llm, prompts)
        idea = {"description": state["user_requirement"]}
        spec = {
            "simulation_id": "foam_agent_case",
            "case_name": state.get("case_name") or "openfoam_case",
        }
        exp = {"output_dir": str(case_path.resolve()), "returncode": 0}
        report = agent.interpret(idea, spec, exp, verbose=True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        report = {
            "rerun_required": False,
            "simulation_success": False,
            "interpreter_error": str(e),
            "summary": "Interpreter failed; not requesting automatic rerun.",
        }

    try:
        artifact = case_path / "interpreter_report.json"
        artifact.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {artifact}", flush=True)
    except Exception as ex:
        print(f"Could not write interpreter_report.json: {ex}", flush=True)

    return {"interpreter_revise_applied": None, "interpreter_report": report}
