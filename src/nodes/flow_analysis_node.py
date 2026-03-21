"""Flow-field narrative after interpreter approves: analysis_viz/ plus VLM text."""

from __future__ import annotations

import json
from pathlib import Path


def flow_analysis_node(state):
    cfg = state["config"]
    if not getattr(cfg, "enable_flow_field_analysis", True):
        return {}

    case_dir = state.get("case_dir")
    print("============================== Flow field analysis ==============================")

    if not case_dir:
        return {
            "flow_analysis_text": "",
            "flow_analysis_bundle": {"skipped": True, "reason": "no case_dir"},
        }

    case_path = Path(case_dir)
    if not case_path.is_dir():
        return {
            "flow_analysis_text": "",
            "flow_analysis_bundle": {"skipped": True, "reason": "case_dir missing"},
        }

    try:
        from interpreter.flow_analysis_agent import FlowFieldAnalysisAgent
        from interpreter.llm_factory import create_interpreter_llm

        llm = create_interpreter_llm(cfg)
        agent = FlowFieldAnalysisAgent(llm)
        out = agent.run_for_case(
            foam_output_dir=case_path,
            user_requirement=state.get("user_requirement") or "",
            case_name=state.get("case_name") or "openfoam_case",
            topic=state.get("case_name"),
            verbose=True,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"flow_analysis_text": "", "flow_analysis_bundle": {"ok": False, "error": str(e)}}

    text = out.get("analysis_text") or ""
    bundle = {
        "ok": True,
        "viz_spec": out.get("viz_spec", ""),
        "visualization": out.get("visualization", {}),
        "image_paths": out.get("image_paths", []),
    }

    try:
        spec = bundle.get("viz_spec") or ""
        rp = case_path / "flow_analysis_report.md"
        md = "# Flow field analysis\n\n" + text + "\n\n## Visualization specification\n\n" + spec + "\n"
        rp.write_text(md, encoding="utf-8")
        print("Wrote", rp, flush=True)
        jp = case_path / "flow_analysis_bundle.json"
        jp.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    except Exception as ex:
        print("Could not write flow analysis files:", ex, flush=True)

    return {"flow_analysis_text": text, "flow_analysis_bundle": bundle}
