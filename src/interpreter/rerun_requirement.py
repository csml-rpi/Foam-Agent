"""Revise Foam-Agent user requirement from interpreter feedback (cfd-scientist RerunAnalysisAgent logic, no HypothesisAgent)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from interpreter.text_utils import strip_json_fences


def _extract_feedback(interp: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key in (
        "rerun_reason",
        "reasons",
        "issues",
        "summary",
        "requirement_update",
        "requirement_updates",
        "recommended_requirement_update",
        "suggested_requirement_update",
        "next_requirement",
    ):
        val = interp.get(key)
        if isinstance(val, str) and val.strip():
            lines.append(val.strip())
        elif isinstance(val, list):
            lines.extend(str(x).strip() for x in val if str(x).strip())
        elif isinstance(val, dict) and key == "issues":
            xs = val.get("issues")
            if isinstance(xs, list):
                lines.extend(str(x).strip() for x in xs if str(x).strip())
    return [x for x in lines if x]


def _strip_visualization_mentions(req: str) -> str:
    banned = (
        "visualize", "visualization", "plot ", " contour", "streamline", "png", "paraview",
        "pyvista", "figure", "screenshot", "render",
    )
    sentences = [s.strip() for s in req.replace("\n", " ").split(".")]
    kept = []
    for s in sentences:
        low = s.lower()
        if any(tok in low for tok in banned):
            continue
        if s:
            kept.append(s)
    out = ". ".join(kept).strip()
    return out if out.strip() else req


def _repair_requirement(llm: Any, req: str, issues: List[str], guidance: List[str]) -> str:
    system = (
        "You repair CFD prompts for Foam-Agent. Output exactly one corrected plain-English requirement paragraph. "
        "Include solver, domain/mesh, boundary conditions, and time controls. "
        "Do NOT add visualization or plotting instructions."
    )
    user = (
        "Requirement:\n{req}\n\nIssues:\n{issues}\n\nGuidance:\n{guidance}\n\nReturn only the corrected paragraph."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    chain = prompt | llm
    return chain.invoke(
        {
            "req": req,
            "issues": "\n".join(f"- {i}" for i in issues),
            "guidance": "\n".join(f"- {g}" for g in guidance),
        }
    ).content.strip()


def _validate_requirement(llm: Any, req: str) -> Dict[str, Any]:
    system = (
        "You are a strict CFD QA checker for Foam-Agent prompts. "
        "Decide if the requirement is executable (solver, BCs, time controls, mesh hints). "
        "No visualization instructions allowed. "
        "Return ONLY JSON: {\"valid\": true/false, \"issues\": [\"...\"], \"repair_guidance\": [\"...\"]}"
    )
    user = "Requirement:\n{req}\n\nReturn JSON only."
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    chain = prompt | llm
    raw = chain.invoke({"req": req}).content
    try:
        parsed = json.loads(strip_json_fences(raw))
        if not isinstance(parsed, dict):
            raise ValueError("not dict")
        parsed.setdefault("valid", False)
        parsed.setdefault("issues", [])
        parsed.setdefault("repair_guidance", [])
        return parsed
    except Exception:
        return {
            "valid": False,
            "issues": ["Validator returned non-JSON."],
            "repair_guidance": ["Return one coherent Foam-Agent requirement without viz instructions."],
            "raw": raw,
        }


def revise_requirement_after_interpreter(
    llm: Any,
    current_requirement: str,
    interpreter_report: Dict[str, Any],
) -> Dict[str, Any]:
    feedback = _extract_feedback(interpreter_report or {})
    if not feedback:
        verdict = _validate_requirement(llm, current_requirement)
        return {
            "requirement": current_requirement,
            "valid": bool(verdict.get("valid", False)),
            "feedback": [],
            "validator": verdict,
        }

    guidance = [
        "Update the requirement to address interpreter-detected physics or setup issues.",
        "Keep the requirement executable by Foam-Agent (solver, time controls, BCs, mesh).",
        "Do not include visualization instructions.",
    ]
    revised = _repair_requirement(
        llm,
        current_requirement,
        [f"Interpreter feedback: {x}" for x in feedback],
        guidance,
    )
    revised = _strip_visualization_mentions(revised)

    verdict = _validate_requirement(llm, revised)
    if verdict.get("valid", False):
        return {"requirement": revised, "valid": True, "feedback": feedback, "validator": verdict}

    repaired = _repair_requirement(
        llm,
        revised,
        verdict.get("issues", []),
        verdict.get("repair_guidance", []),
    )
    repaired = _strip_visualization_mentions(repaired)
    v2 = _validate_requirement(llm, repaired)
    return {
        "requirement": repaired,
        "valid": bool(v2.get("valid", False)),
        "feedback": feedback,
        "validator": v2,
    }
