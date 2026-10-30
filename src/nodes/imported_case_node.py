"""Existing-case materialisation adapter for the shared agent graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from logger import setup_logging
from openfoam_target import ESI_V2006, FOUNDATION_V10
from services.case_import import CaseImportError, build_case_context, import_case
from utils import load_agent_resources, load_case_files


def case_import_node(state: dict[str, Any]) -> dict[str, Any]:
    """Copy and inspect a case, then hand it to the normal planner."""
    print("<case_import>")
    config = state["config"]
    try:
        manifest = import_case(
            state["case_import_path"],
            config.case_dir,
            case_subdir=state.get("case_import_subdir"),
            overwrite=config.overwrite_case_dir,
            openfoam_target=getattr(config, "openfoam_target", ""),
        )
    except (CaseImportError, OSError) as exc:
        print(f"<case_import_error>{exc}</case_import_error>")
        print("</case_import>")
        return {
            "workflow_status": "failed",
            "error_logs": [{"file": "case_import", "error_content": str(exc)}],
            "termination_reason": "case_import_failed",
        }

    if not getattr(config, "openfoam_target", ""):
        if manifest.platform == ESI_V2006:
            config.openfoam_target = ESI_V2006
        elif manifest.platform == FOUNDATION_V10:
            config.openfoam_target = FOUNDATION_V10

    output_root = Path(manifest.output_root)
    work = output_root / "work"
    report = output_root / "report"
    setup_logging(str(report / "logs"))
    context = build_case_context(work, manifest=manifest)

    llm_service = None
    case_stats = None
    if manifest.platform in {FOUNDATION_V10, ESI_V2006}:
        try:
            llm_service, case_stats = load_agent_resources(config)
        except Exception as exc:
            print(f"<case_import_resource_error>{exc}</case_import_resource_error>")
            print("</case_import>")
            return {
                "case_import_manifest": manifest,
                "case_import_report_dir": str(report),
                "case_dir": str(work),
                "case_context": context,
                "workflow_status": "failed",
                "error_logs": [{"file": "agent_resources", "error_content": str(exc)}],
                "termination_reason": "agent_resources_unavailable",
            }

    update = {
        "case_origin": "imported",
        "case_import_manifest": manifest,
        "case_import_report_dir": str(report),
        "case_dir": str(work),
        "case_name": Path(manifest.case_root).name if manifest.case_root != "." else work.name,
        "case_solver": manifest.application,
        "case_context": context,
        "llm_service": llm_service,
        "case_stats": case_stats,
        "workflow_status": "planning",
        "error_logs": [],
        "termination_reason": None,
    }
    update.update(load_case_files({**state, **update}))
    print(f"<platform>{manifest.platform}</platform>")
    print(f"<application>{manifest.application}</application>")
    print("</case_import>")
    return update
