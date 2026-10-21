"""Materialise non-CAD inputs without deciding mesh or solver policy."""

import shutil
from pathlib import Path
from typing import Any

from logger import setup_logging
from openfoam_target import ESI_V2006, FOUNDATION_V10, configured_openfoam_target
from utils import load_agent_resources, load_case_files
from .case_import import CaseImportError, import_case, build_case_context
from .output_safety import prepare_output_directory
from .plan import resolve_case_dir


def prepare_inputs(state):
    config = state["config"]
    try:
        if state.get("workflow_mode") == "imported_case":
            # Keep the existing adapter's target detection and failure semantics.
            update = prepare_imported_case(state)
            if update.get("workflow_status") == "failed":
                return update
        else:
            requested = config.case_dir or resolve_case_dir(
                "case", run_times=config.run_times, run_directory=config.run_directory,
            )
            case = prepare_output_directory(
                requested, overwrite=config.overwrite_case_dir,
                source_path=state.get("custom_mesh_path"),
            )
            setup_logging(str(case))
            try:
                llm, stats = load_agent_resources(config)
            except Exception as exc:
                return {
                    "case_dir": str(case), "workflow_status": "failed",
                    "termination_reason": "agent_resources_unavailable",
                    "error_logs": [{"file": "agent_resources", "error_content": str(exc)}],
                }
            update = {
                "case_dir": str(case), "case_name": Path(case).name,
                "llm_service": llm, "case_stats": stats,
                "workflow_status": "planning", "error_logs": [],
            }

        current = {**state, **update}
        case = Path(current["case_dir"])
        if state.get("custom_mesh_path"):
            source = Path(state["custom_mesh_path"]).expanduser().resolve()
            destination = case / "input" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination.resolve():
                shutil.copy2(source, destination)
            update["custom_mesh_path"] = str(destination)
        update["case_context"] = current.get("case_context") or build_case_context(case)
        update.update(load_case_files(update))
        return update
    except (OSError, ValueError) as exc:
        return {
            "workflow_status": "failed", "termination_reason": "preparation_failed",
            "error_logs": [{"file": "preparation", "error_content": str(exc)}],
        }


def prepare_imported_case(state: dict[str, Any]) -> dict[str, Any]:
    """Copy and inspect a case, then hand its context to Meshing."""
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

    failure = None
    target = configured_openfoam_target(config) or context.get("platform")
    if target not in {FOUNDATION_V10, ESI_V2006}:
        failure = {
            "workflow_status": "failed",
            "requires_meshing": False,
            "requires_input_writer": False,
            "termination_reason": "Specify --openfoam_target foundation-v10 or esi-v2006.",
        }
    elif manifest.target_mismatch:
        mismatch_errors = [
            issue for issue in context.get("issues", [])
            if issue.startswith("Configured target ") and "differs from detected" in issue
        ] or ["Configured OpenFOAM target does not match the detected case platform."]
        failure = {
            "workflow_status": "failed",
            "target_mismatch": True,
            "requires_meshing": False,
            "requires_input_writer": False,
            "error_logs": [
                {"file": "case_import", "error_content": issue}
                for issue in mismatch_errors
            ],
            "termination_reason": "case_target_mismatch",
        }
    if failure:
        print("</case_import>")
        return {
            "case_import_manifest": manifest, "case_import_report_dir": str(report),
            "case_dir": str(work), "case_context": context, **failure,
        }

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
    update.update(load_case_files(update))
    print(f"<platform>{manifest.platform}</platform>")
    print(f"<application>{manifest.application}</application>")
    print("</case_import>")
    return update
