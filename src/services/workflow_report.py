"""Offline execution reports built from workflow outputs and existing evidence."""

from datetime import datetime, timezone
from functools import wraps
from html import escape
import json
import os
from pathlib import Path
import sys
from time import monotonic
from urllib.parse import quote

from logger import active_log_paths
from .case_paths import safe_case_path
from .output_safety import output_is_owned


_TITLES = {
    "preparation": "Prepare case", "meshing_planner": "Plan mesh",
    "standard_meshing": "Generate and check mesh", "gmsh_meshing": "Generate Gmsh mesh",
    "custom_meshing": "Prepare supplied mesh", "solver_planner": "Plan physics and solver",
    "input_writer": "Write and repair inputs", "static_review": "Review solver inputs",
    "local_runner": "Run solver locally", "hpc_runner": "Run solver on HPC",
    "mesh_reviewer": "Diagnose mesh failure", "solver_reviewer": "Diagnose solver failure",
    "visualization": "Visualize results",
}
_FIELDS = {
    "preparation": ("case_origin", "case_context"),
    "meshing_planner": ("mesh_type", "mesh_source", "mesh_boundary_plan", "mesh_operations", "mesh_subtasks"),
    "solver_planner": ("case_solver", "case_info", "similar_case_advice", "solver_operations", "subtasks"),
    "input_writer": ("written_files", "updated_files", "file_diffs"),
    "static_review": ("static_review_report",),
    "visualization": ("pyvista_visualization", "visualization_error"),
}
_LABELS = {"user": "User requirement", "input": "Supplied input",
           "reference": "Tutorial reference", "inferred": "Model assumption"}
_HISTORY_FIELDS = {
    "timestamp", "event", "stage", "source", "iteration", "runtime_loop", "runner",
    "report", "verification", "reason", "review_analysis", "applied_changes", "static_iteration",
    "updated_files", "file_diffs", "error_logs", "outcome", "job_id",
    "error", "errors", "response", "validation_errors", "schema", "attempt",
}
_STYLE = """
:root { color-scheme: light dark; --bg:#f5f7fa; --card:#fff; --ink:#17263b;
  --muted:#576579; --line:#d8e0ea; --accent:#1d6476; --bad:#a53434; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.6 system-ui,sans-serif; } main { max-width:1040px; margin:auto; padding:36px 22px; }
h1 { margin:4px 0 12px; font-size:2rem; } h2 { margin:32px 0 12px; font-size:1.2rem; }
p { margin:8px 0; } .eyebrow, .meta { color:var(--muted); font-size:.85rem; overflow-wrap:anywhere; }
.eyebrow { text-transform:uppercase; letter-spacing:.12em; } a { color:var(--accent); }
.panel, .step { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 20px; margin:12px 0; } summary { cursor:pointer; overflow-wrap:anywhere; }
.step summary { font-weight:600; } .badge { display:inline-block; border-radius:4px;
  padding:2px 8px; background:var(--bg); font-size:.8rem; margin-left:8px; color:var(--muted); }
.attention, .stopped, .failed { color:var(--bad); } .completed { color:var(--accent); }
.path { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0; font-size:.8rem; }
.path a { border:1px solid var(--line); border-radius:4px; padding:4px 8px; text-decoration:none; }
.path a:not(:last-child)::after { content:" →"; }
.overview > dl { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
.overview pre { font:inherit; } .overview dd { margin-bottom:0; }
ol.timeline { list-style:none; padding:0 0 0 18px; border-left:2px solid var(--line); }
dl { margin:12px 0; } dt { font-size:.85rem; font-weight:600; color:var(--muted); }
dd { margin:2px 0 12px; } pre { font: .85rem/1.6 ui-monospace,monospace; white-space:pre-wrap;
  overflow-wrap:anywhere; margin:6px 0; max-height:32rem; overflow:auto; }
ul { padding-left:22px; } li { overflow-wrap:anywhere; } .artifacts { display:flex; gap:16px; flex-wrap:wrap; }
@media(prefers-color-scheme:dark) { :root { --bg:#141b25; --card:#1d2735; --ink:#e4eaf2;
  --muted:#afbdce; --line:#354459; --accent:#8bd2dc; --bad:#ffabab; } }
@media(max-width:600px) { main { padding:20px 12px; } .panel,.step { padding:12px; }
  .overview > dl { grid-template-columns:1fr; } }
"""


def _diagnostic(message):
    """Keep report diagnostics independent of case logs and best-effort only."""
    try:
        text = message + "\n"
        stream = sys.stderr
        try:
            descriptor = stream.fileno()
        except (AttributeError, OSError, ValueError):
            stream.write(text)
            stream.flush()
        else:
            # A failed buffered write could otherwise be retried at process exit.
            os.write(descriptor, text.encode(stream.encoding or "utf-8", errors="backslashreplace"))
    except Exception:
        pass


def _serializable(value):
    return value.model_dump() if hasattr(value, "model_dump") else str(value)


def _value(value):
    """Render recorded data as text, never interpreting model output as markup."""
    if isinstance(value, dict):
        return "<dl>" + "".join(
            f"<div><dt>{escape(str(key).replace('_', ' ').capitalize())}</dt>"
            f"<dd>{_value(_LABELS.get(item, item) if key == 'basis' and isinstance(item, str) else item)}</dd></div>"
            for key, item in value.items()) + "</dl>"
    if isinstance(value, list):
        return "<ul>" + "".join(f"<li>{_value(item)}</li>" for item in value) + "</ul>"
    return "<pre>" + escape("Not recorded" if value is None else str(value)) + "</pre>"


def _skipped_run(name, update):
    # Runners retain the preceding native outcome when rejecting repeated failed inputs.
    return name.endswith("_runner") and update.get("termination_reason") == "repair_made_no_progress"


def _details(name, state, update):
    fields = _FIELDS.get(name, ())
    if name.endswith("_meshing"):
        fields = ("mesh_info", "mesh_execution", "static_review_report", "written_files", "updated_files", "file_diffs")
    elif name.endswith("_runner"):
        fields = ("solver_execution", "job_id")
    source = update if (name in {"input_writer", "static_review", "visualization"}
                        or name.endswith(("_meshing", "_runner"))) else state
    details = {key: source[key] for key in fields if source.get(key) is not None}
    if name == "meshing_planner":
        reference = state.get("mesh_reference") or {}
        details["tutorial_reference"] = {k: reference.get(k) for k in ("case", "reason")}
    if name == "solver_planner":
        details["reference_evidence"] = {
            "tutorial": update.get("tutorial_reference"),
            "execution_script": update.get("allrun_reference"),
        }
    if name.endswith("_reviewer"):
        history = state.get(name.split("_")[0] + "_review_history") or []
        details["review_decision"] = history[-1] if history else state.get("review_analysis")
    for key in ("rewrite_plan", "cross_stage_request", "termination_reason", "workflow_message", "error_logs"):
        if update.get(key):
            details[key] = update[key]
    if _skipped_run(name, update):
        details["solver_execution"] = {"executed": False}
        details.pop("job_id", None)
        if "error_logs" in details:
            details["previous_execution_errors"] = details.pop("error_logs")
    # Detach snapshots from mutable graph state; never serialize the LLM client or Config.
    return json.loads(json.dumps(details, default=_serializable, ensure_ascii=False))


class WorkflowReport:
    """One observer per invocation, kept outside graph state and model prompts."""

    def __init__(self, requirement):
        self.requirement = requirement
        self.started = datetime.now(timezone.utc).isoformat()
        self.clock = monotonic()
        self.steps = []
        self.state = {}

    def observe(self, name, node):
        @wraps(node)
        def observed(state):
            self.state = state
            started = monotonic()
            try:
                update = node(state)
            except BaseException as exc:
                self._record(name, state, {}, started, exc)
                raise
            self._record(name, state, update, started)
            return update
        return observed

    def _record(self, name, state, update, started, error=None):
        try:
            self.state = {**state, **update}
            status = "completed"
            if _skipped_run(name, update):
                status = "skipped"
            elif error or update.get("termination_reason") or update.get("workflow_status") == "failed":
                status = "stopped"
            elif name.endswith("_reviewer"):
                status = "repair planned" if self.state.get("rewrite_plan") or self.state.get("cross_stage_request") else "completed"
            elif update.get("error_logs"):
                status = "attention"
            elif update.get("repair_source") == "static_review" and (update.get("rewrite_plan") or update.get("cross_stage_request")):
                status = "repair planned"
            details = _details(name, self.state, update)
            if error:
                details["exception"] = f"{type(error).__name__}: {error}"
            self.steps.append(dict(node=name, status=status, details=details,
                                   seconds=monotonic() - started,
                                   timestamp=datetime.now(timezone.utc).isoformat()))
        except Exception as exc:
            _diagnostic(f"<workflow_report_warning>Could not record {name}: {exc}</workflow_report_warning>")

    def finish(self, state=None, error=None):
        """Write only after preparation acquired output ownership and logging."""
        try:
            logs = active_log_paths()
            if not logs:
                return
            final = state if state is not None else self.state
            log_root = logs[0].parent
            imported = final.get("workflow_mode") == "imported_case"
            owner = log_root.parent.parent if imported else log_root
            if not output_is_owned(owner):
                return
            directory = owner / "report" if imported else owner / ".foamagent"
            path = safe_case_path(directory, "", "workflow_report.html")
            directory.mkdir(parents=True, exist_ok=True)
            content = self._render(final, directory, logs, error)
            path.write_text(content, encoding="utf-8")
            _diagnostic(f"<workflow_report>{path}</workflow_report>")
        except Exception as exc:
            # A display failure must not change CFD results or mask the original error.
            _diagnostic(f"<workflow_report_warning>Could not write report: {exc}</workflow_report_warning>")

    def _render(self, state, directory, logs, error):
        outcome = "failed" if error else state.get("workflow_status", "incomplete")
        runners = [step for step in self.steps if step["node"].endswith("_runner") and step["status"] != "skipped"]
        execution = ("Completed" if runners[-1]["status"] == "completed" else "Unsuccessful") if runners else "Not reached"
        summary = {"workflow": outcome.replace("_", " ").capitalize(), "last_solver_attempt": execution,
                   "physical_correctness": "Not assessed by this report"}
        stopped = ""
        if error or state.get("termination_reason"):
            stopped = '<p class="attention">Stopped: ' + escape(str(error or state["termination_reason"])) + '</p>'
        path = []
        timeline = []
        visits = {}
        for index, step in enumerate(self.steps, 1):
            name = step["node"]
            if step["status"] == "skipped":
                title = "Skip solver execution"
            else:
                visits[name] = visits.get(name, 0) + 1
                title = _TITLES[name] + (f" · attempt {visits[name]}" if visits[name] > 1 else "")
            path.append(f'<a href="#step-{index}">{index}. {escape(title)}</a>')
            status = escape(step["status"])
            timeline.append(
                f'<li id="step-{index}"><details class="step"'
                + (' open' if step["status"] in {"stopped", "attention", "skipped"} else '')
                + f'><summary>{index}. {escape(title)} <span class="badge {status}">{status.capitalize()}</span></summary>'
                + f'<p class="meta">{escape(step["timestamp"])} · {step["seconds"]:.2f} s</p>'
                + (_value(step["details"]) if step["details"] else '<p>No further explanation was recorded.</p>')
                + '</details></li>')
        artifacts = []
        candidates = [*logs, directory / "review_history.jsonl", directory / "static_review.json",
                      directory / "static_review_references.json"]
        if state.get("case_dir"):
            candidates.extend(Path(state["case_dir"]) / name for name in ("similar_case.txt", "Allrun.out", "Allrun.err"))
        for candidate in candidates:
            try:
                candidate = safe_case_path(candidate.parent, "", candidate.name)
                if candidate.is_file():
                    href = quote(os.path.relpath(candidate, directory), safe="/")
                    label = candidate.name + (" (latest)" if candidate.name == "similar_case.txt" else "")
                    artifacts.append(f'<a href="{escape(href)}">{escape(label)}</a>')
            except (OSError, ValueError):
                continue
        journal = self._history(directory)
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'">'
            f'<title>Foam-Agent · Workflow report</title><style>{_STYLE}</style></head><body><main>'
            '<p class="eyebrow">Foam-Agent / Run history</p><h1>Workflow report</h1>'
            '<p>Recorded decisions, checks and repairs from this run. Model explanations are fallible; '
            'successful execution alone does not establish physical correctness.</p>'
            f'<div class="panel overview">{_value(summary)}{stopped}</div>'
            f'<p class="meta">Started {escape(self.started)} · {monotonic() - self.clock:.2f} s</p>'
            f'<p class="meta">Case: {escape(str(state.get("case_dir") or "Not recorded"))}</p>'
            f'<details class="panel"><summary>User requirement</summary>{_value(self.requirement or state.get("user_requirement"))}</details>'
            '<h2>Execution path</h2><p class="meta">Actual order, including retries. Expand a step to inspect its evidence.</p>'
            f'<nav class="path" aria-label="Execution path">{"".join(path)}</nav>'
            f'<ol class="timeline">{"".join(timeline)}</ol>'
            '<h2>Review and repair history</h2>' + journal
            + '<h2>Source records</h2><div class="artifacts">' + "".join(artifacts)
            + '</div></main></body></html>'
        )

    @staticmethod
    def _history(directory):
        entries = []
        try:
            path = safe_case_path(directory, "", "review_history.jsonl")
            if path.is_file():
                with path.open(encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line)
                            if not {"report", "verification", "reason", "review_analysis", "file_diffs", "error_logs", "error", "response"}.intersection(entry):
                                continue
                            title = " · ".join(str(entry[key]).replace("_", " ") for key in ("stage", "event", "timestamp") if entry.get(key))
                            details = {key: value for key, value in entry.items() if key in _HISTORY_FIELDS}
                            entries.append(f'<details class="panel"><summary>{escape(title)}</summary>{_value(details)}</details>')
                        except (ValueError, AttributeError, TypeError):
                            entries.append('<p>A history entry could not be read; see the source journal.</p>')
        except (OSError, ValueError) as exc:
            entries.append(f'<p>History unavailable: {escape(str(exc))}</p>')
        return "".join(entries) or '<p>No review or repair events were recorded.</p>'
