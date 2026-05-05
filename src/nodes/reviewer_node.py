# reviewer_node.py
import os
import glob
import re
import shutil
from services.review import review_error_logs
from logger import log_review


def _compute_progress_score(case_dir: str, error_logs: list) -> float:
    """Highest timestep reached across log.* files; fall back to -len(error_logs)."""
    best_t = None
    for log_path in glob.glob(os.path.join(case_dir, "log.*")):
        try:
            with open(log_path, errors="ignore") as f:
                for line in f:
                    m = re.match(r"^Time = ([\d.eE+\-]+)", line)
                    if m:
                        t = float(m.group(1))
                        if best_t is None or t > best_t:
                            best_t = t
        except Exception:
            pass
    return float(best_t) if best_t is not None else -len(error_logs)


def reviewer_node(state):
    """Reviewer node: single-call review (FA 1.1.0 style) + best-state snapshot."""
    print("<reviewer>")
    if len(state["error_logs"]) == 0:
        print("No error to review.")
        print("</reviewer>")
        return state

    log_review(str(state["error_logs"]), "error_logs")

    case_dir = state["case_dir"]
    error_logs = state.get("error_logs", [])
    loop_count = state.get("loop_count", 0)
    history_text = state.get("history_text") or []

    # Best-state snapshot before this loop's rewrite can regress it
    snapshot_updates = {}
    score = _compute_progress_score(case_dir, error_logs)
    best_score = state.get("best_progress_score")
    if best_score is None:
        best_score = float("-inf")
    if score > best_score:
        snap = case_dir.rstrip("/") + "_best"
        if os.path.exists(snap):
            shutil.rmtree(snap)
        shutil.copytree(case_dir, snap)
        print(f"<snapshot>progress={score:.4g} > {best_score:.4g}, saved to {snap}</snapshot>")
        snapshot_updates = {"best_case_snapshot_dir": snap, "best_progress_score": score}

    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get("tutorial_reference", ""),
        foamfiles=state.get("foamfiles"),
        error_logs=error_logs,
        user_requirement=state.get("user_requirement", ""),
        similar_case_advice=state.get("similar_case_advice"),
        history_text=history_text,
        loop_count=loop_count,
    )

    log_review(review_content, "review_analysis")

    print("</reviewer>")

    return {
        **snapshot_updates,
        "history_text": updated_history,
        "review_analysis": review_content,
        "loop_count": loop_count + 1,
        "input_writer_mode": "rewrite",
    }
