import os
import shutil


def restore_best_node(state):
    """Restore the best-snapshot case directory when the review loop exits at max_loop."""
    snap = state.get("best_case_snapshot_dir")
    if snap and os.path.exists(snap):
        case_dir = state["case_dir"]
        shutil.rmtree(case_dir)
        shutil.copytree(snap, case_dir)
        print(
            f"Restored best snapshot from {snap} "
            f"(score={state.get('best_progress_score')})"
        )
    else:
        print("No best snapshot to restore.")
    return {}
