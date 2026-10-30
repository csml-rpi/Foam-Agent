"""Restore the checkpoint without promoting an incomplete run to success."""

from services.best_snapshot import restore_best_snapshot


def restore_best_node(state):
    return restore_best_snapshot(state)
