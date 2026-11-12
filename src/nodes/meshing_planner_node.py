"""Thin adapter for mesh planning in the main workflow."""

from services.meshing import plan_meshing


def meshing_planner_node(state):
    return plan_meshing(state)
