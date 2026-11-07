"""Thin main-graph adapter for custom .msh input or existing polyMesh."""

from services import meshing


def custom_meshing_node(state):
    return meshing.execute_meshing(state, meshing.custom_meshing)
