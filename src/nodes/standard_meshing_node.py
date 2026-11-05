"""Thin main-graph adapter for planned OpenFOAM mesh operations."""

from services import meshing


def standard_meshing_node(state):
    return meshing.execute_meshing(state, meshing.standard_meshing)
