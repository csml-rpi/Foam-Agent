"""Thin main-graph adapter for Gmsh generation and conversion."""

from services import meshing


def gmsh_meshing_node(state):
    return meshing.execute_meshing(state, meshing.gmsh_meshing)
