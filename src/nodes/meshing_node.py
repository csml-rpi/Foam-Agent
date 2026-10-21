"""Thin adapter selecting a mesh backend and a single execution pass."""

from services import meshing


def meshing_node(state):
    mesh_type = state.get("mesh_type", "standard_mesh")
    if mesh_type == "gmsh_mesh":
        backend = meshing.gmsh_meshing
    elif mesh_type == "custom_mesh":
        backend = meshing.imported_meshing
    else:
        backend = meshing.standard_meshing
    return meshing.execute_meshing(state, backend)
