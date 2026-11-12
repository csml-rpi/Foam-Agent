"""Mesh review adapter."""
from services.review import review_mesh

def mesh_reviewer_node(state):
    return review_mesh(state)
