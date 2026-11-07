"""Solver review adapter."""
from services.review import review_solver

def solver_reviewer_node(state):
    return review_solver(state)
