"""Thin adapter for mesh and solver review and repair routing."""

from services.review import review_case


def reviewer_node(state):
    return review_case(state)
