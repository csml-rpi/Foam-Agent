"""Initial solver-stage review adapter."""

from services.static_review import review_stage_inputs


def static_review_node(state):
    return review_stage_inputs(state, "solver")
