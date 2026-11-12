"""Thin adapter for shared non-CAD input preparation."""

from services.preparation import prepare_inputs


def preparation_node(state):
    return prepare_inputs(state)
