"""Meshing child graph with independent planning and backend nodes."""

from langgraph.graph import END, START, StateGraph
from utils import GraphState
from services import meshing


def meshing_planner_node(state):
    return meshing.plan_meshing(state)


def standard_meshing_node(state):
    return meshing.execute_meshing(state, meshing.standard_meshing)


def gmsh_meshing_node(state):
    return meshing.execute_meshing(state, meshing.gmsh_meshing)


def imported_meshing_node(state):
    return meshing.execute_meshing(state, meshing.imported_meshing)


def route_mesh_backend(state):
    return {
        "standard_mesh": "standard_meshing",
        "gmsh_mesh": "gmsh_meshing",
        "custom_mesh": "imported_meshing",
    }[state["mesh_type"]]


def build_meshing_subgraph():
    graph = StateGraph(GraphState)
    graph.add_node("meshing_planner", meshing_planner_node)
    graph.add_node("standard_meshing", standard_meshing_node)
    graph.add_node("gmsh_meshing", gmsh_meshing_node)
    graph.add_node("imported_meshing", imported_meshing_node)
    graph.add_edge(START, "meshing_planner")
    graph.add_conditional_edges("meshing_planner", route_mesh_backend, {
        name: name for name in ("standard_meshing", "gmsh_meshing", "imported_meshing")
    })
    for name in ("standard_meshing", "gmsh_meshing", "imported_meshing"):
        graph.add_edge(name, END)
    return graph.compile()


meshing_node = build_meshing_subgraph()
