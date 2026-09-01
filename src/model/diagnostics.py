"""Node/pipe-level diagnostics -- the "why" behind the depth grid.

Surfaced through the API's /network/diagnostics endpoint and the dashboard's
click-through popups, and used by run_cycle.py's console summary.
"""
from __future__ import annotations

from .drainage_graph import DrainageGraph


def node_diagnostics(graph: DrainageGraph, catchment_area_m2: dict[str, float] | None = None) -> list[dict]:
    catchment_area_m2 = catchment_area_m2 or {}
    rows = []
    for node in graph.nodes.values():
        rows.append({
            "id": node.id,
            "x": node.x,
            "y": node.y,
            "ground_elevation_m": node.ground_elevation,
            "invert_elevation_m": node.invert_elevation,
            "head_m": round(node.head, 3),
            "fill_fraction": round(node.fill_fraction, 3),
            "is_surcharged": node.is_surcharged,
            "is_outfall": node.is_outfall,
            "catchment_area_m2": round(catchment_area_m2.get(node.id, 0.0), 1),
        })
    return rows


def pipe_diagnostics(graph: DrainageGraph) -> list[dict]:
    rows = []
    for edge in graph.edges.values():
        rows.append({
            "id": edge.id,
            "from_node": edge.from_node,
            "to_node": edge.to_node,
            "diameter_m": edge.diameter,
            "condition_factor": edge.condition_factor,
            "full_flow_capacity_m3s": round(edge.full_flow_capacity(), 4),
            "current_flow_m3s": round(edge.flow, 4),
            "utilization": round(edge.utilization, 3),
        })
    return rows


def network_summary(graph: DrainageGraph, catchment_area_m2: dict[str, float] | None = None) -> dict:
    nodes = node_diagnostics(graph, catchment_area_m2)
    pipes = pipe_diagnostics(graph)
    return {
        "nodes": nodes,
        "pipes": pipes,
        "n_surcharged": sum(1 for n in nodes if n["is_surcharged"]),
        "n_nodes": len(nodes),
        "n_pipes": len(pipes),
        "worst_pipe_utilization": max((p["utilization"] for p in pipes), default=0.0),
    }
