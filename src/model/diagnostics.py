"""Node/pipe-level diagnostics -- the "why" behind the depth grid.

Surfaced through the API's /network/diagnostics endpoint and the dashboard's
click-through popups, and used by run_cycle.py's console summary.
"""

from __future__ import annotations

import math

from .drainage_graph import DrainageGraph


def _safe_round(value: float, digits: int) -> float:
    """Round a numeric diagnostic value while preventing NaN/Inf in JSON."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(value):
        return 0.0

    return round(value, digits)


def node_diagnostics(
    graph: DrainageGraph,
    catchment_area_m2: dict[str, float] | None = None,
) -> list[dict]:
    """Return diagnostic information for every drainage node."""

    areas = catchment_area_m2 if catchment_area_m2 is not None else {}

    rows: list[dict] = []

    for node in graph.nodes.values():
        rows.append(
            {
                "id": node.id,
                "x": _safe_round(node.x, 3),
                "y": _safe_round(node.y, 3),
                "ground_elevation_m": _safe_round(
                    node.ground_elevation, 3
                ),
                "invert_elevation_m": _safe_round(
                    node.invert_elevation, 3
                ),
                "head_m": _safe_round(node.head, 3),
                "fill_fraction": _safe_round(
                    node.fill_fraction, 3
                ),
                "is_surcharged": bool(node.is_surcharged),
                "is_outfall": bool(node.is_outfall),
                "catchment_area_m2": _safe_round(
                    areas.get(node.id, 0.0), 1
                ),
            }
        )

    return rows


def pipe_diagnostics(graph: DrainageGraph) -> list[dict]:
    """Return diagnostic information for every drainage pipe."""

    rows: list[dict] = []

    for edge in graph.edges.values():
        capacity = edge.full_flow_capacity()
        utilization = edge.utilization

        rows.append(
            {
                "id": edge.id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "diameter_m": _safe_round(edge.diameter, 3),
                "condition_factor": _safe_round(
                    edge.condition_factor, 3
                ),
                "full_flow_capacity_m3s": _safe_round(
                    capacity, 4
                ),
                "current_flow_m3s": _safe_round(
                    edge.flow, 4
                ),
                "utilization": _safe_round(
                    utilization, 3
                ),
            }
        )

    return rows


def network_summary(
    graph: DrainageGraph,
    catchment_area_m2: dict[str, float] | None = None,
) -> dict:
    """Return complete network diagnostics and headline statistics."""

    nodes = node_diagnostics(
        graph,
        catchment_area_m2,
    )

    pipes = pipe_diagnostics(graph)

    # Calculate from the actual graph values rather than the rounded
    # diagnostic representation.
    worst_pipe_utilization = max(
        (
            float(edge.utilization)
            for edge in graph.edges.values()
            if math.isfinite(float(edge.utilization))
        ),
        default=0.0,
    )

    return {
        "nodes": nodes,
        "pipes": pipes,
        "n_surcharged": sum(
            1 for node in nodes if node["is_surcharged"]
        ),
        "n_nodes": len(nodes),
        "n_pipes": len(pipes),
        "worst_pipe_utilization": round(
            worst_pipe_utilization,
            3,
        ),
    }