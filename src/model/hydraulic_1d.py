"""1D hydraulic routing through the drainage network graph.

Simplified explicit storage-routing scheme (SWMM-style dynamic wave is the
production target; this is a lightweight kinematic approximation suitable for
real-time nowcasting and easy to swap out for a full Saint-Venant solver).
"""
from __future__ import annotations

from .drainage_graph import DrainageGraph, Node

GRAVITY = 9.81
DEFAULT_NODE_STORAGE_AREA = 4.0  # m^2, bare manhole chamber fallback when no per-node value is given


def route_timestep(
    graph: DrainageGraph,
    dt: float,
    node_storage_area: float | dict[str, float] = DEFAULT_NODE_STORAGE_AREA,
) -> None:
    """Advance the drainage network by one timestep of length dt (seconds).

    node_storage_area: effective ponding area (m^2) at a node once it's full,
    representing the manhole chamber / local street cross-section — this is what
    lets HGL rise above ground and produce a surface source term rather than an
    unbounded head. Pass a single float to use the same value everywhere, or a
    dict[node_id -> area] so a node serving a large catchment (a junction/low
    point with more street frontage) gets a physically larger ponding footprint
    than an inlet with a small catchment. See
    `coupled_forecast.node_storage_areas_from_catchments` for how this is derived.
    """
    def storage_area(node_id: str) -> float:
        if isinstance(node_storage_area, dict):
            return node_storage_area.get(node_id, DEFAULT_NODE_STORAGE_AREA)
        return node_storage_area

    # 1. Compute attempted outflow on every edge from upstream node head.
    # Use the net head difference between upstream and downstream nodes as
    # the driving head: this prevents flow from being computed uphill when
    # the downstream node has built up backpressure (backwater effect).
    # Without this fix, a surcharged downstream node has high head but the
    # solver would still compute positive outflow from the upstream node,
    # effectively pumping water against the hydraulic gradient.
    for edge in graph.edges.values():
        upstream = graph.nodes[edge.from_node]
        downstream = graph.nodes[edge.to_node]
        capacity = edge.full_flow_capacity()
        # Net hydraulic head driving flow from upstream -> downstream.
        # Clamped to 0 so we never reverse flow in this kinematic scheme.
        driving_head = max(0.0, upstream.head - downstream.head)
        # simple proportional draw-down toward capacity, capped by available head
        demand = storage_area(edge.from_node) * driving_head / dt
        edge.flow = min(capacity, demand)

    # 2. Update node mass balance: inflow (external + upstream edges) - outflow (downstream edges)
    net_flow: dict[str, float] = {nid: n.inflow for nid, n in graph.nodes.items()}
    for edge in graph.edges.values():
        net_flow[edge.to_node] = net_flow.get(edge.to_node, 0.0) + edge.flow
        net_flow[edge.from_node] = net_flow.get(edge.from_node, 0.0) - edge.flow

    for node_id, node in graph.nodes.items():
        if node.is_outfall:
            continue
        d_head = (net_flow[node_id] * dt) / storage_area(node_id)
        node.head = max(node.invert_elevation, node.head + d_head)

    # reset external inflow accumulator for next timestep's assignment
    for node in graph.nodes.values():
        node.inflow = 0.0


def surface_source_terms(graph: DrainageGraph) -> dict[str, float]:
    """Excess flow (m3/s) at each surcharged node — the source injected into the 2D solver."""
    sources: dict[str, float] = {}
    for node in graph.surcharged_nodes():
        # excess head above ground, converted to an equivalent discharge estimate
        excess_head = node.head - node.ground_elevation
        if excess_head > 0:
            sources[node.id] = excess_head * 2.0  # placeholder weir-type relation; calibrate per node
    return sources
