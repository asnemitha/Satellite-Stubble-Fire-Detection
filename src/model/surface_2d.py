"""1D hydraulic routing through the drainage network graph.

Simplified explicit storage-routing scheme.

The model represents each node as a storage volume and each pipe as a
capacity-limited connection between node hydraulic heads.

Coupling with the 2D surface model is mass-conservative:

    external runoff -> node storage -> pipe routing
                              |
                              v
                         surcharge
                              |
                              v
                         2D surface
                              |
                              v
                           re-entry
                              |
                              +----> node storage

This is a lightweight kinematic approximation suitable for real-time
nowcasting. A full Saint-Venant / SWMM-style dynamic-wave solver can replace
the routing core later without changing the graph interface.
"""

from __future__ import annotations

import math

from .drainage_graph import DrainageGraph


DEFAULT_NODE_STORAGE_AREA = 4.0  # m²


def _storage_area(
    node_id: str,
    node_storage_area: float | dict[str, float],
) -> float:
    """Return the effective storage area for a node."""

    if isinstance(node_storage_area, dict):
        area = node_storage_area.get(
            node_id,
            DEFAULT_NODE_STORAGE_AREA,
        )
    else:
        area = node_storage_area

    try:
        area = float(area)
    except (TypeError, ValueError):
        area = DEFAULT_NODE_STORAGE_AREA

    if not math.isfinite(area) or area <= 0.0:
        return DEFAULT_NODE_STORAGE_AREA

    return area


def route_timestep(
    graph: DrainageGraph,
    dt: float,
    node_storage_area: float | dict[str, float] = DEFAULT_NODE_STORAGE_AREA,
) -> None:
    """Advance the 1D drainage network by one timestep.

    Args:
        graph:
            Drainage network containing nodes and pipes.

        dt:
            Timestep in seconds.

        node_storage_area:
            Effective storage/ponding area at each node in m².
            Either a single value for all nodes or a dictionary mapping
            node IDs to areas.

    Notes:
        ``node.inflow`` is an external inflow accumulator in m³/s.
        It is consumed exactly once during this timestep and reset to zero
        before returning.

        Outfall nodes are treated as fixed downstream boundaries. Their
        incoming flow leaves the model domain.
    """

    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be a finite positive number.")

    # ------------------------------------------------------------------
    # 1. Calculate pipe flows.
    #
    # Flow is driven by the difference between the upstream and
    # downstream hydraulic grade lines.
    #
    # We intentionally clamp the driving head to zero because this
    # simplified solver does not model reverse flow.
    # ------------------------------------------------------------------

    for edge in graph.edges.values():

        if edge.from_node not in graph.nodes:
            raise ValueError(
                f"Edge '{edge.id}' references unknown "
                f"from_node '{edge.from_node}'."
            )

        if edge.to_node not in graph.nodes:
            raise ValueError(
                f"Edge '{edge.id}' references unknown "
                f"to_node '{edge.to_node}'."
            )

        upstream = graph.nodes[edge.from_node]
        downstream = graph.nodes[edge.to_node]

        capacity = edge.full_flow_capacity()

        if capacity <= 0.0:
            edge.flow = 0.0
            continue

        # Hydraulic gradient driving the pipe.
        driving_head = max(
            0.0,
            upstream.head - downstream.head,
        )

        # Convert available head into an approximate discharge.
        #
        # [m²] * [m] / [s] = [m³/s]
        demand = (
            _storage_area(
                edge.from_node,
                node_storage_area,
            )
            * driving_head
            / dt
        )

        edge.flow = min(
            capacity,
            max(0.0, demand),
        )

    # ------------------------------------------------------------------
    # 2. Node mass balance.
    #
    # net_flow is positive when water enters a node and negative when
    # water leaves it.
    # ------------------------------------------------------------------

    net_flow: dict[str, float] = {
        node_id: float(node.inflow)
        for node_id, node in graph.nodes.items()
    }

    for edge in graph.edges.values():

        # Water leaves from_node.
        net_flow[edge.from_node] -= edge.flow

        # Water enters to_node.
        net_flow[edge.to_node] = (
            net_flow.get(edge.to_node, 0.0)
            + edge.flow
        )

    # ------------------------------------------------------------------
    # 3. Convert node volume change into hydraulic-head change.
    #
    # dV = Q * dt
    # dH = dV / A
    # ------------------------------------------------------------------

    for node_id, node in graph.nodes.items():

        # An outfall is a boundary condition rather than a storage node.
        # Incoming water is discharged from the model.
        if node.is_outfall:
            continue

        area = _storage_area(
            node_id,
            node_storage_area,
        )

        volume_change = net_flow[node_id] * dt

        head_change = volume_change / area

        node.head += head_change

        # Do not allow the hydraulic head to fall below the pipe invert.
        node.head = max(
            node.invert_elevation,
            node.head,
        )

    # ------------------------------------------------------------------
    # 4. Consume this timestep's external inflows.
    #
    # IMPORTANT:
    # rainfall runoff and surface re-entry are accumulated before
    # route_timestep() and must not persist into the next timestep.
    # ------------------------------------------------------------------

    for node in graph.nodes.values():
        node.inflow = 0.0


def surface_source_terms(
    graph: DrainageGraph,
    dt: float,
    node_storage_area: float | dict[str, float] = DEFAULT_NODE_STORAGE_AREA,
) -> dict[str, float]:
    """Convert 1D surcharge above street level into 2D source flow.

    This operation is deliberately MASS CONSERVATIVE.

    If a node's hydraulic head rises above the ground elevation, the excess
    volume above street level is removed from the 1D node storage and
    returned as a discharge rate for the 2D surface model.

    Therefore:

        volume removed from 1D == volume injected into 2D

    Args:
        graph:
            Drainage network.

        dt:
            Timestep in seconds.

        node_storage_area:
            Effective storage area at each node in m².

    Returns:
        Dictionary mapping node IDs to surface discharge rates in m³/s.
    """

    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be a finite positive number.")

    sources: dict[str, float] = {}

    for node_id, node in graph.nodes.items():

        # Outfalls do not surcharge onto the street.
        if node.is_outfall:
            continue

        excess_head = (
            node.head - node.ground_elevation
        )

        if excess_head <= 0.0:
            continue

        area = _storage_area(
            node_id,
            node_storage_area,
        )

        # Volume physically stored above street level.
        excess_volume = excess_head * area

        if excess_volume <= 0.0:
            continue

        # Convert the excess volume into a discharge rate for this
        # timestep.
        source_rate = excess_volume / dt

        if source_rate <= 0.0:
            continue

        # Remove exactly the volume that will be injected into the
        # 2D surface.
        #
        # This is the key mass-conservation correction.
        node.head = node.ground_elevation

        sources[node_id] = source_rate

    return sources