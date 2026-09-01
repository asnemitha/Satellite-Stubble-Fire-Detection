"""Flood-aware routing.

Intersects a road network graph with a forecast depth grid: any road edge
passing through a cell whose depth exceeds the safety threshold is marked
impassable (or heavily penalized), then shortest-path routing runs on the
remaining/penalized graph.

Production target: swap the toy grid-graph builder for a real OSM road
network (via OSMnx or an OSRM/Valhalla custom cost profile) — the routing
logic (edge cost from depth) stays the same either way.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


def depth_at_cell(
    depth_cm: np.ndarray,
    row: int,
    col: int,
) -> float:
    if (
        0 <= row < depth_cm.shape[0]
        and 0 <= col < depth_cm.shape[1]
    ):
        return float(depth_cm[row, col])

    return 0.0


def build_road_graph_from_grid(
    shape: tuple[int, int],
) -> nx.Graph:
    """Build a toy 4-connected grid road graph.

    Each node represents a street intersection at a DEM/surface grid
    cell. Replace this with a real OSM-derived graph in production.
    Downstream cost assignment and routing are independent of how the
    graph is built.
    """
    rows, cols = shape
    graph = nx.Graph()

    for r in range(rows):
        for c in range(cols):
            graph.add_node((r, c))

            if r > 0:
                graph.add_edge(
                    (r, c),
                    (r - 1, c),
                    base_length=1.0,
                )

            if c > 0:
                graph.add_edge(
                    (r, c),
                    (r, c - 1),
                    base_length=1.0,
                )

    return graph


def assign_flood_costs(
    graph: nx.Graph,
    depth_cm: np.ndarray,
    safe_depth_cm: float,
    impassable_multiplier: float = 1000.0,
) -> nx.Graph:
    """Set edge weights based on forecast flood depth.

    An edge whose either endpoint exceeds safe_depth_cm gets a heavy
    multiplier rather than outright removal. This allows the router
    to find a route through a partially flooded network if there is
    truly no alternative.
    """
    for u, v, data in graph.edges(data=True):
        depth_u = depth_at_cell(
            depth_cm,
            *u,
        )

        depth_v = depth_at_cell(
            depth_cm,
            *v,
        )

        depth = max(
            depth_u,
            depth_v,
        )

        flooded = depth > safe_depth_cm

        penalty = (
            impassable_multiplier
            if flooded
            else 1.0
        )

        data["weight"] = (
            data.get("base_length", 1.0)
            * penalty
        )

        data["flooded"] = flooded

    return graph


def shortest_safe_path(
    graph: nx.Graph,
    origin_cell: tuple[int, int],
    dest_cell: tuple[int, int],
) -> tuple[list[tuple[int, int]], int]:
    """Return the path and number of flooded edges traversed."""
    path = nx.shortest_path(
        graph,
        origin_cell,
        dest_cell,
        weight="weight",
    )

    flooded_count = 0

    for u, v in zip(
        path[:-1],
        path[1:],
    ):
        if graph.edges[u, v].get("flooded"):
            flooded_count += 1

    return path, flooded_count


def latlon_to_cell(
    lat: float,
    lon: float,
    origin: tuple[float, float],
    cell_size_m: float,
) -> tuple[int, int]:
    lat0, lon0 = origin

    row = int(
        (lat0 - lat)
        * 111_000
        / cell_size_m
    )

    col = int(
        (lon - lon0)
        * 111_000
        * math.cos(math.radians(lat0))
        / cell_size_m
    )

    return row, col


def cell_to_latlon(
    row: int,
    col: int,
    origin: tuple[float, float],
    cell_size_m: float,
) -> tuple[float, float]:
    lat0, lon0 = origin

    lat = (
        lat0
        - (row * cell_size_m) / 111_000
    )

    lon = (
        lon0
        + (col * cell_size_m)
        / (
            111_000
            * math.cos(math.radians(lat0))
        )
    )

    return lat, lon