import numpy as np

from src.api.routing import (
    assign_flood_costs,
    build_road_graph_from_grid,
    shortest_safe_path,
)


def test_route_avoids_flooded_cells():
    shape = (5, 5)
    depth = np.zeros(shape)

    # Column 2 is flooded except row 0.
    # A detour exists via the top row.
    depth[1:, 2] = 50.0

    graph = build_road_graph_from_grid(shape)

    graph = assign_flood_costs(
        graph,
        depth,
        safe_depth_cm=15.0,
    )

    path, flooded_count = shortest_safe_path(
        graph,
        (2, 0),
        (2, 4),
    )

    assert flooded_count == 0
    assert (0, 2) in path


def test_route_crosses_flood_when_no_detour_exists():
    shape = (5, 5)
    depth = np.zeros(shape)

    # Entire column 2 is flooded.
    # No dry crossing exists anywhere.
    depth[:, 2] = 50.0

    graph = build_road_graph_from_grid(shape)

    graph = assign_flood_costs(
        graph,
        depth,
        safe_depth_cm=15.0,
    )

    path, flooded_count = shortest_safe_path(
        graph,
        (2, 0),
        (2, 4),
    )

    # Can't avoid crossing entirely;
    # router should find the minimum-cost crossing.
    assert flooded_count >= 1


def test_route_direct_when_no_flooding():
    shape = (5, 5)
    depth = np.zeros(shape)

    graph = build_road_graph_from_grid(shape)

    graph = assign_flood_costs(
        graph,
        depth,
        safe_depth_cm=15.0,
    )

    path, flooded_count = shortest_safe_path(
        graph,
        (0, 0),
        (0, 4),
    )

    assert flooded_count == 0
    assert len(path) == 5