import numpy as np

from src.api.routing import assign_flood_costs, build_road_graph_from_grid, shortest_safe_path


def test_route_avoids_flooded_cells():
    shape = (5, 5)
    depth = np.zeros(shape)
    depth[1:, 2] = 50.0  # column 2 flooded except row 0 -> a detour exists via the top row

    graph = build_road_graph_from_grid(shape)
    graph = assign_flood_costs(graph, depth, safe_depth_cm=15.0)

    path, flooded_count = shortest_safe_path(graph, (2, 0), (2, 4))
    assert flooded_count == 0  # router should detour via row 0 to cross column 2
    assert (0, 2) in path  # the only dry crossing of column 2


def test_route_crosses_flood_when_no_detour_exists():
    shape = (5, 5)
    depth = np.zeros(shape)
    depth[:, 2] = 50.0  # entire column 2 flooded -> no dry crossing exists anywhere

    graph = build_road_graph_from_grid(shape)
    graph = assign_flood_costs(graph, depth, safe_depth_cm=15.0)

    path, flooded_count = shortest_safe_path(graph, (2, 0), (2, 4))
    # can't avoid crossing entirely; router should still find the minimum-cost crossing
    assert flooded_count >= 1


def test_route_direct_when_no_flooding():
    shape = (5, 5)
    depth = np.zeros(shape)
    graph = build_road_graph_from_grid(shape)
    graph = assign_flood_costs(graph, depth, safe_depth_cm=15.0)

    path, flooded_count = shortest_safe_path(graph, (0, 0), (0, 4))
    assert flooded_count == 0
    assert len(path) == 5  # straight line, no detour needed
