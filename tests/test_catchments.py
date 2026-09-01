import numpy as np
import pytest

from src.model.catchments import (
    aggregate_runoff_by_catchment,
    build_catchment_map,
    catchment_areas_m2,
)
from src.model.coupled_forecast import ForecastConfig, run_forecast
from src.model.drainage_graph import DrainageGraph, Edge, Node
from src.model.surface_2d import SurfaceGrid


def test_catchment_map_covers_every_cell():
    node_cell_map = {"a": (1, 1), "b": (8, 8)}

    label_grid, node_ids = build_catchment_map(
        node_cell_map,
        shape=(10, 10),
    )

    assert label_grid.shape == (10, 10)
    assert set(np.unique(label_grid)).issubset({0, 1})
    assert node_ids == ["a", "b"]


def test_catchment_map_assigns_nearest_node():
    node_cell_map = {"a": (0, 0), "b": (9, 9)}

    label_grid, node_ids = build_catchment_map(
        node_cell_map,
        shape=(10, 10),
    )

    # A cell right next to "a" should belong to "a"'s catchment (index 0)
    assert label_grid[0, 1] == 0

    # A cell right next to "b" should belong to "b"'s catchment (index 1)
    assert label_grid[9, 8] == 1


def test_catchment_areas_sum_to_total_grid_area():
    node_cell_map = {
        "a": (1, 1),
        "b": (8, 8),
        "c": (5, 1),
    }

    shape = (10, 10)
    cell_size = 5.0

    label_grid, node_ids = build_catchment_map(
        node_cell_map,
        shape,
    )

    areas = catchment_areas_m2(
        label_grid,
        node_ids,
        cell_size**2,
    )

    expected_area = (
        shape[0] * shape[1] * cell_size**2
    )

    assert sum(areas.values()) == pytest.approx(
        expected_area,
        rel=1e-6,
    )

    # Every node should get some catchment
    # on a 10x10 grid with 3 well-spread nodes.
    assert all(a > 0 for a in areas.values())


def test_aggregate_runoff_integrates_over_whole_catchment_not_one_cell():
    """Regression test for the core bug.

    A node's inflow must reflect its whole catchment,
    not just the single cell it sits on.
    """
    node_cell_map = {"a": (2, 2)}
    shape = (10, 10)

    label_grid, node_ids = build_catchment_map(
        node_cell_map,
        shape,
    )

    runoff_grid = np.full(
        shape,
        1e-5,
    )

    cell_area = 25.0

    totals = aggregate_runoff_by_catchment(
        runoff_grid,
        label_grid,
        node_ids,
        cell_area,
    )

    single_cell_value = (
        runoff_grid[2, 2] * cell_area
    )

    # Catchment = entire grid; inflow must be
    # approximately 100x the single-cell value.
    expected_total = (
        shape[0] * shape[1] * single_cell_value
    )

    assert totals["a"] == pytest.approx(
        expected_total,
        rel=1e-6,
    )

    assert totals["a"] > single_cell_value * 50


def test_run_forecast_produces_nonzero_flooding_under_heavy_rain():
    """End-to-end regression test.

    A heavy, sustained rain event over a small
    under-capacity network must produce visible flooding.
    Before the catchment-area fix, this was always zero.
    """
    g = DrainageGraph()

    g.add_node(
        Node(
            id="inlet",
            x=0,
            y=0,
            ground_elevation=10.0,
            invert_elevation=9.3,
        )
    )

    g.add_node(
        Node(
            id="outfall",
            x=50,
            y=0,
            ground_elevation=8.0,
            invert_elevation=7.0,
            is_outfall=True,
        )
    )

    g.add_edge(
        Edge(
            id="p1",
            from_node="inlet",
            to_node="outfall",
            length=50,
            diameter=0.2,
            slope=0.003,
            condition_factor=0.4,
        )
    )

    shape = (20, 20)
    cell_size = 5.0

    dem = np.full(
        shape,
        10.0,
    )

    surface = SurfaceGrid(
        dem,
        cell_size,
    )

    node_cell_map = {
        "inlet": (10, 10),
        "outfall": (10, 19),
    }

    imperviousness = np.full(
        shape,
        0.9,
    )

    cfg = ForecastConfig(
        dt_seconds=15,
        horizon_minutes=60,
        output_interval_minutes=15,
    )

    rainfall_frames = [
        np.full(shape, 90.0)
    ]

    outputs = run_forecast(
        g,
        surface,
        node_cell_map,
        rainfall_frames,
        imperviousness,
        cfg,
    )

    assert max(
        o.max() for o in outputs
    ) > 0.0, (
        "Expected visible flooding under heavy rain "
        "on an undersized, silted pipe"
    )