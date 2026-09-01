import numpy as np
import pytest

from src.model.drainage_graph import DrainageGraph, Edge, Node
from src.model.hydraulic_1d import route_timestep, surface_source_terms
from src.model.surface_2d import SurfaceGrid
from src.model.coupled_forecast import ForecastConfig, rainfall_to_runoff, run_forecast


def test_node_surcharge_flag():
    n = Node(id="a", x=0, y=0, ground_elevation=10.0, invert_elevation=8.0)
    assert not n.is_surcharged
    n.head = 10.0
    assert n.is_surcharged


def test_edge_capacity_zero_slope():
    e = Edge(id="e1", from_node="a", to_node="b", length=10, diameter=0.3, slope=0.0)
    assert e.full_flow_capacity() == 0.0


def test_edge_capacity_positive_slope():
    e = Edge(id="e1", from_node="a", to_node="b", length=10, diameter=0.3, slope=0.01)
    assert e.full_flow_capacity() > 0.0


def test_condition_factor_derates_capacity():
    full = Edge(id="e1", from_node="a", to_node="b", length=10, diameter=0.3, slope=0.01, condition_factor=1.0)
    silted = Edge(id="e2", from_node="a", to_node="b", length=10, diameter=0.3, slope=0.01, condition_factor=0.5)
    assert silted.full_flow_capacity() == pytest.approx(full.full_flow_capacity() * 0.5)


def test_route_timestep_conserves_flow_direction():
    g = DrainageGraph()
    g.add_node(Node(id="a", x=0, y=0, ground_elevation=10.0, invert_elevation=8.0))
    g.add_node(Node(id="b", x=10, y=0, ground_elevation=9.0, invert_elevation=7.0, is_outfall=True))
    g.add_edge(Edge(id="p1", from_node="a", to_node="b", length=10, diameter=0.3, slope=0.02))

    g.nodes["a"].inflow = 0.1  # m3/s
    route_timestep(g, dt=10.0)
    assert g.nodes["a"].head >= g.nodes["a"].invert_elevation


def test_surcharge_generates_source_term():
    g = DrainageGraph()
    g.add_node(Node(id="a", x=0, y=0, ground_elevation=10.0, invert_elevation=8.0))
    g.nodes["a"].head = 10.5  # above ground
    sources = surface_source_terms(g)
    assert "a" in sources
    assert sources["a"] > 0


def test_no_surcharge_no_source_term():
    g = DrainageGraph()
    g.add_node(Node(id="a", x=0, y=0, ground_elevation=10.0, invert_elevation=8.0))
    g.nodes["a"].head = 9.0
    assert surface_source_terms(g) == {}


def test_surface_grid_inject_raises_depth():
    dem = np.full((5, 5), 10.0)
    grid = SurfaceGrid(dem, cell_size=5.0)
    assert grid.depth[2, 2] == 0.0
    grid.inject(2, 2, volume_rate=1.0, dt=10.0)
    assert grid.depth[2, 2] > 0.0


def test_surface_grid_step_runs_without_error():
    dem = np.full((5, 5), 10.0)
    dem[2, 2] = 9.5
    grid = SurfaceGrid(dem, cell_size=5.0)
    grid.inject(2, 2, volume_rate=2.0, dt=5.0)
    grid.step(dt=1.0)
    assert np.all(grid.depth >= 0)


def test_rainfall_to_runoff_scales_with_imperviousness():
    cfg = ForecastConfig()
    rain = np.full((3, 3), 50.0)
    low_imperv = np.full((3, 3), 0.1)
    high_imperv = np.full((3, 3), 0.9)
    runoff_low = rainfall_to_runoff(rain, low_imperv, cfg)
    runoff_high = rainfall_to_runoff(rain, high_imperv, cfg)
    assert runoff_high.mean() > runoff_low.mean()


def test_run_forecast_returns_expected_frame_count():
    g = DrainageGraph()
    g.add_node(Node(id="a", x=0, y=0, ground_elevation=10.0, invert_elevation=9.5))
    g.add_node(Node(id="b", x=10, y=0, ground_elevation=9.5, invert_elevation=9.0, is_outfall=True))
    g.add_edge(Edge(id="p1", from_node="a", to_node="b", length=10, diameter=0.2, slope=0.005))

    dem = np.full((4, 4), 10.0)
    surface = SurfaceGrid(dem, cell_size=5.0)
    node_cell_map = {"a": (1, 1), "b": (2, 2)}
    imperviousness = np.full((4, 4), 0.8)
    cfg = ForecastConfig(dt_seconds=10, horizon_minutes=20, output_interval_minutes=10)
    rainfall_frames = [np.full((4, 4), 40.0)]

    outputs = run_forecast(g, surface, node_cell_map, rainfall_frames, imperviousness, cfg)
    assert len(outputs) == 2
    assert all(o.shape == (4, 4) for o in outputs)
    assert all((o >= 0).all() for o in outputs)
