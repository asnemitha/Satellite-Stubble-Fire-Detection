"""End-to-end coupled 1D-2D forecast loop.

rainfall nowcast -> runoff -> 1D drainage routing -> surcharge -> 2D surface
inundation -> (re-entry into 1D where surface inlets have spare capacity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catchments import aggregate_runoff_by_catchment, build_catchment_map, catchment_areas_m2
from .drainage_graph import DrainageGraph
from .hydraulic_1d import route_timestep, surface_source_terms
from .surface_2d import SurfaceGrid


@dataclass
class ForecastConfig:
    dt_seconds: float = 30.0
    horizon_minutes: int = 180
    output_interval_minutes: int = 5
    imperviousness_runoff_coeff: float = 0.9  # fraction of rain on impervious surface becoming runoff
    manhole_base_area_m2: float = 2.0         # bare chamber footprint, every node gets at least this
    street_ponding_fraction: float = 0.015    # fraction of a node's catchment area that can pond at
                                               # the inlet/junction before dispersing further -- this is
                                               # what turns "a 5000 m^2 catchment" into "the low point of
                                               # that street has ~75 m^2 of surface to pond on", not the
                                               # whole catchment area itself
    elevation_weight: float = 0.0             # >0 biases catchment delineation away from uphill inlets


def rainfall_to_runoff(rain_mm_per_hr: np.ndarray, imperviousness: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """Convert a rainfall intensity grid (mm/hr) to a runoff rate grid (m/s)."""
    rain_m_per_s = (rain_mm_per_hr / 1000.0) / 3600.0
    coeff = imperviousness * cfg.imperviousness_runoff_coeff + (1 - imperviousness) * 0.15
    return rain_m_per_s * coeff


def node_storage_areas_from_catchments(
    catchment_area_m2: dict[str, float], cfg: ForecastConfig
) -> dict[str, float]:
    """Derive a physically-motivated ponding footprint per node from how much
    terrain actually drains to it, instead of using one constant for every
    node regardless of catchment size (see ForecastConfig docstring above)."""
    return {
        nid: cfg.manhole_base_area_m2 + cfg.street_ponding_fraction * area
        for nid, area in catchment_area_m2.items()
    }


def run_forecast(
    graph: DrainageGraph,
    surface: SurfaceGrid,
    node_cell_map: dict[str, tuple[int, int]],
    rainfall_frames: list[np.ndarray],  # one rainfall intensity grid (mm/hr) per output interval, 0..horizon
    imperviousness: np.ndarray,
    cfg: ForecastConfig,
) -> list[np.ndarray]:
    """Run the coupled model across the forecast horizon.

    Returns a list of depth-in-cm grids, one per output_interval_minutes step,
    covering t+0 .. t+horizon_minutes.
    """
    steps_per_output = int((cfg.output_interval_minutes * 60) / cfg.dt_seconds)
    n_outputs = cfg.horizon_minutes // cfg.output_interval_minutes

    # --- catchment delineation (fixes the 1:1 cell<->node bug) ---
    # Each inlet now receives runoff integrated over the whole patch of
    # terrain that drains to it (nearest-inlet/Voronoi catchment), not just
    # the single grid cell it happens to sit on.
    label_grid, node_ids = build_catchment_map(
        node_cell_map, surface.dem.shape, dem=surface.dem, elevation_weight=cfg.elevation_weight
    )
    catchment_area = catchment_areas_m2(label_grid, node_ids, surface.cell_size**2)
    node_storage_area = node_storage_areas_from_catchments(catchment_area, cfg)

    outputs: list[np.ndarray] = []
    for out_idx in range(n_outputs):
        rain_grid = rainfall_frames[min(out_idx, len(rainfall_frames) - 1)]
        runoff_grid = rainfall_to_runoff(rain_grid, imperviousness, cfg)

        for _ in range(steps_per_output):
            # 1. integrate runoff over each node's full catchment, not just its own cell
            inflows = aggregate_runoff_by_catchment(runoff_grid, label_grid, node_ids, surface.cell_size**2)
            for node_id, vol_rate in inflows.items():
                graph.nodes[node_id].inflow += vol_rate

            # 2. route through the 1D network
            route_timestep(graph, cfg.dt_seconds, node_storage_area=node_storage_area)

            # 3. surcharge -> 2D source injection
            for node_id, rate in surface_source_terms(graph).items():
                r, c = node_cell_map[node_id]
                surface.inject(r, c, rate, cfg.dt_seconds)

            # 4. advance the 2D surface solver
            surface.step(cfg.dt_seconds)

            # 5. re-entry: where a node is no longer surcharged and there's
            # spare downstream pipe capacity, let ponded water drain back
            # into the network (reciprocal 1D<->2D coupling, not just
            # one-way overflow -- see DESIGN.md sec 4.3)
            _attempt_reentry(graph, surface, node_cell_map)

        outputs.append(surface.depth_cm().copy())

    return outputs


def _attempt_reentry(graph: DrainageGraph, surface: SurfaceGrid, node_cell_map: dict[str, tuple[int, int]]) -> None:
    """Let ponded surface water drain back into the pipe network at inlets
    that are no longer surcharged and have spare downstream capacity."""
    for node_id, node in graph.nodes.items():
        if node.is_surcharged:
            continue
        r, c = node_cell_map[node_id]
        depth = surface.depth[r, c]
        if depth <= 0:
            continue
        downstream = graph.downstream_edges(node_id)
        spare = sum(max(0.0, e.full_flow_capacity() - e.flow) for e in downstream)
        if spare <= 0:
            continue
        # cap re-entry so a single step can't drain more water than is present
        reentry_rate = min(spare, depth * (surface.cell_size**2) / 30.0)
        surface.depth[r, c] = max(0.0, depth - (reentry_rate * 30.0) / (surface.cell_size**2))
        node.inflow += reentry_rate
