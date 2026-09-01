"""End-to-end coupled 1D-2D forecast loop.

rainfall nowcast -> runoff -> 1D drainage routing -> surcharge -> 2D surface
inundation -> (re-entry into 1D where surface inlets have spare capacity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catchments import (
    aggregate_runoff_by_catchment,
    build_catchment_map,
    catchment_areas_m2,
)
from .drainage_graph import DrainageGraph
from .hydraulic_1d import route_timestep, surface_source_terms
from .surface_2d import SurfaceGrid


@dataclass
class ForecastConfig:
    dt_seconds: float = 30.0
    horizon_minutes: int = 180
    output_interval_minutes: int = 5

    imperviousness_runoff_coeff: float = 0.9

    manhole_base_area_m2: float = 2.0

    street_ponding_fraction: float = 0.015

    elevation_weight: float = 0.0


def rainfall_to_runoff(
    rain_mm_per_hr: np.ndarray,
    imperviousness: np.ndarray,
    cfg: ForecastConfig,
) -> np.ndarray:
    """Convert rainfall intensity from mm/hr to runoff rate in m/s."""

    rain_m_per_s = (rain_mm_per_hr / 1000.0) / 3600.0

    coeff = (
        imperviousness * cfg.imperviousness_runoff_coeff
        + (1.0 - imperviousness) * 0.15
    )

    return rain_m_per_s * coeff


def node_storage_areas_from_catchments(
    catchment_area_m2: dict[str, float],
    cfg: ForecastConfig,
) -> dict[str, float]:
    """Derive a ponding footprint for each drainage node."""

    return {
        node_id: (
            cfg.manhole_base_area_m2
            + cfg.street_ponding_fraction * area
        )
        for node_id, area in catchment_area_m2.items()
    }


def run_forecast(
    graph: DrainageGraph,
    surface: SurfaceGrid,
    node_cell_map: dict[str, tuple[int, int]],
    rainfall_frames: list[np.ndarray],
    imperviousness: np.ndarray,
    cfg: ForecastConfig,
) -> list[np.ndarray]:
    """Run the coupled 1D-2D model across the forecast horizon.

    Returns:
        List of surface depth grids in centimetres, sampled every
        output_interval_minutes.
    """

    # ----------------------------
    # Validate configuration
    # ----------------------------

    if cfg.dt_seconds <= 0:
        raise ValueError("dt_seconds must be greater than zero.")

    if cfg.output_interval_minutes <= 0:
        raise ValueError(
            "output_interval_minutes must be greater than zero."
        )

    if cfg.horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be greater than zero.")

    if not rainfall_frames:
        raise ValueError("rainfall_frames must contain at least one frame.")

    output_seconds = cfg.output_interval_minutes * 60.0

    if output_seconds % cfg.dt_seconds != 0:
        raise ValueError(
            "output_interval_minutes must be an integer multiple "
            "of dt_seconds."
        )

    steps_per_output = int(output_seconds / cfg.dt_seconds)

    total_seconds = cfg.horizon_minutes * 60.0

    if total_seconds % output_seconds != 0:
        raise ValueError(
            "horizon_minutes must be an integer multiple of "
            "output_interval_minutes."
        )

    n_outputs = int(total_seconds / output_seconds)

    # ----------------------------
    # Catchment delineation
    # ----------------------------

    label_grid, node_ids = build_catchment_map(
        node_cell_map,
        surface.dem.shape,
        dem=surface.dem,
        elevation_weight=cfg.elevation_weight,
    )

    catchment_area = catchment_areas_m2(
        label_grid,
        node_ids,
        surface.cell_size**2,
    )

    node_storage_area = node_storage_areas_from_catchments(
        catchment_area,
        cfg,
    )

    # ----------------------------
    # Forecast loop
    # ----------------------------

    outputs: list[np.ndarray] = []

    for out_idx in range(n_outputs):

        # Hold the last available rainfall frame if the nowcast
        # contains fewer frames than the forecast horizon.
        rain_grid = rainfall_frames[
            min(out_idx, len(rainfall_frames) - 1)
        ]

        runoff_grid = rainfall_to_runoff(
            rain_grid,
            imperviousness,
            cfg,
        )

        for _ in range(steps_per_output):

            # 1. Catchment runoff -> drainage nodes
            inflows = aggregate_runoff_by_catchment(
                runoff_grid,
                label_grid,
                node_ids,
                surface.cell_size**2,
            )

            for node_id, volume_rate in inflows.items():
                if node_id not in graph.nodes:
                    continue

                graph.nodes[node_id].inflow += volume_rate

            # 2. Route water through the 1D drainage network
            route_timestep(
                graph,
                cfg.dt_seconds,
                node_storage_area=node_storage_area,
            )

            # 3. Surcharge -> surface
            source_terms = surface_source_terms(graph)

            for node_id, rate in source_terms.items():
                if node_id not in node_cell_map:
                    continue

                r, c = node_cell_map[node_id]

                surface.inject(
                    r,
                    c,
                    rate,
                    cfg.dt_seconds,
                )

            # 4. Advance the 2D surface
            surface.step(cfg.dt_seconds)

            # 5. Surface -> 1D re-entry
            _attempt_reentry(
                graph,
                surface,
                node_cell_map,
                cfg.dt_seconds,
            )

        # Store a copy so later surface updates do not modify
        # previously stored outputs.
        outputs.append(surface.depth_cm().copy())

    return outputs


def _attempt_reentry(
    graph: DrainageGraph,
    surface: SurfaceGrid,
    node_cell_map: dict[str, tuple[int, int]],
    dt_seconds: float,
) -> None:
    """Drain ponded surface water back into the drainage network.

    Re-entry occurs only when:
      1. the inlet is not surcharged,
      2. ponded water exists at the inlet cell, and
      3. downstream pipes have spare capacity.

    The amount removed from the surface is limited both by available
    ponded water and by downstream hydraulic capacity.
    """

    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be greater than zero.")

    cell_area = surface.cell_size**2

    for node_id, node in graph.nodes.items():

        # A surcharged node cannot accept additional surface water.
        if node.is_surcharged:
            continue

        if node_id not in node_cell_map:
            continue

        r, c = node_cell_map[node_id]

        depth = surface.depth[r, c]

        if depth <= 0.0:
            continue

        # Find downstream links.
        downstream = graph.downstream_edges(node_id)

        if not downstream:
            continue

        # Total unused downstream conveyance capacity [m3/s].
        spare_capacity = sum(
            max(
                0.0,
                edge.full_flow_capacity() - edge.flow,
            )
            for edge in downstream
        )

        if spare_capacity <= 0.0:
            continue

        # Volume of water currently ponded in this cell [m3].
        ponded_volume = depth * cell_area

        # Maximum rate needed to remove all ponded water during
        # this timestep [m3/s].
        ponded_rate = ponded_volume / dt_seconds

        # Actual re-entry rate [m3/s].
        reentry_rate = min(
            spare_capacity,
            ponded_rate,
        )

        if reentry_rate <= 0.0:
            continue

        # Convert flow rate to volume over this timestep.
        reentry_volume = reentry_rate * dt_seconds

        # Remove the corresponding depth from the 2D surface.
        depth_removed = reentry_volume / cell_area

        surface.depth[r, c] = max(
            0.0,
            depth - depth_removed,
        )

        # Add the returning water to the 1D node.
        node.inflow += reentry_rate