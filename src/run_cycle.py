"""Orchestration entry point: runs one full nowcast cycle.

  1. Pull rainfall nowcast frames (radar source)
  2. Pull DEM + imperviousness (terrain source)
  3. Load drainage network graph
  4. Run the coupled 1D-2D forecast
  5. Publish results into the API's forecast store

In production this is triggered by a scheduler (Airflow/Temporal) whenever
new radar data lands, on a ~5-10 min cadence. Here it runs once, standalone,
against synthetic data so the whole pipeline is exercisable end-to-end
without any real external data source.

Usage:
    python -m src.run_cycle                     # default "heavy" scenario
    python -m src.run_cycle --scenario extreme   # pick a scenario (see configs/scenarios.py)
    python -m src.run_cycle --scenario light --serve
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from configs.scenarios import SCENARIOS, get_scenario
from src.ingestion.dem_loader import add_depression, synthetic_dem, synthetic_imperviousness
from src.ingestion.radar_nowcast import SyntheticRadarSource
from src.model.catchments import build_catchment_map, catchment_areas_m2
from src.model.coupled_forecast import ForecastConfig, run_forecast
from src.model.diagnostics import network_summary
from src.model.drainage_graph import DrainageGraph, Edge, Node

# "Chronic flood point" demo story: one branch's outlet pipe is undersized
# and heavily silted, and its catchment sits in a genuine local depression --
# the combination Indian municipal post-mortems repeatedly flag as the cause
# of the same junction flooding every single monsoon (see DESIGN.md sec 7).
CHRONIC_BRANCH_INDEX = 1


def build_demo_drainage_network(shape: tuple[int, int], cell_size: float) -> tuple[DrainageGraph, dict, dict]:
    """A branching synthetic storm-drain network: several streets of inlets
    feeding branch collectors, which feed a trunk main to a single outfall --
    standing in for a real municipal GIS import (see src/ingestion/drainage_loader.py
    for the real-data path).

    Returns (graph, node_cell_map, meta) where meta carries demo annotations
    (e.g. which node is the chronic flood point) for the UI/README narrative.
    """
    rows, cols = shape
    graph = DrainageGraph()
    node_cell_map: dict[str, tuple[int, int]] = {}

    n_branches = 4
    inlets_per_branch = 5
    branch_cols = [int(cols * (i + 1) / (n_branches + 1)) for i in range(n_branches)]
    collector_row = int(rows * 0.68)
    outfall_r, outfall_c = int(rows * 0.68), cols - 3

    graph.add_node(Node(
        id="outfall", x=outfall_c * cell_size, y=outfall_r * cell_size,
        ground_elevation=6.0, invert_elevation=4.0, is_outfall=True,
    ))
    node_cell_map["outfall"] = (outfall_r, outfall_c)

    collector_nodes: list[str] = []
    for b, col in enumerate(branch_cols):
        prev_id = None
        r = 0
        for i in range(inlets_per_branch):
            r = int(rows * (0.08 + 0.5 * i / (inlets_per_branch - 1)))
            node_id = f"b{b}_inlet{i}"
            elev = 9.5 - 0.06 * i
            graph.add_node(Node(
                id=node_id, x=col * cell_size, y=r * cell_size,
                ground_elevation=elev, invert_elevation=elev - 1.1,
            ))
            node_cell_map[node_id] = (r, col)
            if prev_id:
                graph.add_edge(Edge(
                    id=f"pipe_{prev_id}_{node_id}", from_node=prev_id, to_node=node_id,
                    length=cell_size * rows * 0.5 / (inlets_per_branch - 1),
                    diameter=0.45, slope=0.006, condition_factor=0.85,
                ))
            prev_id = node_id

        collector_id = f"collector_{b}"
        c_elev = 7.6
        graph.add_node(Node(
            id=collector_id, x=col * cell_size, y=collector_row * cell_size,
            ground_elevation=c_elev, invert_elevation=c_elev - 1.3,
        ))
        node_cell_map[collector_id] = (collector_row, col)
        is_chronic = b == CHRONIC_BRANCH_INDEX
        graph.add_edge(Edge(
            id=f"pipe_{prev_id}_{collector_id}", from_node=prev_id, to_node=collector_id,
            length=cell_size * abs(collector_row - r),
            diameter=0.25 if is_chronic else 0.6,
            slope=0.005,
            condition_factor=0.35 if is_chronic else 0.85,  # heavily silted at the chronic spot
        ))
        collector_nodes.append(collector_id)

    prev = None
    for cid in collector_nodes:
        if prev:
            graph.add_edge(Edge(
                id=f"trunk_{prev}_{cid}", from_node=prev, to_node=cid,
                length=cell_size * (branch_cols[collector_nodes.index(cid)] - branch_cols[collector_nodes.index(prev)]),
                diameter=0.6, slope=0.003, condition_factor=0.9,
            ))
        prev = cid
    graph.add_edge(Edge(
        id="trunk_to_outfall", from_node=collector_nodes[-1], to_node="outfall",
        length=cell_size * (cols - branch_cols[-1]), diameter=0.75, slope=0.004, condition_factor=0.9,
    ))

    meta = {
        "chronic_flood_node": collector_nodes[CHRONIC_BRANCH_INDEX],
        "n_branches": n_branches,
        "inlets_per_branch": inlets_per_branch,
    }
    return graph, node_cell_map, meta


def build_demo_terrain(shape: tuple[int, int], cell_size: float, chronic_cell: tuple[int, int]):
    dem = synthetic_dem(shape, cell_size)
    # guarantee the chronic-flood-point story is visible on the terrain too,
    # not just the pipe network -- real chronic flood points are usually
    # *both* a genuine low point *and* an undersized/silted pipe, not one or
    # the other
    add_depression(dem, chronic_cell[0], chronic_cell[1], depth=0.45, radius=6.0)
    imperviousness = synthetic_imperviousness(shape)
    return dem, imperviousness


def run_one_cycle(
    scenario_id: str = "heavy",
    grid_shape: tuple[int, int] = (80, 80),
    cell_size: float = 6.0,
    horizon_minutes: int = 180,
    output_interval_minutes: int = 15,
    verbose: bool = True,
):
    scenario = get_scenario(scenario_id)
    log = print if verbose else (lambda *a, **k: None)

    log(f"[1/5] Building demo drainage network ({grid_shape[0]}x{grid_shape[1]} @ {cell_size}m cells)...")
    graph, node_cell_map, meta = build_demo_drainage_network(grid_shape, cell_size)
    log(f"      {len(graph.nodes)} nodes, {len(graph.edges)} pipes "
        f"(chronic flood point: {meta['chronic_flood_node']})")

    log(f"[2/5] Loading terrain + fetching '{scenario.label}' rainfall nowcast "
        f"(peak {scenario.peak_mm_per_hr:.0f} mm/hr)...")
    chronic_cell = node_cell_map[meta["chronic_flood_node"]]
    dem, imperviousness = build_demo_terrain(grid_shape, cell_size, chronic_cell)
    from src.model.surface_2d import SurfaceGrid
    surface = SurfaceGrid(dem, cell_size)

    radar = SyntheticRadarSource(grid_shape, peak_mm_per_hr=scenario.peak_mm_per_hr)
    cfg = ForecastConfig(dt_seconds=15, horizon_minutes=horizon_minutes, output_interval_minutes=output_interval_minutes)
    frames = radar.get_nowcast(cfg.horizon_minutes, cfg.output_interval_minutes)
    rainfall_frames = [f.rain_mm_per_hr for f in frames]

    log("[3/5] Delineating inlet catchments (nearest-inlet Voronoi)...")
    label_grid, node_ids = build_catchment_map(node_cell_map, grid_shape, dem=dem)
    areas = catchment_areas_m2(label_grid, node_ids, cell_size**2)
    log(f"      catchment areas range {min(areas.values()):.0f}-{max(areas.values()):.0f} m^2 "
        f"(was fixed at {cell_size**2:.0f} m^2/node before the fix)")

    log("[4/5] Running coupled forecast...")
    t0 = time.time()
    outputs = run_forecast(graph, surface, node_cell_map, rainfall_frames, imperviousness, cfg)
    log(f"      done in {time.time() - t0:.1f}s, {len(outputs)} output frames")

    lead_times = list(range(0, cfg.horizon_minutes, cfg.output_interval_minutes))
    depth_by_lead = dict(zip(lead_times, outputs))

    log("[5/5] Summary:")
    for lead, grid in depth_by_lead.items():
        n_flooded = int((grid > 5.0).sum())
        log(f"      t+{lead:>3}min  max={grid.max():6.1f}cm  cells>5cm={n_flooded}")

    diag = network_summary(graph, areas)
    log(f"\n      Network state at end of run: {diag['n_surcharged']}/{diag['n_nodes']} nodes surcharged, "
        f"worst pipe utilization {diag['worst_pipe_utilization']:.0%}")

    return {
        "depth_by_lead": depth_by_lead,
        "graph": graph,
        "node_cell_map": node_cell_map,
        "catchment_area": areas,
        "meta": meta,
        "scenario": scenario,
        "cell_size": cell_size,
        "grid_shape": grid_shape,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Run one urban flood nowcast cycle.")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="heavy",
                    help="Rainfall scenario preset (see configs/scenarios.py)")
    p.add_argument("--serve", action="store_true", help="Load the result into the API's in-memory store")
    p.add_argument("--rows", type=int, default=80)
    p.add_argument("--cols", type=int, default=80)
    p.add_argument("--cell-size", type=float, default=6.0)
    p.add_argument("--horizon", type=int, default=180)
    p.add_argument("--interval", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_one_cycle(
        scenario_id=args.scenario,
        grid_shape=(args.rows, args.cols),
        cell_size=args.cell_size,
        horizon_minutes=args.horizon,
        output_interval_minutes=args.interval,
    )

    if args.serve:
        from datetime import datetime, timezone
        from src.api.main import load_forecast, load_network_state
        load_forecast(result["depth_by_lead"], issued_at=datetime.now(timezone.utc), cell_size_m=args.cell_size)
        load_network_state(result["graph"], result["node_cell_map"], result["catchment_area"], result["meta"])
        print("\nForecast + network state loaded into API store. Start the server separately with:")
        print("  uvicorn src.api.main:app --reload --port 8000")
        print("(NOTE: this only persists within this process; wire a shared store -- Redis/object")
        print(" storage -- to hand off between the orchestration job and a separately-running API.)")
