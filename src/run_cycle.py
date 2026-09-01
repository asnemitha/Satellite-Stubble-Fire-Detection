"""Orchestration entry point: runs one full nowcast cycle.

Pipeline:

    rainfall nowcast
        -> runoff
        -> catchment aggregation
        -> 1D drainage routing
        -> surcharge
        -> 2D surface inundation
        -> 1D re-entry

Steps:
    1. Build/load drainage network
    2. Load terrain + imperviousness + rainfall nowcast
    3. Delineate catchments
    4. Run coupled 1D-2D forecast
    5. Publish results into the API forecast store

In production this can be triggered by a scheduler such as Airflow or
Temporal whenever new radar data arrives.

For this demo it runs once against synthetic data so that the complete
pipeline can be exercised without external data sources.

Usage:
    python -m src.run_cycle
    python -m src.run_cycle --scenario extreme
    python -m src.run_cycle --scenario light --serve
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from configs.scenarios import SCENARIOS, get_scenario

from src.ingestion.dem_loader import (
    add_depression,
    synthetic_dem,
    synthetic_imperviousness,
)
from src.ingestion.radar_nowcast import SyntheticRadarSource

from src.model.catchments import (
    build_catchment_map,
    catchment_areas_m2,
)
from src.model.coupled_forecast import (
    ForecastConfig,
    run_forecast,
)
from src.model.diagnostics import network_summary
from src.model.drainage_graph import (
    DrainageGraph,
    Edge,
    Node,
)
from src.model.surface_2d import SurfaceGrid


# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------

# One branch is deliberately undersized and heavily degraded so that the
# demo produces a persistent/chronic flood point.
CHRONIC_BRANCH_INDEX = 1


# ---------------------------------------------------------------------------
# Synthetic drainage network
# ---------------------------------------------------------------------------

def build_demo_drainage_network(
    shape: tuple[int, int],
    cell_size: float,
) -> tuple[
    DrainageGraph,
    dict[str, tuple[int, int]],
    dict,
]:
    """Build a branching synthetic storm-drain network.

    Several streets of inlets feed branch collectors. The collectors then
    feed a trunk main and finally a single outfall.

    Returns:
        graph:
            Drainage network graph.

        node_cell_map:
            Mapping from node ID to its corresponding 2D grid cell.

        meta:
            Demo annotations used by the API/dashboard.
    """

    rows, cols = shape

    if rows < 10 or cols < 10:
        raise ValueError(
            "Demo grid must be at least 10x10 cells."
        )

    if cell_size <= 0:
        raise ValueError(
            "cell_size must be greater than zero."
        )

    graph = DrainageGraph()

    node_cell_map: dict[str, tuple[int, int]] = {}

    n_branches = 4
    inlets_per_branch = 5

    branch_cols = [
        int(cols * (i + 1) / (n_branches + 1))
        for i in range(n_branches)
    ]

    collector_row = int(rows * 0.68)

    outfall_r = collector_row
    outfall_c = cols - 3

    # ------------------------------------------------------------------
    # Outfall
    # ------------------------------------------------------------------

    graph.add_node(
        Node(
            id="outfall",
            x=outfall_c * cell_size,
            y=outfall_r * cell_size,
            ground_elevation=6.0,
            invert_elevation=4.0,
            is_outfall=True,
        )
    )

    node_cell_map["outfall"] = (
        outfall_r,
        outfall_c,
    )

    # ------------------------------------------------------------------
    # Branches
    # ------------------------------------------------------------------

    collector_nodes: list[str] = []

    for branch_index, col in enumerate(branch_cols):

        previous_node_id: str | None = None

        previous_row: int | None = None

        for inlet_index in range(inlets_per_branch):

            row = int(
                rows
                * (
                    0.08
                    + 0.5
                    * inlet_index
                    / (inlets_per_branch - 1)
                )
            )

            node_id = (
                f"b{branch_index}_inlet{inlet_index}"
            )

            elevation = (
                9.5
                - 0.06 * inlet_index
            )

            graph.add_node(
                Node(
                    id=node_id,
                    x=col * cell_size,
                    y=row * cell_size,
                    ground_elevation=elevation,
                    invert_elevation=elevation - 1.1,
                )
            )

            node_cell_map[node_id] = (
                row,
                col,
            )

            # Connect consecutive inlet nodes.
            if previous_node_id is not None:
                assert previous_row is not None

                graph.add_edge(
                    Edge(
                        id=(
                            f"pipe_"
                            f"{previous_node_id}_"
                            f"{node_id}"
                        ),
                        from_node=previous_node_id,
                        to_node=node_id,
                        length=(
                            cell_size
                            * abs(row - previous_row)
                        ),
                        diameter=0.45,
                        slope=0.006,
                        condition_factor=0.85,
                    )
                )

            previous_node_id = node_id
            previous_row = row

        # --------------------------------------------------------------
        # Branch collector
        # --------------------------------------------------------------

        collector_id = (
            f"collector_{branch_index}"
        )

        collector_elevation = 7.6

        graph.add_node(
            Node(
                id=collector_id,
                x=col * cell_size,
                y=collector_row * cell_size,
                ground_elevation=collector_elevation,
                invert_elevation=collector_elevation - 1.3,
            )
        )

        node_cell_map[collector_id] = (
            collector_row,
            col,
        )

        is_chronic = (
            branch_index == CHRONIC_BRANCH_INDEX
        )

        # Deliberately undersize/degrade the chronic branch.
        graph.add_edge(
            Edge(
                id=(
                    f"pipe_"
                    f"{previous_node_id}_"
                    f"{collector_id}"
                ),
                from_node=previous_node_id,
                to_node=collector_id,
                length=(
                    cell_size
                    * abs(
                        collector_row
                        - previous_row
                    )
                ),
                diameter=(
                    0.25
                    if is_chronic
                    else 0.60
                ),
                slope=0.005,
                condition_factor=(
                    0.35
                    if is_chronic
                    else 0.85
                ),
            )
        )

        collector_nodes.append(
            collector_id
        )

    # ------------------------------------------------------------------
    # Collector-to-collector trunk
    # ------------------------------------------------------------------

    previous_collector: str | None = None
    previous_col: int | None = None

    for collector_id, branch_col in zip(
        collector_nodes,
        branch_cols,
    ):

        if previous_collector is not None:
            assert previous_col is not None

            graph.add_edge(
                Edge(
                    id=(
                        f"trunk_"
                        f"{previous_collector}_"
                        f"{collector_id}"
                    ),
                    from_node=previous_collector,
                    to_node=collector_id,
                    length=(
                        cell_size
                        * abs(branch_col - previous_col)
                    ),
                    diameter=0.60,
                    slope=0.003,
                    condition_factor=0.90,
                )
            )

        previous_collector = collector_id
        previous_col = branch_col

    # ------------------------------------------------------------------
    # Final trunk -> outfall
    # ------------------------------------------------------------------

    graph.add_edge(
        Edge(
            id="trunk_to_outfall",
            from_node=collector_nodes[-1],
            to_node="outfall",
            length=(
                cell_size
                * abs(
                    outfall_c
                    - branch_cols[-1]
                )
            ),
            diameter=0.75,
            slope=0.004,
            condition_factor=0.90,
        )
    )

    meta = {
        "chronic_flood_node": (
            collector_nodes[
                CHRONIC_BRANCH_INDEX
            ]
        ),
        "n_branches": n_branches,
        "inlets_per_branch": inlets_per_branch,
    }

    return (
        graph,
        node_cell_map,
        meta,
    )


# ---------------------------------------------------------------------------
# Synthetic terrain
# ---------------------------------------------------------------------------

def build_demo_terrain(
    shape: tuple[int, int],
    cell_size: float,
    chronic_cell: tuple[int, int],
):
    """Build synthetic DEM and imperviousness layers.

    A local depression is inserted at the chronic flood point so the
    demonstration represents both hydraulic and terrain controls.
    """

    dem = synthetic_dem(
        shape,
        cell_size,
    )

    add_depression(
        dem,
        chronic_cell[0],
        chronic_cell[1],
        depth=0.45,
        radius=6.0,
    )

    imperviousness = synthetic_imperviousness(
        shape
    )

    return (
        dem,
        imperviousness,
    )


# ---------------------------------------------------------------------------
# One forecast cycle
# ---------------------------------------------------------------------------

def run_one_cycle(
    scenario_id: str = "heavy",
    grid_shape: tuple[int, int] = (80, 80),
    cell_size: float = 6.0,
    horizon_minutes: int = 180,
    output_interval_minutes: int = 15,
    verbose: bool = True,
):
    """Run one complete synthetic flood-nowcast cycle."""

    # ------------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------------

    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available scenarios: "
            f"{', '.join(sorted(SCENARIOS))}"
        )

    rows, cols = grid_shape

    if rows <= 0 or cols <= 0:
        raise ValueError(
            "grid_shape must contain positive dimensions."
        )

    if cell_size <= 0:
        raise ValueError(
            "cell_size must be greater than zero."
        )

    if horizon_minutes <= 0:
        raise ValueError(
            "horizon_minutes must be greater than zero."
        )

    if output_interval_minutes <= 0:
        raise ValueError(
            "output_interval_minutes must be greater than zero."
        )

    scenario = get_scenario(
        scenario_id
    )

    log = (
        print
        if verbose
        else lambda *args, **kwargs: None
    )

    # ------------------------------------------------------------------
    # 1. Build drainage network
    # ------------------------------------------------------------------

    log(
        "[1/5] Building demo drainage network "
        f"({rows}x{cols} @ {cell_size}m cells)..."
    )

    (
        graph,
        node_cell_map,
        meta,
    ) = build_demo_drainage_network(
        grid_shape,
        cell_size,
    )

    log(
        f"      {len(graph.nodes)} nodes, "
        f"{len(graph.edges)} pipes "
        f"(chronic flood point: "
        f"{meta['chronic_flood_node']})"
    )

    # ------------------------------------------------------------------
    # 2. Terrain + rainfall
    # ------------------------------------------------------------------

    log(
        "[2/5] Loading terrain + fetching "
        f"'{scenario.label}' rainfall nowcast "
        f"(peak "
        f"{scenario.peak_mm_per_hr:.0f} mm/hr)..."
    )

    chronic_node = meta[
        "chronic_flood_node"
    ]

    chronic_cell = node_cell_map[
        chronic_node
    ]

    dem, imperviousness = (
        build_demo_terrain(
            grid_shape,
            cell_size,
            chronic_cell,
        )
    )

    surface = SurfaceGrid(
        dem,
        cell_size,
    )

    radar = SyntheticRadarSource(
        grid_shape,
        peak_mm_per_hr=scenario.peak_mm_per_hr,
    )

    # Keep the routing timestep small enough for the explicit solver.
    cfg = ForecastConfig(
        dt_seconds=15.0,
        horizon_minutes=horizon_minutes,
        output_interval_minutes=output_interval_minutes,
    )

    frames = radar.get_nowcast(
        cfg.horizon_minutes,
        cfg.output_interval_minutes,
    )

    rainfall_frames = [
        frame.rain_mm_per_hr
        for frame in frames
    ]

    if not rainfall_frames:
        raise RuntimeError(
            "Radar nowcast returned no rainfall frames."
        )

    # ------------------------------------------------------------------
    # 3. Catchment delineation
    # ------------------------------------------------------------------

    log(
        "[3/5] Delineating inlet catchments "
        "(nearest-inlet Voronoi)..."
    )

    (
        label_grid,
        node_ids,
    ) = build_catchment_map(
        node_cell_map,
        grid_shape,
        dem=dem,
    )

    areas = catchment_areas_m2(
        label_grid,
        node_ids,
        cell_size**2,
    )

    if not areas:
        raise RuntimeError(
            "Catchment delineation produced no catchment areas."
        )

    min_area = min(
        areas.values()
    )

    max_area = max(
        areas.values()
    )

    log(
        f"      catchment areas range "
        f"{min_area:.0f}-{max_area:.0f} m² "
        f"(previously fixed at "
        f"{cell_size**2:.0f} m²/node)"
    )

    # ------------------------------------------------------------------
    # 4. Coupled 1D-2D forecast
    # ------------------------------------------------------------------

    log(
        "[4/5] Running coupled forecast..."
    )

    t0 = time.time()

    outputs = run_forecast(
        graph=graph,
        surface=surface,
        node_cell_map=node_cell_map,
        rainfall_frames=rainfall_frames,
        imperviousness=imperviousness,
        cfg=cfg,
    )

    elapsed = time.time() - t0

    log(
        f"      done in {elapsed:.1f}s, "
        f"{len(outputs)} output frames"
    )

    # ------------------------------------------------------------------
    # Lead-time mapping
    # ------------------------------------------------------------------
    #
    # run_forecast() returns one state after every output interval.
    #
    # Therefore:
    #
    # output[0] -> t+interval
    # output[1] -> t+2*interval
    # ...
    # output[-1] -> t+horizon
    #
    # The old implementation incorrectly labelled output[0] as t+0.
    # ------------------------------------------------------------------

    lead_times = [
        (index + 1)
        * cfg.output_interval_minutes
        for index in range(len(outputs))
    ]

    depth_by_lead = dict(
        zip(
            lead_times,
            outputs,
        )
    )

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------

    log("[5/5] Summary:")

    for lead, grid in depth_by_lead.items():

        n_flooded = int(
            (grid > 5.0).sum()
        )

        log(
            f"      t+{lead:>3}min  "
            f"max={grid.max():6.1f}cm  "
            f"cells>5cm={n_flooded}"
        )

    diag = network_summary(
        graph,
        areas,
    )

    log(
        "\n      Network state at end of run: "
        f"{diag['n_surcharged']}/"
        f"{diag['n_nodes']} nodes surcharged, "
        f"worst pipe utilization "
        f"{diag['worst_pipe_utilization']:.0%}"
    )

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run one urban flood nowcast cycle."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="heavy",
        help=(
            "Rainfall scenario preset "
            "(see configs/scenarios.py)"
        ),
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Load the result into the API's "
            "in-memory store."
        ),
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--cell-size",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=15,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    args = _parse_args()

    result = run_one_cycle(
        scenario_id=args.scenario,
        grid_shape=(
            args.rows,
            args.cols,
        ),
        cell_size=args.cell_size,
        horizon_minutes=args.horizon,
        output_interval_minutes=args.interval,
    )

    if args.serve:

        from src.api.main import (
            load_forecast,
            load_network_state,
        )

        load_forecast(
            result["depth_by_lead"],
            issued_at=datetime.now(
                timezone.utc
            ),
            cell_size_m=args.cell_size,
        )

        load_network_state(
            result["graph"],
            result["node_cell_map"],
            result["catchment_area"],
            result["meta"],
        )

        print(
            "\nForecast + network state loaded "
            "into API store."
        )

        print(
            "Start the server separately with:"
        )

        print(
            "  uvicorn src.api.main:app "
            "--reload --port 8000"
        )

        print(
            "\nNOTE: this only persists within this "
            "process; use a shared store such as "
            "Redis/object storage to hand off results "
            "between the orchestration job and a "
            "separately-running API."
        )