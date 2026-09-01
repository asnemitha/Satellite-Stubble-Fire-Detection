"""Flood nowcast API.

Endpoints:
  GET  /forecast/depth        -> depth grid (cm) for a given lead time
  GET  /forecast/point        -> depth-over-time at a lat/lon
  GET  /forecast/summary      -> quick stats across the whole forecast horizon
  GET  /network/diagnostics   -> node/pipe level state
  GET  /scenarios             -> available demo rainfall scenario presets
  POST /simulate/run          -> run a full nowcast cycle synchronously
  POST /route/safe            -> flood-avoiding route between two points

In production the simulation runs as a separate scheduled job
(src/run_cycle.py under Airflow/Temporal) and this service only reads
pre-computed forecast rasters from shared storage. /simulate/run exists
so this repo is demoable with a single process -- it literally imports
and calls run_one_cycle() inline, which is fine for a toy grid but would
never scale to a real city-sized 2D solve inside a request/response cycle.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.routing import (
    assign_flood_costs,
    build_road_graph_from_grid,
    cell_to_latlon,
    latlon_to_cell,
    shortest_safe_path,
)
from src.model.diagnostics import network_summary


app = FastAPI(
    title="Urban Flood Nowcast API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Demo-friendly; scope this down in production.
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory placeholder store. Replace with the real tile/raster store
# (object storage + cache) and a shared network-state store in production.
_FORECAST_STORE: dict[int, np.ndarray] = {}
_FORECAST_ISSUED_AT: datetime | None = None

_GRID_ORIGIN = (
    19.0760,
    72.8777,
)  # (lat, lon) of grid cell (0, 0) -- demo overlay location

_CELL_SIZE_M = 6.0

_NETWORK_STATE: dict = {}
# {"graph": ..., "node_cell_map": ..., "catchment_area": ..., "meta": ...}

_LAST_SCENARIO_ID: str | None = None


class RouteRequest(BaseModel):
    origin: tuple[float, float]
    destination: tuple[float, float]
    depart_at: datetime | None = None
    profile: str = "car"
    max_safe_depth_cm: float | None = None


class RouteResponse(BaseModel):
    path: list[tuple[float, float]]
    avoided_flooded_segments: int
    forecast_lead_minutes_used: int


class SimulateRequest(BaseModel):
    scenario: str = "heavy"
    rows: int = 80
    cols: int = 80
    cell_size_m: float = 6.0
    horizon_minutes: int = 180
    interval_minutes: int = 15


DEFAULT_SAFE_DEPTH_CM = {
    "car": 15.0,
    "pedestrian": 10.0,
    "emergency": 30.0,
}


def _nearest_lead_minutes(
    depart_at: datetime | None,
) -> int:
    if (
        depart_at is None
        or _FORECAST_ISSUED_AT is None
    ):
        return 0

    delta_min = int(
        (
            depart_at - _FORECAST_ISSUED_AT
        ).total_seconds()
        // 60
    )

    available = sorted(
        _FORECAST_STORE.keys()
    )

    if not available:
        raise HTTPException(
            503,
            "No forecast available yet",
        )

    return min(
        available,
        key=lambda m: abs(m - delta_min),
    )


@app.get("/forecast/depth")
def get_depth_grid(
    lead_minutes: int = Query(
        ...,
        ge=0,
        le=180,
    ),
):
    if not _FORECAST_STORE:
        raise HTTPException(
            503,
            "No forecast available yet",
        )

    # Snap to the nearest available lead time instead of hard 404-ing.
    # This handles clients that request lead times that aren't exact
    # multiples of the model's output interval.
    if lead_minutes not in _FORECAST_STORE:
        available = sorted(
            _FORECAST_STORE.keys()
        )

        lead_minutes = min(
            available,
            key=lambda m: abs(m - lead_minutes),
        )

    grid = _FORECAST_STORE[lead_minutes]

    return {
        "lead_minutes": lead_minutes,
        "issued_at": _FORECAST_ISSUED_AT,
        "shape": grid.shape,
        "cell_size_m": _CELL_SIZE_M,
        "origin": _GRID_ORIGIN,
        "depth_cm": grid.tolist(),
    }


@app.get("/forecast/point")
def get_point_forecast(
    lat: float,
    lon: float,
):
    if not _FORECAST_STORE:
        raise HTTPException(
            503,
            "No forecast available yet",
        )

    row, col = _latlon_to_cell(
        lat,
        lon,
    )

    series = []

    for lead, grid in sorted(
        _FORECAST_STORE.items()
    ):
        if (
            0 <= row < grid.shape[0]
            and 0 <= col < grid.shape[1]
        ):
            series.append(
                {
                    "lead_minutes": lead,
                    "depth_cm": float(
                        grid[row, col]
                    ),
                }
            )

    return {
        "lat": lat,
        "lon": lon,
        "series": series,
    }


@app.get("/forecast/summary")
def get_forecast_summary():
    if not _FORECAST_STORE:
        raise HTTPException(
            503,
            "No forecast available yet",
        )

    rows = []

    for lead, grid in sorted(
        _FORECAST_STORE.items()
    ):
        rows.append(
            {
                "lead_minutes": lead,
                "max_depth_cm": float(
                    grid.max()
                ),
                "mean_depth_cm": float(
                    grid.mean()
                ),
                "cells_over_5cm": int(
                    (grid > 5.0).sum()
                ),
                "cells_over_15cm": int(
                    (grid > 15.0).sum()
                ),
                "cells_over_30cm": int(
                    (grid > 30.0).sum()
                ),
            }
        )

    return {
        "issued_at": _FORECAST_ISSUED_AT,
        "scenario": _LAST_SCENARIO_ID,
        "total_cells": int(
            next(
                iter(
                    _FORECAST_STORE.values()
                )
            ).size
        ),
        "series": rows,
    }


@app.get("/network/diagnostics")
def get_network_diagnostics():
    if not _NETWORK_STATE:
        raise HTTPException(
            503,
            "No network state loaded yet -- "
            "run /simulate/run first",
        )

    summary = network_summary(
        _NETWORK_STATE["graph"],
        _NETWORK_STATE["catchment_area"],
    )

    summary["chronic_flood_node"] = (
        _NETWORK_STATE["meta"].get(
            "chronic_flood_node"
        )
    )

    node_cell_map = _NETWORK_STATE[
        "node_cell_map"
    ]

    for node in summary["nodes"]:
        r, c = node_cell_map[
            node["id"]
        ]

        node["lat"], node["lon"] = (
            cell_to_latlon(
                r,
                c,
                _GRID_ORIGIN,
                _CELL_SIZE_M,
            )
        )

    return summary


@app.get("/scenarios")
def get_scenarios():
    from configs.scenarios import SCENARIOS

    return {
        sid: {
            "label": scenario.label,
            "description": scenario.description,
            "peak_mm_per_hr": scenario.peak_mm_per_hr,
        }
        for sid, scenario in SCENARIOS.items()
    }


@app.post("/simulate/run")
def simulate_run(
    req: SimulateRequest,
):
    from src.run_cycle import run_one_cycle

    try:
        result = run_one_cycle(
            scenario_id=req.scenario,
            grid_shape=(
                req.rows,
                req.cols,
            ),
            cell_size=req.cell_size_m,
            horizon_minutes=req.horizon_minutes,
            output_interval_minutes=(
                req.interval_minutes
            ),
            verbose=False,
        )

    except KeyError as exc:
        raise HTTPException(
            400,
            str(exc),
        )

    load_forecast(
        result["depth_by_lead"],
        issued_at=datetime.now(
            timezone.utc
        ),
        cell_size_m=req.cell_size_m,
    )

    load_network_state(
        result["graph"],
        result["node_cell_map"],
        result["catchment_area"],
        result["meta"],
    )

    global _LAST_SCENARIO_ID
    _LAST_SCENARIO_ID = req.scenario

    return {
        "status": "ok",
        "scenario": req.scenario,
        "lead_times_minutes": sorted(
            result["depth_by_lead"].keys()
        ),
        "max_depth_cm": float(
            max(
                grid.max()
                for grid in result[
                    "depth_by_lead"
                ].values()
            )
        ),
        "chronic_flood_node": (
            result["meta"][
                "chronic_flood_node"
            ]
        ),
    }


@app.post(
    "/route/safe",
    response_model=RouteResponse,
)
def route_safe(
    req: RouteRequest,
):
    if not _FORECAST_STORE:
        raise HTTPException(
            503,
            "No forecast available yet",
        )

    lead = _nearest_lead_minutes(
        req.depart_at
    )

    threshold = (
        req.max_safe_depth_cm
        or DEFAULT_SAFE_DEPTH_CM.get(
            req.profile,
            15.0,
        )
    )

    depth_cm = _FORECAST_STORE[lead]

    origin_cell = latlon_to_cell(
        *req.origin,
        _GRID_ORIGIN,
        _CELL_SIZE_M,
    )

    dest_cell = latlon_to_cell(
        *req.destination,
        _GRID_ORIGIN,
        _CELL_SIZE_M,
    )

    # NOTE: This builds/costs a fresh grid-graph per request for clarity.
    # In production, build the real road graph once at startup and only
    # re-run assign_flood_costs() per request/forecast update.
    road_graph = build_road_graph_from_grid(
        depth_cm.shape
    )

    road_graph = assign_flood_costs(
        road_graph,
        depth_cm,
        threshold,
    )

    try:
        path_cells, flooded_count = (
            shortest_safe_path(
                road_graph,
                origin_cell,
                dest_cell,
            )
        )

    except Exception as exc:
        raise HTTPException(
            400,
            f"No route found: {exc}",
        )

    path_latlon = [
        cell_to_latlon(
            r,
            c,
            _GRID_ORIGIN,
            _CELL_SIZE_M,
        )
        for r, c in path_cells
    ]

    return RouteResponse(
        path=path_latlon,
        avoided_flooded_segments=flooded_count,
        forecast_lead_minutes_used=lead,
    )


def _latlon_to_cell(
    lat: float,
    lon: float,
) -> tuple[int, int]:
    return latlon_to_cell(
        lat,
        lon,
        _GRID_ORIGIN,
        _CELL_SIZE_M,
    )


def load_forecast(
    depth_grids_by_lead: dict[int, np.ndarray],
    issued_at: datetime | None = None,
    cell_size_m: float | None = None,
) -> None:
    """Load forecast results into the in-memory store.

    Called by the orchestration job or /simulate/run
    after each coupled model run.
    """
    global _FORECAST_STORE
    global _FORECAST_ISSUED_AT
    global _CELL_SIZE_M

    _FORECAST_STORE = depth_grids_by_lead

    _FORECAST_ISSUED_AT = (
        issued_at
        or datetime.now(timezone.utc)
    )

    if cell_size_m is not None:
        _CELL_SIZE_M = cell_size_m


def load_network_state(
    graph,
    node_cell_map: dict,
    catchment_area: dict,
    meta: dict,
) -> None:
    """Load network state for /network/diagnostics."""
    global _NETWORK_STATE

    _NETWORK_STATE = {
        "graph": graph,
        "node_cell_map": node_cell_map,
        "catchment_area": catchment_area,
        "meta": meta,
    }