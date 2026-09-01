"""Catchment delineation: which surface cells drain to which inlet node.

This module exists to fix the bug that made the original skeleton predict
zero flooding under any rainfall: `coupled_forecast.py` was assigning each
inlet node runoff from exactly one grid cell (its own), instead of the
whole area of terrain that actually drains toward it. A real storm inlet
serves hundreds to thousands of square meters, not 25 m^2 (one 5m cell) --
so pipes were receiving a tiny fraction of the real inflow and never
surcharged.

Proper catchment delineation needs flow-direction analysis (D8/D-infinity)
over the DEM, which needs a real hydrologically-conditioned DEM to be
meaningful. Absent that, this module uses the standard fallback: assign
each cell to its nearest inlet by straight-line (or optionally
slope-weighted) distance -- a Voronoi/Thiessen-polygon catchment. This is
a real, documented simplification (used routinely when flow-direction
data isn't available), not a placeholder -- but it should be swapped for
D8 flow accumulation once a hydrologically-conditioned DEM is available
(see docs/DESIGN.md sec 7).
"""
from __future__ import annotations

import numpy as np


def build_catchment_map(
    node_cell_map: dict[str, tuple[int, int]],
    shape: tuple[int, int],
    dem: np.ndarray | None = None,
    elevation_weight: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """Assign every surface cell to its nearest inlet node.

    Args:
        node_cell_map: node_id -> (row, col) of that inlet on the surface grid.
        shape: (rows, cols) of the surface grid.
        dem: optional elevation grid; if provided with elevation_weight > 0,
            cells are biased toward inlets they can plausibly drain downhill
            to (cheap stand-in for real flow-direction routing).
        elevation_weight: 0 = pure nearest-neighbor (Voronoi) catchments.
            >0 additionally penalizes assigning a cell to an inlet that sits
            *higher* than it (water doesn't flow uphill to a drain).

    Returns:
        (label_grid, node_ids): label_grid[r, c] is an index into node_ids
        giving which node's catchment cell (r, c) belongs to. node_ids with
        no cells assigned (shouldn't normally happen) simply get an empty
        catchment.
    """
    rows, cols = shape
    node_ids = list(node_cell_map.keys())
    if not node_ids:
        return np.full(shape, -1, dtype=int), node_ids

    node_rc = np.array([node_cell_map[nid] for nid in node_ids], dtype=float)  # (n, 2)
    rr, cc = np.mgrid[0:rows, 0:cols]
    # squared distance from every cell to every node -> (rows, cols, n)
    d2 = (rr[..., None] - node_rc[:, 0]) ** 2 + (cc[..., None] - node_rc[:, 1]) ** 2

    if dem is not None and elevation_weight > 0:
        node_elev = np.array([dem[r, c] for (r, c) in node_rc.astype(int)])
        cell_elev = dem[..., None]
        uphill_penalty = np.clip(node_elev - cell_elev, 0, None) * elevation_weight
        cost = d2 + uphill_penalty**2
    else:
        cost = d2

    label = np.argmin(cost, axis=-1)
    return label, node_ids


def aggregate_runoff_by_catchment(
    runoff_grid: np.ndarray,
    label_grid: np.ndarray,
    node_ids: list[str],
    cell_area: float,
) -> dict[str, float]:
    """Sum runoff (m^3/s) over every cell in each node's catchment.

    This is the piece that actually fixes the bug: instead of reading a
    single cell's runoff rate, we now integrate over the node's whole
    contributing area.
    """
    totals: dict[str, float] = {}
    flat_labels = label_grid.ravel()
    flat_runoff = runoff_grid.ravel()
    for i, nid in enumerate(node_ids):
        mask = flat_labels == i
        totals[nid] = float(flat_runoff[mask].sum()) * cell_area
    return totals


def catchment_areas_m2(label_grid: np.ndarray, node_ids: list[str], cell_area: float) -> dict[str, float]:
    """Contributing area (m^2) per node -- useful for diagnostics/UI and for
    sizing node_storage_area realistically instead of using one constant for
    every inlet regardless of how much street it actually serves."""
    flat = label_grid.ravel()
    return {nid: float((flat == i).sum()) * cell_area for i, nid in enumerate(node_ids)}
