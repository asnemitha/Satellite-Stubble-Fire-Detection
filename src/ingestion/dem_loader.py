"""DEM and land-use (imperviousness) ingestion.

Production target: load a bare-earth DEM (buildings removed) from a GeoTIFF,
and an imperviousness raster derived from land-use/land-cover classification,
both clipped/resampled to a common grid.

Provides a synthetic fallback (a bowl-shaped terrain with a couple of local
low points, typical of the micro-topography that causes hyper-local pooling)
for development without real rasters.
"""

from __future__ import annotations

import numpy as np

try:
    import rasterio  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def load_dem_geotiff(
    path: str,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Load a DEM from a GeoTIFF.

    Returns:
        (elevation_grid, cell_size_m, origin_latlon)
    """
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required to load real DEM files: "
            "pip install rasterio"
        )

    import rasterio as rio

    with rio.open(path) as src:
        dem = src.read(1).astype(float)
        cell_size = abs(src.transform.a)
        origin = (
            src.transform.f,
            src.transform.c,
        )

    return dem, cell_size, origin


def load_imperviousness(
    path: str,
) -> np.ndarray:
    """Load a 0-1 imperviousness raster matching the DEM grid."""
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required to load real land-use rasters: "
            "pip install rasterio"
        )

    import rasterio as rio

    with rio.open(path) as src:
        return np.clip(
            src.read(1).astype(float),
            0,
            1,
        )


def synthetic_dem(
    shape: tuple[int, int],
    cell_size: float = 5.0,
    base_elevation: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Create a synthetic DEM with local depressions.

    The terrain has a gentle overall slope and several local depressions
    representing micro-topography that can cause hyper-local ponding.
    """
    rows, cols = shape

    yy, xx = np.mgrid[
        0:rows,
        0:cols,
    ]

    # Overall gentle slope toward one edge.
    slope_component = (
        xx / cols
    ) * 1.5

    rng = np.random.default_rng(seed)

    dem = (
        base_elevation
        - slope_component
    )

    n_depressions = max(
        1,
        (rows * cols) // 400,
    )

    for _ in range(n_depressions):
        cy, cx = rng.integers(
            0,
            rows,
        ), rng.integers(
            0,
            cols,
        )

        depth = rng.uniform(
            0.15,
            0.6,
        )

        radius = rng.uniform(
            2,
            6,
        )

        dist2 = (
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        dem -= (
            depth
            * np.exp(
                -dist2
                / (2 * radius**2)
            )
        )

    return dem


def add_depression(
    dem: np.ndarray,
    row: int,
    col: int,
    depth: float,
    radius: float,
) -> np.ndarray:
    """Add an extra local low point to a DEM.

    The depression is carved in-place and the modified DEM is returned
    for convenient chaining.

    Used by the demo to guarantee a visible chronic flood point at a
    specific junction, on top of the generic random micro-topography
    from ``synthetic_dem``.
    """
    rows, cols = dem.shape

    yy, xx = np.mgrid[
        0:rows,
        0:cols,
    ]

    dist2 = (
        (xx - col) ** 2
        + (yy - row) ** 2
    )

    dem -= (
        depth
        * np.exp(
            -dist2
            / (2 * radius**2)
        )
    )

    return dem


def synthetic_imperviousness(
    shape: tuple[int, int],
    seed: int = 0,
) -> np.ndarray:
    """Create a 0-1 grid mimicking mixed surface imperviousness."""
    rng = np.random.default_rng(
        seed + 1
    )

    base = rng.uniform(
        0.5,
        0.95,
        size=shape,
    )

    return np.clip(
        base,
        0,
        1,
    )