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


def load_dem_geotiff(path: str) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Load a DEM from a GeoTIFF. Returns (elevation_grid, cell_size_m, origin_latlon)."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to load real DEM files: pip install rasterio")
    import rasterio as rio
    with rio.open(path) as src:
        dem = src.read(1).astype(float)
        cell_size = abs(src.transform.a)
        origin = (src.transform.f, src.transform.c)  # (lat, lon) of top-left
    return dem, cell_size, origin


def load_imperviousness(path: str) -> np.ndarray:
    """Load a 0-1 imperviousness raster, same grid as the DEM."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to load real land-use rasters: pip install rasterio")
    import rasterio as rio
    with rio.open(path) as src:
        return np.clip(src.read(1).astype(float), 0, 1)


def synthetic_dem(shape: tuple[int, int], cell_size: float = 5.0,
                   base_elevation: float = 10.0, seed: int = 0) -> np.ndarray:
    """A gently sloped surface with a few local depressions -- the kind of
    micro-topography that causes hyper-local ponding even when the broader
    area drains fine."""
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    # overall gentle slope toward one edge (mimics a street sloping to an outfall)
    slope_component = (xx / cols) * 1.5

    rng = np.random.default_rng(seed)
    dem = base_elevation - slope_component
    n_depressions = max(1, (rows * cols) // 400)
    for _ in range(n_depressions):
        cy, cx = rng.integers(0, rows), rng.integers(0, cols)
        depth = rng.uniform(0.15, 0.6)
        radius = rng.uniform(2, 6)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        dem -= depth * np.exp(-dist2 / (2 * radius ** 2))
    return dem


def add_depression(dem: np.ndarray, row: int, col: int, depth: float, radius: float) -> np.ndarray:
    """Carve an extra local low point into a DEM in-place (returns it too, for
    chaining). Used by the demo to guarantee a visible, narratable "chronic
    flood point" at a specific junction, on top of the generic random
    micro-topography from `synthetic_dem` -- real cities have these (a
    specific junction that floods every monsoon due to genuine low elevation
    plus drainage bottleneck); this reproduces that pattern for the demo
    rather than leaving it to chance."""
    rows, cols = dem.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist2 = (xx - col) ** 2 + (yy - row) ** 2
    dem -= depth * np.exp(-dist2 / (2 * radius**2))
    return dem


def synthetic_imperviousness(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """0-1 grid mimicking mixed road/rooftop/open-ground imperviousness."""
    rng = np.random.default_rng(seed + 1)
    base = rng.uniform(0.5, 0.95, size=shape)
    return np.clip(base, 0, 1)
