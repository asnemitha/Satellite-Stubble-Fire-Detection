"""Radar rainfall ingestion and short-term nowcasting.

Production target: pull reflectivity/QPE from IMD Doppler Weather Radar (DWR)
feeds, convert to rainfall intensity, and extrapolate forward using an optical
flow method (e.g. pySTEPS) blended with NWP guidance beyond ~60 minutes.

This module defines the interface and a synthetic fallback so the rest of the
pipeline can be developed and tested without a live radar feed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RadarFrame:
    timestamp: float          # unix seconds
    rain_mm_per_hr: np.ndarray  # grid, same shape/extent as the model DEM


class RadarSource:
    """Interface a real radar/nowcast provider should implement."""

    def get_latest_frame(self) -> RadarFrame:
        raise NotImplementedError

    def get_nowcast(self, horizon_minutes: int, interval_minutes: int) -> list[RadarFrame]:
        """Return extrapolated rainfall frames from t+0 to t+horizon_minutes."""
        raise NotImplementedError


class SyntheticRadarSource(RadarSource):
    """Deterministic synthetic rainfall generator for dev/testing.

    Produces a moving convective cell so the coupled model has non-trivial,
    time-varying rainfall to route without needing a live feed.
    """

    def __init__(self, grid_shape: tuple[int, int], peak_mm_per_hr: float = 80.0, seed: int = 0):
        self.grid_shape = grid_shape
        self.peak = peak_mm_per_hr
        self.rng = np.random.default_rng(seed)

    def _cell_grid(self, t_min: float) -> np.ndarray:
        rows, cols = self.grid_shape
        yy, xx = np.mgrid[0:rows, 0:cols]
        # storm cell drifts across the grid over the forecast horizon
        cx = cols * (0.1 + 0.6 * (t_min / 180.0))
        cy = rows * 0.5
        radius = max(rows, cols) * 0.25
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        intensity = self.peak * np.exp(-dist2 / (2 * radius ** 2))
        noise = self.rng.normal(0, 2.0, size=self.grid_shape)
        return np.clip(intensity + noise, 0, None)

    def get_latest_frame(self) -> RadarFrame:
        return RadarFrame(timestamp=0.0, rain_mm_per_hr=self._cell_grid(0))

    def get_nowcast(self, horizon_minutes: int, interval_minutes: int) -> list[RadarFrame]:
        frames = []
        for t in range(0, horizon_minutes + 1, interval_minutes):
            frames.append(RadarFrame(timestamp=t * 60.0, rain_mm_per_hr=self._cell_grid(t)))
        return frames
