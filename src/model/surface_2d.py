"""2D surface inundation model.

Local-inertial (diffusive wave) approximation of shallow water flow over a DEM
grid — the LISFLOOD-FP style method used because it is stable at real-time
timesteps and vectorizes cleanly (numpy here; swap for CuPy/CUDA for GPU scale).
"""
from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.81


class SurfaceGrid:
    def __init__(self, dem: np.ndarray, cell_size: float, manning_n: np.ndarray | float = 0.03):
        """
        dem: 2D array of ground elevation (m), shape (rows, cols)
        cell_size: grid resolution (m)
        manning_n: scalar or per-cell roughness array
        """
        self.dem = dem
        self.cell_size = cell_size
        self.manning_n = manning_n if isinstance(manning_n, np.ndarray) else np.full_like(dem, manning_n)
        self.depth = np.zeros_like(dem)          # water depth (m)
        self.qx = np.zeros_like(dem)              # flux in x (m2/s)
        self.qy = np.zeros_like(dem)              # flux in y (m2/s)

    def inject(self, row: int, col: int, volume_rate: float, dt: float) -> None:
        """Add a source term (m3/s) at a cell, e.g. a surcharging manhole or rainfall."""
        self.depth[row, col] += (volume_rate * dt) / (self.cell_size ** 2)

    def step(self, dt: float, cfl: float = 0.4, max_substeps: int = 500) -> None:
        """Advance the surface by dt seconds, internally subdividing into
        CFL-stable sub-steps.

        The explicit local-inertial scheme below is only stable when dt is
        small relative to how fast a shallow-water wave can cross a cell
        (dt <~ cfl * cell_size / sqrt(g * depth)). A fixed, hand-picked
        dt_seconds that's fine for shallow ponding blows up (NaN via
        overflow) the moment a real surcharge event pushes local depth up --
        exactly the "flooding actually happens now" case the rest of this
        fix was aiming for. So: pick a stable sub-step from the current
        worst-case depth each call, rather than trusting the caller's dt
        blindly. `max_substeps` is a hard cap (not a floor) so a pathological
        depth spike degrades to a bounded amount of extra compute instead of
        stalling the run; if it's hit, the excess time is applied at the
        smallest stable step anyway rather than silently going unstable.
        """
        remaining = dt
        sub_dt = dt  # safe fallback; overwritten immediately inside the loop
        for _ in range(max_substeps):
            if remaining <= 1e-9:
                return
            hmax = float(np.max(self.depth))
            if hmax > 1e-6:
                stable_dt = cfl * self.cell_size / math.sqrt(GRAVITY * hmax)
            else:
                stable_dt = remaining
            sub_dt = min(stable_dt, remaining)
            self._step_once(sub_dt)
            remaining -= sub_dt
        if remaining > 1e-9:
            # exhausted the substep budget with time left -- take it at the
            # last-known-stable size rather than dumping it in one unstable step
            self._step_once(min(sub_dt, remaining))

    def _step_once(self, dt: float) -> None:
        wse = self.dem + self.depth  # water surface elevation

        # water surface slope, x and y directions
        dwse_dx = np.zeros_like(wse)
        dwse_dx[:, :-1] = (wse[:, 1:] - wse[:, :-1]) / self.cell_size
        dwse_dy = np.zeros_like(wse)
        dwse_dy[:-1, :] = (wse[1:, :] - wse[:-1, :]) / self.cell_size

        h_flow_x = np.maximum(self.depth[:, :-1], self.depth[:, 1:]) if wse.shape[1] > 1 else self.depth
        h_flow_y = np.maximum(self.depth[:-1, :], self.depth[1:, :]) if wse.shape[0] > 1 else self.depth

        n = self.manning_n
        # local inertial flux update (simplified explicit form)
        qx_new = np.zeros_like(self.qx)
        qy_new = np.zeros_like(self.qy)
        eps = 1e-6

        hx = np.zeros_like(wse); hx[:, :-1] = h_flow_x
        qx_new[:, :-1] = (self.qx[:, :-1] - GRAVITY * hx[:, :-1] * dt * dwse_dx[:, :-1]) / (
            1 + GRAVITY * dt * n[:, :-1] ** 2 * np.abs(self.qx[:, :-1]) / (hx[:, :-1] ** (7 / 3) + eps)
        )
        hy = np.zeros_like(wse); hy[:-1, :] = h_flow_y
        qy_new[:-1, :] = (self.qy[:-1, :] - GRAVITY * hy[:-1, :] * dt * dwse_dy[:-1, :]) / (
            1 + GRAVITY * dt * n[:-1, :] ** 2 * np.abs(self.qy[:-1, :]) / (hy[:-1, :] ** (7 / 3) + eps)
        )

        self.qx, self.qy = qx_new, qy_new

        # continuity: update depth from flux divergence
        div = np.zeros_like(self.depth)
        div[:, 1:] += self.qx[:, :-1]
        div[:, :-1] -= self.qx[:, :-1]
        div[1:, :] += self.qy[:-1, :]
        div[:-1, :] -= self.qy[:-1, :]

        self.depth = np.maximum(0.0, self.depth + (div * dt) / self.cell_size)
        self._apply_open_boundary(dt)

    def _apply_open_boundary(self, dt: float, weir_coeff: float = 1.2) -> None:
        """Let water leave the edge of the modeled tile instead of piling up
        against an implicit wall.

        The finite-volume update above has no flux term across the outer
        boundary, so a closed rectangular study area would behave like a
        bathtub -- every drop of rain that isn't captured by a pipe stays on
        the grid forever, and depth grows without bound over a multi-hour
        run regardless of how well the drainage network performs. Real study
        tiles sit inside a larger city that keeps draining past the edge, so
        each boundary cell gets a broad-crested-weir-style outflow term
        proportional to depth^1.5 -- an approximation of "water keeps moving
        toward wherever this tile's downstream neighbor is", not a precise
        transmissive boundary condition.
        """
        if self.depth.shape[0] < 2 or self.depth.shape[1] < 2:
            return
        q = weir_coeff * np.power(np.maximum(self.depth, 0.0), 1.5) * dt / self.cell_size
        self.depth[0, :] = np.maximum(0.0, self.depth[0, :] - q[0, :])
        self.depth[-1, :] = np.maximum(0.0, self.depth[-1, :] - q[-1, :])
        self.depth[:, 0] = np.maximum(0.0, self.depth[:, 0] - q[:, 0])
        self.depth[:, -1] = np.maximum(0.0, self.depth[:, -1] - q[:, -1])

    def depth_cm(self) -> np.ndarray:
        return self.depth * 100.0
