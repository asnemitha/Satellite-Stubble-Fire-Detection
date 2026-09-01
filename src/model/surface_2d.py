"""2D surface inundation model.

A lightweight local-inertial / diffusive-wave approximation for shallow-water
flow over a DEM grid.

The solver is designed to couple with the 1D drainage model:

    1D surcharge -> surface.inject(...)
    2D routing   -> surface.step(...)
    2D ponding   -> coupled_forecast re-entry

Units:
    DEM              : m
    depth            : m
    qx / qy          : m²/s  (unit-width discharge)
    volume           : m³
    rainfall/source  : m³/s
"""

from __future__ import annotations

import math

import numpy as np


GRAVITY = 9.81

DEFAULT_MANNING_N = 0.03
MIN_DEPTH = 1.0e-6
MIN_FLOW_DEPTH = 1.0e-5


class SurfaceGrid:
    """Finite-volume 2D surface-water grid."""

    def __init__(
        self,
        dem: np.ndarray,
        cell_size: float,
        manning_n: np.ndarray | float = DEFAULT_MANNING_N,
    ) -> None:
        """Create a 2D surface grid.

        Args:
            dem:
                2D ground elevation array in metres.

            cell_size:
                Square cell size in metres.

            manning_n:
                Manning roughness. Either one scalar value or a 2D array
                matching the DEM shape.
        """

        dem = np.asarray(
            dem,
            dtype=float,
        )

        if dem.ndim != 2:
            raise ValueError(
                "dem must be a 2D array."
            )

        if dem.size == 0:
            raise ValueError(
                "dem must not be empty."
            )

        if not np.all(np.isfinite(dem)):
            raise ValueError(
                "dem contains non-finite values."
            )

        cell_size = float(cell_size)

        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError(
                "cell_size must be a finite positive number."
            )

        # Store copies so external modifications do not unexpectedly alter
        # the solver's terrain.
        self.dem = dem.copy()

        self.cell_size = cell_size

        # --------------------------------------------------------------
        # Manning roughness
        # --------------------------------------------------------------

        if np.isscalar(manning_n):

            roughness = float(manning_n)

            if (
                not math.isfinite(roughness)
                or roughness <= 0.0
            ):
                raise ValueError(
                    "manning_n must be positive."
                )

            self.manning_n = np.full(
                dem.shape,
                roughness,
                dtype=float,
            )

        else:

            roughness = np.asarray(
                manning_n,
                dtype=float,
            )

            if roughness.shape != dem.shape:
                raise ValueError(
                    "manning_n array must have the same "
                    "shape as dem."
                )

            if not np.all(np.isfinite(roughness)):
                raise ValueError(
                    "manning_n contains non-finite values."
                )

            if np.any(roughness <= 0.0):
                raise ValueError(
                    "all Manning n values must be positive."
                )

            self.manning_n = roughness.copy()

        # --------------------------------------------------------------
        # Dynamic state
        # --------------------------------------------------------------

        # Water depth above ground [m].
        self.depth = np.zeros(
            dem.shape,
            dtype=float,
        )

        # Unit-width discharge [m²/s].
        #
        # qx[i,j] is the discharge crossing the vertical face between
        # cells (i,j) and (i,j+1).
        #
        # qy[i,j] is the discharge crossing the horizontal face between
        # cells (i,j) and (i+1,j).
        #
        # We keep the arrays the same shape as the DEM and use the unused
        # outer faces as zero-flux boundaries.
        self.qx = np.zeros(
            dem.shape,
            dtype=float,
        )

        self.qy = np.zeros(
            dem.shape,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns)."""

        return self.depth.shape

    @property
    def cell_area(self) -> float:
        """Area of one surface cell in m²."""

        return self.cell_size**2

    # ------------------------------------------------------------------
    # Mass accounting
    # ------------------------------------------------------------------

    def total_volume_m3(self) -> float:
        """Return total water volume currently stored on the grid."""

        return float(
            np.sum(self.depth)
            * self.cell_area
        )

    # ------------------------------------------------------------------
    # Source injection
    # ------------------------------------------------------------------

    def inject(
        self,
        row: int,
        col: int,
        volume_rate: float,
        dt: float,
    ) -> None:
        """Inject water into one surface cell.

        Args:
            row, col:
                Target grid cell.

            volume_rate:
                Source discharge in m³/s.

            dt:
                Duration of the source application in seconds.

        The injected volume is:

            dV = Q * dt

        and the corresponding depth increase is:

            dH = dV / cell_area
        """

        if not isinstance(row, (int, np.integer)):
            raise TypeError(
                "row must be an integer."
            )

        if not isinstance(col, (int, np.integer)):
            raise TypeError(
                "col must be an integer."
            )

        if not (
            0 <= row < self.depth.shape[0]
            and 0 <= col < self.depth.shape[1]
        ):
            raise IndexError(
                f"Surface cell ({row}, {col}) "
                f"is outside grid {self.depth.shape}."
            )

        volume_rate = float(volume_rate)
        dt = float(dt)

        if not math.isfinite(volume_rate):
            raise ValueError(
                "volume_rate must be finite."
            )

        if volume_rate < 0.0:
            raise ValueError(
                "volume_rate cannot be negative."
            )

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "dt must be a finite positive number."
            )

        self.depth[row, col] += (
            volume_rate
            * dt
            / self.cell_area
        )

        # Prevent tiny negative values caused by floating-point operations.
        self.depth[row, col] = max(
            0.0,
            self.depth[row, col],
        )

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        cfl: float = 0.4,
        max_substeps: int = 500,
    ) -> None:
        """Advance the surface model by ``dt`` seconds.

        The requested timestep is automatically divided into smaller
        shallow-water CFL-stable substeps.

        Args:
            dt:
                Total integration time in seconds.

            cfl:
                CFL safety factor.

            max_substeps:
                Maximum number of internal substeps.
        """

        dt = float(dt)

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "dt must be a finite positive number."
            )

        cfl = float(cfl)

        if (
            not math.isfinite(cfl)
            or cfl <= 0.0
            or cfl >= 1.0
        ):
            raise ValueError(
                "cfl must be between 0 and 1."
            )

        if (
            not isinstance(max_substeps, int)
            or max_substeps <= 0
        ):
            raise ValueError(
                "max_substeps must be a positive integer."
            )

        remaining = dt

        for _ in range(max_substeps):

            if remaining <= 1.0e-10:
                return

            hmax = float(
                np.max(self.depth)
            )

            if hmax <= MIN_DEPTH:
                # With essentially no water, there is no meaningful
                # shallow-water wave speed. A full step is safe because
                # there is no surface flow to amplify.
                sub_dt = remaining

            else:
                wave_speed = math.sqrt(
                    GRAVITY * hmax
                )

                stable_dt = (
                    cfl
                    * self.cell_size
                    / max(wave_speed, 1.0e-12)
                )

                sub_dt = min(
                    remaining,
                    stable_dt,
                )

            # Absolute safety against pathological floating-point values.
            if (
                not math.isfinite(sub_dt)
                or sub_dt <= 0.0
            ):
                raise RuntimeError(
                    "Surface solver produced an invalid "
                    "internal timestep."
                )

            self._step_once(sub_dt)

            remaining -= sub_dt

        # Do not silently leave part of the requested time unintegrated.
        #
        # If the CFL limit requires more than max_substeps, continue with
        # the smallest stable timestep rather than applying a large unstable
        # timestep.
        while remaining > 1.0e-10:

            hmax = float(
                np.max(self.depth)
            )

            if hmax <= MIN_DEPTH:
                sub_dt = remaining

            else:
                wave_speed = math.sqrt(
                    GRAVITY * hmax
                )

                stable_dt = (
                    cfl
                    * self.cell_size
                    / max(wave_speed, 1.0e-12)
                )

                sub_dt = min(
                    remaining,
                    stable_dt,
                )

            if (
                not math.isfinite(sub_dt)
                or sub_dt <= 0.0
            ):
                raise RuntimeError(
                    "Surface solver could not determine "
                    "a stable timestep."
                )

            self._step_once(sub_dt)

            remaining -= sub_dt

    # ------------------------------------------------------------------
    # One finite-volume step
    # ------------------------------------------------------------------

    def _step_once(
        self,
        dt: float,
    ) -> None:
        """Perform one explicit finite-volume surface-flow step."""

        rows, cols = self.depth.shape

        if rows < 1 or cols < 1:
            return

        # --------------------------------------------------------------
        # 1. Water surface elevation
        # --------------------------------------------------------------

        wse = (
            self.dem
            + self.depth
        )

        # --------------------------------------------------------------
        # 2. X-direction fluxes
        # --------------------------------------------------------------
        #
        # qx[:, j] represents flow across the face separating:
        #
        #     cell j  ->  cell j+1
        #
        # Positive qx means flow from left to right.
        # --------------------------------------------------------------

        qx_new = np.zeros_like(
            self.qx
        )

        if cols > 1:

            h_left = self.depth[:, :-1]
            h_right = self.depth[:, 1:]

            h_face = np.maximum(
                h_left,
                h_right,
            )

            water_exists = (
                h_face > MIN_FLOW_DEPTH
            )

            slope_x = (
                wse[:, 1:]
                - wse[:, :-1]
            ) / self.cell_size

            q_old = self.qx[:, :-1]

            n_face = 0.5 * (
                self.manning_n[:, :-1]
                + self.manning_n[:, 1:]
            )

            # Local inertial update.
            denominator = np.ones_like(
                h_face
            )

            wet = water_exists

            denominator[wet] += (
                GRAVITY
                * dt
                * n_face[wet] ** 2
                * np.abs(q_old[wet])
                / (
                    h_face[wet] ** (7.0 / 3.0)
                )
            )

            q_new = (
                q_old
                - GRAVITY
                * h_face
                * dt
                * slope_x
            ) / denominator

            q_new = np.where(
                water_exists,
                q_new,
                0.0,
            )

            # Numerical limiter: don't allow absurd discharge values
            # generated by a nearly dry cell.
            max_reasonable_q = (
                h_face
                * self.cell_size
                / max(dt, 1.0e-12)
            )

            q_new = np.clip(
                q_new,
                -max_reasonable_q,
                max_reasonable_q,
            )

            qx_new[:, :-1] = q_new

        # --------------------------------------------------------------
        # 3. Y-direction fluxes
        # --------------------------------------------------------------
        #
        # qy[i, :] represents flow across the face separating:
        #
        #     cell i  ->  cell i+1
        #
        # Positive qy means flow downward in array coordinates.
        # --------------------------------------------------------------

        qy_new = np.zeros_like(
            self.qy
        )

        if rows > 1:

            h_top = self.depth[:-1, :]
            h_bottom = self.depth[1:, :]

            h_face = np.maximum(
                h_top,
                h_bottom,
            )

            water_exists = (
                h_face > MIN_FLOW_DEPTH
            )

            slope_y = (
                wse[1:, :]
                - wse[:-1, :]
            ) / self.cell_size

            q_old = self.qy[:-1, :]

            n_face = 0.5 * (
                self.manning_n[:-1, :]
                + self.manning_n[1:, :]
            )

            denominator = np.ones_like(
                h_face
            )

            wet = water_exists

            denominator[wet] += (
                GRAVITY
                * dt
                * n_face[wet] ** 2
                * np.abs(q_old[wet])
                / (
                    h_face[wet] ** (7.0 / 3.0)
                )
            )

            q_new = (
                q_old
                - GRAVITY
                * h_face
                * dt
                * slope_y
            ) / denominator

            q_new = np.where(
                water_exists,
                q_new,
                0.0,
            )

            max_reasonable_q = (
                h_face
                * self.cell_size
                / max(dt, 1.0e-12)
            )

            q_new = np.clip(
                q_new,
                -max_reasonable_q,
                max_reasonable_q,
            )

            qy_new[:-1, :] = q_new

        # --------------------------------------------------------------
        # 4. Continuity equation
        # --------------------------------------------------------------
        #
        # dH/dt = -(div Q)
        #
        # Since qx/qy are unit-width discharges [m²/s], the divergence
        # is [m/s].
        #
        # For cell (i,j):
        #
        #   dH/dt =
        #       (qx_left - qx_right) / dx
        #     + (qy_top  - qy_bottom) / dy
        # --------------------------------------------------------------

        dh_dt = np.zeros_like(
            self.depth
        )

        if cols > 1:

            qx_left = np.zeros_like(
                self.depth
            )

            qx_right = np.zeros_like(
                self.depth
            )

            qx_right[:, :-1] = qx_new[:, :-1]
            qx_left[:, 1:] = qx_new[:, :-1]

            dh_dt += (
                qx_left
                - qx_right
            ) / self.cell_size

        if rows > 1:

            qy_top = np.zeros_like(
                self.depth
            )

            qy_bottom = np.zeros_like(
                self.depth
            )

            qy_bottom[:-1, :] = qy_new[:-1, :]
            qy_top[1:, :] = qy_new[:-1, :]

            dh_dt += (
                qy_top
                - qy_bottom
            ) / self.cell_size

        # --------------------------------------------------------------
        # 5. Update depth
        # --------------------------------------------------------------

        new_depth = (
            self.depth
            + dh_dt * dt
        )

        # Remove tiny negative numerical values.
        self.depth = np.maximum(
            0.0,
            new_depth,
        )

        # Prevent NaN/Inf from propagating through the coupled model.
        if not np.all(
            np.isfinite(self.depth)
        ):
            raise FloatingPointError(
                "2D surface solver produced "
                "non-finite water depths."
            )

        # Store fluxes after the depth update.
        self.qx = qx_new
        self.qy = qy_new

        # --------------------------------------------------------------
        # 6. Open boundary
        # --------------------------------------------------------------

        self._apply_open_boundary(
            dt
        )

    # ------------------------------------------------------------------
    # Open boundary
    # ------------------------------------------------------------------

    def _apply_open_boundary(
        self,
        dt: float,
        weir_coeff: float = 1.2,
    ) -> None:
        """Allow water to leave the outer boundary of the study tile.

        This is an approximate boundary treatment rather than a full
        transmissive boundary condition.

        The boundary discharge is represented using:

            Q ~ C * h^(3/2)

        and is applied independently to the outermost rows/columns.
        """

        rows, cols = self.depth.shape

        if rows < 2 and cols < 2:
            return

        weir_coeff = float(
            weir_coeff
        )

        if (
            not math.isfinite(weir_coeff)
            or weir_coeff < 0.0
        ):
            raise ValueError(
                "weir_coeff must be non-negative."
            )

        if weir_coeff == 0.0:
            return

        # We want the boundary depth reduction to correspond to:
        #
        #   dV = Q * dt
        #
        #   dH = dV / cell_area
        #
        # The discharge associated with one boundary cell is
        # approximately:
        #
        #   Q = C * h^(3/2) * cell_size
        #
        # Therefore:
        #
        #   dH = C*h^(3/2)*dt/cell_size
        #
        boundary_depth_change = (
            weir_coeff
            * np.power(
                np.maximum(
                    self.depth,
                    0.0,
                ),
                1.5,
            )
            * dt
            / self.cell_size
        )

        if rows >= 1:

            self.depth[0, :] = np.maximum(
                0.0,
                self.depth[0, :]
                - boundary_depth_change[0, :],
            )

            if rows > 1:

                self.depth[-1, :] = np.maximum(
                    0.0,
                    self.depth[-1, :]
                    - boundary_depth_change[-1, :],
                )

        if cols >= 1:

            self.depth[:, 0] = np.maximum(
                0.0,
                self.depth[:, 0]
                - boundary_depth_change[:, 0],
            )

            if cols > 1:

                self.depth[:, -1] = np.maximum(
                    0.0,
                    self.depth[:, -1]
                    - boundary_depth_change[:, -1],
                )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def depth_cm(self) -> np.ndarray:
        """Return water depth in centimetres."""

        return self.depth * 100.0