"""Local, no-server visualization of a demo forecast cycle.

Runs the same run_one_cycle() function used by the CLI and API,
then renders depth-grid snapshots at selected lead times to a PNG.

Usage:
    python scripts/visualize_demo.py
    python scripts/visualize_demo.py --scenario extreme --leads 15 30 60 120
    python scripts/visualize_demo.py --out my_run.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Configure matplotlib.
# ----------------------------------------------------------------------

matplotlib.use("Agg")


# ----------------------------------------------------------------------
# Add project root to Python path.
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Project import.
# ----------------------------------------------------------------------

from src.run_cycle import run_one_cycle


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--scenario",
        default="heavy",
        choices=[
            "light",
            "moderate",
            "heavy",
            "extreme",
        ],
        help="Rainfall scenario to simulate.",
    )

    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=[15, 30, 60, 120],
        help=(
            "Lead times in minutes to render. "
            "They must exist in the forecast output."
        ),
    )

    parser.add_argument(
        "--out",
        default="demo_frames.png",
        help="Output PNG filename.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the forecast and create the visualization."""

    args = parse_args()

    # ------------------------------------------------------------------
    # Validate arguments.
    # ------------------------------------------------------------------

    if any(lead < 0 for lead in args.leads):
        raise SystemExit(
            "Lead times cannot be negative."
        )

    if len(set(args.leads)) != len(args.leads):
        raise SystemExit(
            "Duplicate lead times were provided."
        )

    print(
        f"Running scenario '{args.scenario}'..."
    )

    # ------------------------------------------------------------------
    # Run forecast.
    # ------------------------------------------------------------------

    try:
        result = run_one_cycle(
            scenario_id=args.scenario,
            verbose=True,
        )
    except Exception as exc:
        print(
            "\nForecast failed.",
            file=sys.stderr,
        )
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    # ------------------------------------------------------------------
    # Extract forecast data.
    # ------------------------------------------------------------------

    depth_by_lead = result["depth_by_lead"]
    meta = result["meta"]
    node_cell_map = result["node_cell_map"]

    available = sorted(
        depth_by_lead.keys()
    )

    missing = [
        lead
        for lead in args.leads
        if lead not in depth_by_lead
    ]

    if missing:
        raise SystemExit(
            "\nRequested lead times are not available.\n"
            f"Requested: {missing}\n"
            f"Available: {available}\n\n"
            "Use one of the available lead times with --leads."
        )

    # ------------------------------------------------------------------
    # Find chronic flood point.
    # ------------------------------------------------------------------

    chronic_node = meta.get(
        "chronic_flood_node"
    )

    if chronic_node is None:
        raise SystemExit(
            "Metadata does not contain 'chronic_flood_node'."
        )

    if chronic_node not in node_cell_map:
        raise SystemExit(
            f"Chronic flood node '{chronic_node}' "
            "is missing from node_cell_map."
        )

    chronic_row, chronic_col = node_cell_map[
        chronic_node
    ]

    # ------------------------------------------------------------------
    # Create figure.
    # ------------------------------------------------------------------

    n_plots = len(args.leads)

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(
            max(4.0 * n_plots, 5.0),
            4.5,
        ),
        squeeze=False,
    )

    axes = axes[0]

    # ------------------------------------------------------------------
    # Common colour scale.
    # ------------------------------------------------------------------

    vmax = max(
        float(
            np.max(depth_by_lead[lead])
        )
        for lead in args.leads
    )

    if vmax <= 0.0 or not np.isfinite(vmax):
        vmax = 1.0

    # ------------------------------------------------------------------
    # Render forecast frames.
    # ------------------------------------------------------------------

    image = None

    for ax, lead in zip(
        axes,
        args.leads,
    ):
        grid = np.asarray(
            depth_by_lead[lead],
            dtype=float,
        )

        # Check dimensions.
        if grid.ndim != 2:
            raise ValueError(
                f"Depth grid at t+{lead} min is not 2D: "
                f"shape={grid.shape}"
            )

        # Check values.
        if not np.isfinite(grid).all():
            raise ValueError(
                f"Depth grid at t+{lead} min contains "
                "NaN or infinite values."
            )

        # Plot depth.
        image = ax.imshow(
            grid,
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(
            f"t+{lead} min\n"
            f"max {grid.max():.1f} cm",
            fontsize=11,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # Mark chronic flood point.
        ax.plot(
            chronic_col,
            chronic_row,
            marker="x",
            color="red",
            markersize=9,
            markeredgewidth=2,
        )

    # ------------------------------------------------------------------
    # Figure title.
    # ------------------------------------------------------------------

    fig.suptitle(
        (
            f"'{args.scenario}' scenario — "
            "surface water depth"
        ),
        fontsize=13,
    )

    # ------------------------------------------------------------------
    # Shared colour bar.
    # ------------------------------------------------------------------

    if image is not None:
        fig.colorbar(
            image,
            ax=axes.tolist(),
            shrink=0.8,
            label="Depth (cm)",
        )

    # ------------------------------------------------------------------
    # Layout.
    # ------------------------------------------------------------------

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.93)
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    output_path = Path(
        args.out
    ).expanduser()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved visualization to:"
    )
    print(
        f"  {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
