"""Local, no-server visualization of a demo forecast cycle.

Runs the same `run_one_cycle()` function the CLI and API both use, then
renders depth-grid snapshots at a few lead times to a PNG. Useful for
sanity-checking the model on a machine where you don't want to install
fastapi/uvicorn or open the Leaflet dashboard.

Usage:
    python scripts/visualize_demo.py
    python scripts/visualize_demo.py --scenario extreme --leads 0 15 30 60 120
    python scripts/visualize_demo.py --out my_run.png
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running as `python scripts/visualize_demo.py` from the repo root
# without requiring the package to be installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

from src.run_cycle import run_one_cycle


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="heavy",
                    choices=["light", "moderate", "heavy", "extreme"])
    p.add_argument("--leads", type=int, nargs="+", default=[0, 30, 60, 120],
                    help="Lead times (minutes) to render. Must be multiples "
                         "of the output interval (15 by default).")
    p.add_argument("--out", default="demo_frames.png")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Running scenario '{args.scenario}'...")
    result = run_one_cycle(scenario_id=args.scenario, verbose=True)
    depth_by_lead = result["depth_by_lead"]
    meta = result["meta"]
    node_cell_map = result["node_cell_map"]

    missing = [l for l in args.leads if l not in depth_by_lead]
    if missing:
        available = sorted(depth_by_lead.keys())
        sys.exit(f"Requested leads {missing} not in this run's output "
                  f"(available: {available}).")

    fig, axes = plt.subplots(1, len(args.leads), figsize=(4 * len(args.leads), 4.2))
    if len(args.leads) == 1:
        axes = [axes]

    vmax = max(depth_by_lead[l].max() for l in args.leads) or 1.0

    im = None
    for ax, lead in zip(axes, args.leads):
        grid = depth_by_lead[lead]
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"t+{lead} min\nmax {grid.max():.0f} cm", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        # Mark the network's known chronic flood point for reference.
        r, c = node_cell_map[meta["chronic_flood_node"]]
        ax.plot(c, r, marker="x", color="red", markersize=8, markeredgewidth=2)

    fig.suptitle(
        f"'{args.scenario}' scenario -- depth (cm), chronic flood node marked in red",
        fontsize=12,
    )
    fig.colorbar(im, ax=axes, shrink=0.8, label="depth (cm)")
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
