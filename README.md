# Urban Flood Nowcasting System

Real-time, 0-3hr street-level flood forecasting for Indian metros. See
[`docs/DESIGN.md`](docs/DESIGN.md) for full architecture.

## Status

Working demo: synthetic terrain + a branching demo drainage network, coupled
1D pipe routing + 2D surface inundation, a scenario system (light/moderate/
heavy/extreme rainfall), and a FastAPI + Leaflet dashboard. The catchment-
aggregation bug that previously made the model predict zero flooding under
any rainfall is fixed (see `src/model/catchments.py`). Still **not
calibrated** against a real catchment or historical event -- see
"Next steps" below.

## Layout

```
src/model/        Core simulation (drainage graph, 1D routing, 2D solver,
                   catchment delineation, coupling loop, diagnostics)
src/ingestion/     DEM, radar, drainage-network loaders (+ synthetic fallbacks)
src/api/           FastAPI serving layer (forecast, diagnostics, routing)
src/dashboard/     Leaflet-based GIS dashboard
configs/           Demo rainfall scenario presets
tests/             pytest suite
```

## Quick local test (no server, no dashboard)

The simulation core only needs `numpy` and `networkx` -- it does not need
`fastapi`/`uvicorn` at all. This is the fastest way to sanity-check the
model on a machine where you don't want to stand up the API:

```bash
pip install numpy networkx

python -m src.run_cycle --scenario heavy
```

That's it -- no `--serve`, no server, no dashboard. It builds the synthetic
network + terrain, runs the coupled forecast, and prints a per-lead-time
summary straight to the console (max depth, cell count over 5cm, final
network surcharge state). Swap `--scenario` for `light`, `moderate`, or
`extreme` to compare storm intensities (see `configs/scenarios.py`).

If you also have `matplotlib` installed, `scripts/visualize_demo.py` calls
`run_one_cycle()` directly (same function the CLI uses) and saves a PNG
showing the flood extent at several lead times, so you can see the
inundation pattern without needing the Leaflet dashboard or a live API:

```bash
pip install matplotlib
python scripts/visualize_demo.py --scenario heavy --out demo_frames.png
```

## Run it (full stack: API + dashboard)

```bash
pip install -r requirements.txt

# run one forecast cycle on synthetic data and print a summary
python -m src.run_cycle --scenario heavy

# run it and load the result into the API's in-memory store, then serve
python -m src.run_cycle --scenario heavy --serve
uvicorn src.api.main:app --reload --port 8000
```

Or with Docker:

```bash
docker build -t flood-nowcast .
docker run -p 8000:8000 flood-nowcast
```

Then open `src/dashboard/index.html` in a browser (point `API_BASE` at the
running server), or call `/simulate/run` from the API directly to trigger a
run without the CLI.

Scenarios: `light`, `moderate`, `heavy`, `extreme` (see `configs/scenarios.py`).

## Tests

```bash
python -m pytest tests/ -v
```

## Next steps, in priority order

1. **Retune network capacity vs. rainfall intensity.** This is the
   highest-priority fix -- right now the model's *behavior* (coupling,
   surcharge detection, backflow) is correct, but the *magnitude* it
   outputs is not usable for a demo or for judging without this caveat
   attached. Confirmed on an actual run of the `heavy` scenario
   (85 mm/hr peak): max depth hits 105cm by t+30min and 363cm by t+120min,
   with 11/25 nodes surcharged and the worst pipe at 100% utilization by
   the end of the run. Even `moderate` (45 mm/hr) reaches 54cm by t+30min
   and 395cm by t+165min. Real streets do not pool to basement depth from
   ordinary heavy rain, so anything past roughly t+30-45min on the current
   demo network reads as a stuck/undersized-drain scenario rather than a
   generic storm, regardless of which scenario preset is selected.

   Root cause: the demo network has only 25 inlets spread across a
   480x480m tile (~5 nodes per side, ~96m average inlet spacing). Real
   Indian municipal storm-drain networks in dense urban areas typically
   run inlet spacing in the 30-60m range -- so the demo network is
   roughly 2-3x too sparse, and the water has nowhere to drain to once
   local pipe capacity is exceeded. Concretely:
   - **Add more inlets** to bring spacing down into a realistic range
     (roughly 3-9x the current node count for the same tile size), so
     each catchment area shrinks and no single pipe carries an
     unrealistic share of the runoff.
   - **Or resize/re-tune existing pipes**: increase diameters and/or
     revisit the Manning's roughness / capacity formula in
     `src/model/hydraulic_1d.py` against real storm-drain design
     standards (e.g. IS 1742 / CPHEEO manual sizing) rather than the
     current placeholder values.
   - **Validate against a return-period rainfall table**, not just named
     presets: a 2-year design storm should leave the network mostly dry,
     a 25-year storm should show visible but localized surcharge, and
     only an extreme/100-year event should approach the network-wide
     flooding currently seen under "heavy".
   - Whichever fix is used, re-run `scripts/visualize_demo.py` across all
     four scenario presets and confirm "light"/"moderate" stay mostly dry
     and only "heavy"/"extreme" show localized (not domain-wide) ponding
     before considering this closed.
2. **Calibrate against a real sub-catchment** with known pipe as-builts and
   at least one historical flood event.
3. **Wire in real data sources**: IMD radar feed, city LiDAR DEM, municipal
   drainage GIS export (loaders already support this path, see
   `src/ingestion/`).
4. **GPU-accelerate `surface_2d.py`** once grid size moves from demo scale
   to city-block scale (millions of cells).
5. **Replace the routing placeholder** with a real road-graph (OSRM/Valhalla).
6. **Add the orchestration job** (Airflow/Temporal) that triggers a cycle on
   new radar data.
