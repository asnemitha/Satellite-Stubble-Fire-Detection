# Urban Flood Nowcasting System — Design Document

## 1. Objective

Predict street-level urban flood depth (cm resolution) for a 0–3 hour lead time in
Indian metro areas, driven by micro-topography, surface imperviousness, and drainage
network capacity. Serve results through a GIS dashboard and a routing API for
flood-safe navigation.

## 2. System Overview

```
Radar QPN ──┐
DEM ────────┼──> Fusion Layer ──> 1D Drainage Routing ──> 2D Surface Inundation ──> Depth Grid
Drainage DB ┘                          (SWMM-style)         (shallow-water solver)      │
                                                                                          ├──> GIS Dashboard
                                                                                          └──> Routing API
```

Runs on a rolling cycle: ingest → simulate → publish, every 5–10 minutes, producing a
forecast raster stack for t+0 to t+180 min at 5–15 min steps.

## 3. Data Layer

| Source | Content | Refresh | Notes |
|---|---|---|---|
| Doppler radar | Rainfall reflectivity → QPE, QPN via extrapolation | 5–10 min | Blend with NWP (WRF) beyond ~60 min; India Meteorological Department (IMD) DWR network |
| LiDAR/photogrammetric DEM | Bare-earth elevation, ≤1 m resolution | Static, updated periodically | Building footprints removed to get true street-level DTM |
| Land use / land cover | Imperviousness, Manning's roughness | Static/seasonal | From satellite imagery classification |
| Drainage network | Manholes/inlets (nodes), pipes/canals (edges) | Static + condition updates | Often incomplete in Indian municipal records — see §7 |
| Ground truth | IoT water-level sensors, crowd-sourced reports | Real-time | Used for bias correction / assimilation |

### Drainage graph schema

```
Node (manhole/inlet):
  id, x, y, ground_elevation, invert_elevation, max_head, type

Edge (pipe/canal):
  id, from_node, to_node, diameter_or_geometry, slope,
  manning_n, condition_factor (0-1, capacity derating for siltation/blockage)
```

## 4. Modeling Core

### 4.1 Rainfall → Runoff
Per DEM grid cell, convert rainfall intensity to effective runoff using SCS Curve
Number or Green-Ampt infiltration, parameterized by imperviousness fraction. Impervious
cells (roads, rooftops) route ~85-95% of rainfall to surface/inlets quickly; pervious
cells attenuate and delay.

### 4.2 1D Drainage Routing
Route runoff entering at inlet nodes through the pipe graph using a kinematic/dynamic
wave solver (Saint-Venant equations, SWMM-equivalent). Computes, per node per timestep:
- flow depth and hydraulic grade line (HGL)
- pipe capacity utilization
- surcharge state (HGL > pipe crown)

When a node's HGL exceeds its ground elevation, the excess discharge becomes a **source
term** for the 2D surface model — this is the flood-onset condition.

### 4.3 2D Surface Inundation
A reduced shallow-water / diffusive-wave solver propagates surface water across the DEM
mesh, cell to cell, governed by local slope and Manning's roughness. Surcharging nodes
inject flow; surface inlets (where capacity allows) re-absorb flow back into the 1D
network — enabling reciprocal 1D↔2D coupling, not just one-way overflow.

**Numerical approach:** explicit finite-volume scheme (e.g. LISFLOOD-FP-style local
inertial approximation) — accurate enough for pluvial urban flooding, far cheaper than
full Saint-Venant 2D, and GPU-parallelizable across grid cells.

### 4.4 Coupling loop (per timestep, per drainage catchment)
1. Rainfall nowcast → runoff per cell
2. Runoff routed to nearest inlet node
3. 1D solver updates pipe flows/HGL
4. Surcharge nodes → 2D source terms
5. 2D solver propagates surface depth
6. Cells over surface inlets attempt re-entry into 1D network if capacity allows
7. Advance timestep, repeat to t+180 min

## 5. Compute Architecture

- **Streaming ingestion**: Kafka/Pulsar topics per data source; radar and sensor feeds
  are highest frequency.
- **Simulation engine**: GPU-accelerated (CUDA/OpenCL) 2D solver; 1D solver is
  lightweight and CPU-parallel across independent sub-catchments.
- **Orchestration**: cycle scheduler (Airflow/Temporal) triggers a full nowcast run
  whenever new radar data lands, with fallback to scheduled interval.
- **Output store**: forecast depth rasters as cloud-optimized GeoTIFF / vector tiles,
  written to object storage, versioned by forecast-issue-time.

## 6. Outputs

### 6.1 GIS Dashboard (web)
- Map tiles showing depth-by-street, color banded (e.g. 0–5cm, 5–15, 15–30, 30+ cm)
- Time slider across the 0–3hr forecast horizon
- Click-through to node/pipe level diagnostics (capacity %, confidence)

### 6.2 Routing API
- Input: origin, destination, timestamp
- Process: intersect road graph edges with forecast depth grid at the relevant forecast
  time; mark edges impassable above a depth threshold (default 15cm for cars, configurable
  for pedestrians/emergency vehicles)
- Output: route avoiding flooded edges, with confidence/alternate routes
- Consumers: navigation map integrations, emergency dispatch systems

## 7. Known Challenges (design honestly, not glossed over)

- **Real-time performance**: city-block-resolution 2D solving over a whole metro is the
  dominant compute cost — GPU parallelization is not optional at this scale.
- **Drainage data quality**: Indian municipal as-built records are frequently outdated
  or incomplete. Mitigation: infer effective pipe capacity from historical flood
  incident correlation, not just nominal pipe diameter; treat `condition_factor` as a
  learned/calibrated parameter, not a static input.
- **Radar nowcast skill decay**: extrapolation-based nowcasts lose skill past ~60–90 min
  for convective monsoon cells. Publish per-cell forecast uncertainty, not just a point
  value, especially near the 3hr edge.
- **Validation**: sparse ground-truth sensors — use crowd-sourced/social reports and
  post-event surveys for calibration.

## 8. Repository Layout

```
flood-nowcast/
  docs/DESIGN.md          this document
  src/ingestion/           radar, DEM, drainage network loaders
  src/model/               1D routing, 2D solver, coupling loop
  src/api/                 FastAPI routing + forecast query service
  src/dashboard/           web GIS frontend
  configs/                 city/catchment configuration
  tests/
```
