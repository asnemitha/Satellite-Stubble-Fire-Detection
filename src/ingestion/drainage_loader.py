"""Drainage network ingestion.

Production target: parse municipal storm-water GIS exports (shapefile/GeoJSON
of manholes + pipe centerlines) into the DrainageGraph model. Indian municipal
records are frequently incomplete or outdated (see docs/DESIGN.md sec 7), so
this module also supports a `condition_factor` override table for calibrating
effective capacity against known flood incidents rather than trusting nominal
pipe diameters blindly.
"""
from __future__ import annotations

import csv
import json

from src.model.drainage_graph import DrainageGraph, Edge, Node


def load_from_geojson(nodes_path: str, edges_path: str) -> DrainageGraph:
    """Load nodes/edges from GeoJSON FeatureCollections.

    Expected node properties: id, ground_elevation, invert_elevation, is_outfall
    Expected edge properties: id, from_node, to_node, length, diameter, slope,
                               manning_n, condition_factor
    Geometry x/y for nodes is taken from the Point coordinates.
    """
    graph = DrainageGraph()

    with open(nodes_path) as f:
        nodes_fc = json.load(f)
    for feat in nodes_fc["features"]:
        props = feat["properties"]
        x, y = feat["geometry"]["coordinates"][:2]
        graph.add_node(Node(
            id=props["id"],
            x=x, y=y,
            ground_elevation=props["ground_elevation"],
            invert_elevation=props["invert_elevation"],
            is_outfall=props.get("is_outfall", False),
        ))

    with open(edges_path) as f:
        edges_fc = json.load(f)
    for feat in edges_fc["features"]:
        props = feat["properties"]
        graph.add_edge(Edge(
            id=props["id"],
            from_node=props["from_node"],
            to_node=props["to_node"],
            length=props["length"],
            diameter=props["diameter"],
            slope=props["slope"],
            manning_n=props.get("manning_n", 0.013),
            condition_factor=props.get("condition_factor", 1.0),
        ))

    return graph


def apply_condition_calibration(graph: DrainageGraph, calibration_csv: str) -> None:
    """Override condition_factor per edge from a calibration file.

    CSV columns: edge_id, condition_factor
    Use this to encode field/historical-incident knowledge ("this pipe is
    50% silted", "this culvert is known to back up in every monsoon") that
    as-built diameter alone won't capture.
    """
    with open(calibration_csv) as f:
        for row in csv.DictReader(f):
            edge = graph.edges.get(row["edge_id"])
            if edge is not None:
                edge.condition_factor = float(row["condition_factor"])


def node_to_grid_cell(node: Node, grid_origin: tuple[float, float], cell_size: float) -> tuple[int, int]:
    """Map a node's (x, y) coordinate to a (row, col) index on the DEM/surface grid."""
    origin_x, origin_y = grid_origin
    col = int((node.x - origin_x) / cell_size)
    row = int((node.y - origin_y) / cell_size)
    return row, col


def build_node_cell_map(graph: DrainageGraph, grid_origin: tuple[float, float],
                          cell_size: float) -> dict[str, tuple[int, int]]:
    return {nid: node_to_grid_cell(n, grid_origin, cell_size) for nid, n in graph.nodes.items()}
