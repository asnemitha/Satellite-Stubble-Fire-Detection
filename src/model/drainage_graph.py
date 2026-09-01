"""Graph model of the underground drainage network.

Nodes = manholes / inlets. Edges = pipes / canals.
This is the structural backbone that the 1D hydraulic solver operates on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Node:
    id: str
    x: float
    y: float
    ground_elevation: float   # m, street/surface level
    invert_elevation: float   # m, pipe channel bottom level at this node
    is_outfall: bool = False  # discharges to river/sea, no backpressure

    # runtime state, updated each timestep by the 1D solver
    head: float = 0.0         # current hydraulic grade line elevation (m)
    inflow: float = 0.0       # external inflow this timestep (m3/s), e.g. surface runoff

    @property
    def max_depth(self) -> float:
        return self.ground_elevation - self.invert_elevation

    @property
    def is_surcharged(self) -> bool:
        """True once the hydraulic grade line reaches the surface -> street flooding."""
        return self.head >= self.ground_elevation

    @property
    def fill_fraction(self) -> float:
        """How full the node is, 0 (invert) -> 1 (ground level), can exceed 1 when surcharged."""
        span = self.max_depth
        if span <= 0:
            return 1.0
        return (self.head - self.invert_elevation) / span


@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str
    length: float              # m
    diameter: float            # m (circular pipe equivalent)
    slope: float                # m/m, positive = downhill from_node -> to_node
    manning_n: float = 0.013    # roughness coefficient (concrete ~0.013)
    condition_factor: float = 1.0  # 0-1 derate for siltation/blockage/collapse

    # runtime state
    flow: float = 0.0           # m3/s, current flow

    @property
    def area(self) -> float:
        return math.pi * (self.diameter / 2) ** 2

    def full_flow_capacity(self) -> float:
        """Manning's equation full-pipe capacity (m3/s), derated by condition."""
        if self.slope <= 0:
            return 0.0
        r_h = self.diameter / 4  # hydraulic radius of a full circular pipe
        v = (1.0 / self.manning_n) * (r_h ** (2 / 3)) * math.sqrt(self.slope)
        return v * self.area * self.condition_factor

    @property
    def utilization(self) -> float:
        """Current flow as a fraction of full-pipe capacity (0-1+, diagnostics/UI)."""
        cap = self.full_flow_capacity()
        return (self.flow / cap) if cap > 0 else 0.0


@dataclass
class DrainageGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    _adj: dict[str, list[str]] = field(default_factory=dict)  # node_id -> outgoing edge ids

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self._adj.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        self.edges[edge.id] = edge
        self._adj.setdefault(edge.from_node, []).append(edge.id)
        # Ensure to_node also has an adjacency-list entry even if it was never
        # explicitly added via add_node (e.g. the outfall in auto-built networks).
        # Without this, downstream_edges() silently returns [] for such nodes.
        self._adj.setdefault(edge.to_node, [])

    def downstream_edges(self, node_id: str) -> list[Edge]:
        return [self.edges[eid] for eid in self._adj.get(node_id, [])]

    def surcharged_nodes(self) -> list[Node]:
        """Nodes currently backing up to street level -> sources for the 2D surface model."""
        return [n for n in self.nodes.values() if n.is_surcharged]

    @classmethod
    def from_geojson_like(cls, nodes_data: list[dict], edges_data: list[dict]) -> "DrainageGraph":
        g = cls()
        for nd in nodes_data:
            g.add_node(Node(**nd))
        for ed in edges_data:
            g.add_edge(Edge(**ed))
        return g
