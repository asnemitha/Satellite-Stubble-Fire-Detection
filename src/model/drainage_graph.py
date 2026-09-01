"""Graph model of the underground drainage network.

Nodes = manholes / inlets.
Edges = pipes / canals.

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

    # Elevations in metres.
    ground_elevation: float
    invert_elevation: float

    # True for the terminal node discharging to a river/sea.
    is_outfall: bool = False

    # Runtime state, updated by the 1D hydraulic solver.
    head: float = 0.0
    inflow: float = 0.0

    @property
    def max_depth(self) -> float:
        """Maximum available vertical storage between invert and ground."""
        return max(
            0.0,
            self.ground_elevation - self.invert_elevation,
        )

    @property
    def is_surcharged(self) -> bool:
        """Whether the hydraulic grade line has reached the street."""
        return self.head >= self.ground_elevation

    @property
    def fill_fraction(self) -> float:
        """Node hydraulic filling relative to invert-to-ground depth.

        Values:
            0.0 -> water level at invert
            1.0 -> water level at ground
            >1.0 -> hydraulic grade line above ground (surcharge)
        """

        span = self.max_depth

        if span <= 0.0:
            return 1.0 if self.head >= self.ground_elevation else 0.0

        return (
            self.head - self.invert_elevation
        ) / span


@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str

    length: float
    diameter: float
    slope: float

    manning_n: float = 0.013

    # 0 = completely blocked/unusable
    # 1 = nominal capacity
    condition_factor: float = 1.0

    # Runtime flow in m3/s.
    flow: float = 0.0

    @property
    def area(self) -> float:
        """Cross-sectional area of a circular pipe in m2."""
        if self.diameter <= 0.0:
            return 0.0

        return math.pi * (self.diameter / 2.0) ** 2

    def full_flow_capacity(self) -> float:
        """Manning full-pipe capacity in m3/s.

        Capacity is derated by condition_factor.
        """

        # Invalid geometry/roughness cannot convey flow.
        if (
            self.diameter <= 0.0
            or self.manning_n <= 0.0
            or self.slope <= 0.0
            or self.condition_factor <= 0.0
        ):
            return 0.0

        # Prevent invalid condition factors from artificially
        # increasing pipe capacity above nominal.
        condition = min(
            1.0,
            self.condition_factor,
        )

        # Hydraulic radius of a full circular pipe:
        # R = A/P = D/4.
        hydraulic_radius = self.diameter / 4.0

        velocity = (
            1.0 / self.manning_n
        ) * (
            hydraulic_radius ** (2.0 / 3.0)
        ) * math.sqrt(self.slope)

        capacity = (
            velocity
            * self.area
            * condition
        )

        return max(0.0, capacity)

    @property
    def utilization(self) -> float:
        """Current flow as a fraction of full-pipe capacity.

        Can exceed 1.0 when the pipe is overloaded.
        """

        capacity = self.full_flow_capacity()

        if capacity <= 0.0:
            return 0.0

        # Absolute value allows diagnostics to remain meaningful if
        # the hydraulic solver permits reverse flow.
        return abs(self.flow) / capacity


@dataclass
class DrainageGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    # node_id -> outgoing edge IDs
    _adj: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add or replace a drainage node."""

        self.nodes[node.id] = node
        self._adj.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        """Add a directed drainage edge.

        The edge is assumed to flow from from_node -> to_node.
        """

        if edge.from_node not in self.nodes:
            raise ValueError(
                f"Edge '{edge.id}' references unknown "
                f"from_node '{edge.from_node}'."
            )

        if edge.to_node not in self.nodes:
            raise ValueError(
                f"Edge '{edge.id}' references unknown "
                f"to_node '{edge.to_node}'."
            )

        self.edges[edge.id] = edge

        self._adj.setdefault(
            edge.from_node,
            [],
        )

        self._adj[edge.from_node].append(
            edge.id
        )

        self._adj.setdefault(
            edge.to_node,
            [],
        )

    def downstream_edges(self, node_id: str) -> list[Edge]:
        """Return all pipes leaving a node."""

        return [
            self.edges[edge_id]
            for edge_id in self._adj.get(node_id, [])
            if edge_id in self.edges
        ]

    def surcharged_nodes(self) -> list[Node]:
        """Return nodes whose hydraulic grade reaches the street."""

        return [
            node
            for node in self.nodes.values()
            if node.is_surcharged
        ]

    @classmethod
    def from_geojson_like(
        cls,
        nodes_data: list[dict],
        edges_data: list[dict],
    ) -> "DrainageGraph":
        """Construct a graph from node/edge dictionaries."""

        graph = cls()

        for node_data in nodes_data:
            graph.add_node(
                Node(**node_data)
            )

        for edge_data in edges_data:
            graph.add_edge(
                Edge(**edge_data)
            )

        return graph