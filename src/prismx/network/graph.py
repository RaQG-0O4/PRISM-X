"""NetworkX-based portfolio interconnectedness analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd


class NetworkAnalysisError(RuntimeError):
    """Raised when NetworkX analysis cannot be performed."""


def _networkx() -> Any:
    try:
        import networkx as nx
    except ImportError as error:  # pragma: no cover - depends on environment
        raise NetworkAnalysisError(
            "NetworkX is not installed. Run: python -m pip install networkx"
        ) from error
    return nx


@dataclass(frozen=True)
class NetworkSummary:
    threshold: float
    nodes: int
    edges: int
    density: float
    connected_components: int
    largest_component_share: float
    average_clustering: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_correlation_network(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.50,
) -> Any:
    """Create an undirected graph from absolute correlations.

    Nodes are holdings. An edge is added when absolute correlation reaches the
    threshold. The signed correlation is retained as an edge attribute while
    edge weight stores its absolute strength.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if correlation_matrix.empty:
        raise ValueError("Correlation matrix cannot be empty")
    if set(correlation_matrix.index) != set(correlation_matrix.columns):
        raise ValueError("Correlation matrix must have the same row and column labels")

    nx = _networkx()
    graph = nx.Graph()
    columns = list(correlation_matrix.columns)
    graph.add_nodes_from(columns)

    for first_position, first_asset in enumerate(columns):
        for second_asset in columns[first_position + 1 :]:
            correlation = correlation_matrix.loc[first_asset, second_asset]
            if pd.isna(correlation):
                continue
            correlation = float(correlation)
            strength = abs(correlation)
            if strength >= threshold:
                graph.add_edge(
                    first_asset,
                    second_asset,
                    correlation=correlation,
                    weight=strength,
                    distance=1.0 / max(strength, 1e-12),
                )

    return graph


def network_summary(graph: Any, threshold: float) -> NetworkSummary:
    """Calculate graph-level interconnectedness measures."""

    nx = _networkx()
    component_sizes = [len(component) for component in nx.connected_components(graph)]
    largest_share = max(component_sizes, default=0) / max(graph.number_of_nodes(), 1)

    return NetworkSummary(
        threshold=threshold,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        density=float(nx.density(graph)),
        connected_components=nx.number_connected_components(graph),
        largest_component_share=float(largest_share),
        average_clustering=float(nx.average_clustering(graph, weight="weight")),
    )


def network_node_metrics(
    graph: Any,
    portfolio_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Calculate centrality and clustering measures for each holding."""

    nx = _networkx()
    nodes = list(graph.nodes)
    degree_centrality = nx.degree_centrality(graph)
    weighted_degree = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    clustering = nx.clustering(graph, weight="weight")

    result = pd.DataFrame(index=nodes)
    result.index.name = "asset"
    result["portfolio_weight"] = (
        pd.Series(portfolio_weights, dtype=float).reindex(nodes).fillna(0.0)
        if portfolio_weights is not None
        else 0.0
    )
    result["degree"] = pd.Series(dict(graph.degree()), dtype=float).reindex(nodes).fillna(0.0)
    result["degree_centrality"] = pd.Series(degree_centrality).reindex(nodes).fillna(0.0)
    result["weighted_degree"] = pd.Series(weighted_degree).reindex(nodes).fillna(0.0)
    result["betweenness_centrality"] = pd.Series(betweenness).reindex(nodes).fillna(0.0)
    result["weighted_clustering"] = pd.Series(clustering).reindex(nodes).fillna(0.0)
    return result.sort_values("degree_centrality", ascending=False)


def network_edges(graph: Any) -> pd.DataFrame:
    """Return graph edges and their signed/absolute correlation attributes."""

    rows: list[dict[str, object]] = []
    for first_asset, second_asset, attributes in graph.edges(data=True):
        rows.append(
            {
                "asset_1": first_asset,
                "asset_2": second_asset,
                "correlation": attributes.get("correlation"),
                "weight": attributes.get("weight"),
            }
        )
    return pd.DataFrame(rows, columns=["asset_1", "asset_2", "correlation", "weight"])


def compare_network_summaries(
    normal: NetworkSummary,
    stress: NetworkSummary,
) -> pd.DataFrame:
    """Compare normal and stress network-level measures."""

    normal_values = normal.as_dict()
    stress_values = stress.as_dict()
    rows: list[dict[str, object]] = []
    for metric in normal_values:
        if metric == "threshold":
            continue
        normal_value = float(normal_values[metric])
        stress_value = float(stress_values[metric])
        rows.append(
            {
                "metric": metric,
                "normal": normal_value,
                "stress": stress_value,
                "change": stress_value - normal_value,
            }
        )
    return pd.DataFrame(rows).set_index("metric")
