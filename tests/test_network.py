import pandas as pd
import pytest

nx = pytest.importorskip("networkx")

from prismx.network import (  # noqa: E402
    build_correlation_network,
    network_node_metrics,
    network_summary,
)


def test_correlation_network_adds_edges_above_threshold():
    matrix = pd.DataFrame(
        [
            [1.0, 0.80, 0.10],
            [0.80, 1.0, 0.70],
            [0.10, 0.70, 1.0],
        ],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )

    graph = build_correlation_network(matrix, threshold=0.50)
    summary = network_summary(graph, threshold=0.50)

    assert summary.nodes == 3
    assert summary.edges == 2
    assert summary.density == 2 / 3


def test_network_node_metrics_include_all_nodes():
    graph = nx.Graph()
    graph.add_nodes_from(["AAA", "BBB"])
    graph.add_edge("AAA", "BBB", correlation=0.8, weight=0.8, distance=1.25)

    metrics = network_node_metrics(graph, {"AAA": 0.60, "BBB": 0.40})

    assert list(metrics.index) == ["AAA", "BBB"]
    assert metrics.loc["AAA", "portfolio_weight"] == 0.60
