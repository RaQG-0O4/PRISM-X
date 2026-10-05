"""Build normal and stress correlation networks for the portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prismx.analytics import calculate_asset_returns, load_price_history, portfolio_weights_from_input
from prismx.dependence import pairwise_correlation, stress_period_returns
from prismx.network import (
    build_correlation_network,
    compare_network_summaries,
    network_edges,
    network_node_metrics,
    network_summary,
)
from prismx.portfolio import load_portfolio_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/portfolio.example.yaml")
    parser.add_argument("--prices", default="data/raw/prices/portfolio_adjusted_close.csv")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--tail-quantile", type=float, default=0.05)
    parser.add_argument("--output-dir", default="reports/tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    portfolio = load_portfolio_config(args.config)
    prices = load_price_history(args.prices)
    asset_returns = calculate_asset_returns(prices)
    weights = portfolio_weights_from_input(portfolio)

    normal_matrix = pairwise_correlation(asset_returns, weights)
    stress_matrix = stress_period_returns(
        asset_returns,
        weights,
        tail_quantile=args.tail_quantile,
    ).corr()

    normal_graph = build_correlation_network(normal_matrix, args.threshold)
    stress_graph = build_correlation_network(stress_matrix, args.threshold)
    normal_summary = network_summary(normal_graph, args.threshold)
    stress_summary = network_summary(stress_graph, args.threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "network_summary.json").write_text(
        json.dumps(normal_summary.as_dict(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "stress_network_summary.json").write_text(
        json.dumps(stress_summary.as_dict(), indent=2),
        encoding="utf-8",
    )
    compare_network_summaries(normal_summary, stress_summary).to_csv(
        output_dir / "network_summary_comparison.csv"
    )
    network_node_metrics(normal_graph, weights).to_csv(output_dir / "network_node_metrics.csv")
    network_node_metrics(stress_graph, weights).to_csv(
        output_dir / "stress_network_node_metrics.csv"
    )
    network_edges(normal_graph).to_csv(output_dir / "network_edges.csv", index=False)
    network_edges(stress_graph).to_csv(output_dir / "stress_network_edges.csv", index=False)

    print("Network analysis completed.")
    print(
        f"Normal network: {normal_summary.nodes} nodes, "
        f"{normal_summary.edges} edges, density {normal_summary.density:.2%}"
    )
    print(
        f"Stress network: {stress_summary.nodes} nodes, "
        f"{stress_summary.edges} edges, density {stress_summary.density:.2%}"
    )
    print("Most central normal-period holding:")
    print(f"  - {network_node_metrics(normal_graph, weights).index[0]}")
    print("Most central stress-period holding:")
    print(f"  - {network_node_metrics(stress_graph, weights).index[0]}")
    print(f"Reports saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
