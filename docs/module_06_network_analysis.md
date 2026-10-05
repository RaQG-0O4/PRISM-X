# Module 6: Portfolio Interconnectedness Network

## Problem being solved

Correlation tables show pairwise relationships, but they do not immediately reveal which holdings connect multiple parts of the portfolio. A network representation makes concentrated relationships easier to inspect.

## Network design

- Each holding is a node.
- An edge is added when absolute correlation is at least the selected threshold.
- The signed correlation is retained so positive and negative relationships remain distinguishable.
- Edge weight is absolute correlation strength.
- The initial threshold is 0.50 and will be tested for sensitivity.

## Measures included

- Network density
- Number of edges
- Connected components
- Largest connected-component share
- Degree and degree centrality
- Weighted degree
- Betweenness centrality
- Weighted clustering

The same analysis is performed for normal observations and the portfolio's worst 5% historical days. A holding with high centrality is not automatically “bad”; it is a candidate for closer risk investigation.

## Install and run

NetworkX is part of the optional advanced dependencies. Install it now with:

```powershell
python -m pip install networkx
```

Then run:

```powershell
python scripts/analyse_network.py
```

Outputs:

- `reports/tables/network_summary.json`
- `reports/tables/stress_network_summary.json`
- `reports/tables/network_summary_comparison.csv`
- `reports/tables/network_node_metrics.csv`
- `reports/tables/stress_network_node_metrics.csv`
- `reports/tables/network_edges.csv`
- `reports/tables/stress_network_edges.csv`
