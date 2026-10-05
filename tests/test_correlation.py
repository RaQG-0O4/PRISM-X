import numpy as np
import pandas as pd

from prismx.dependence import (
    correlation_pairs,
    pairwise_correlation,
    rolling_pairwise_correlation,
)


def test_pairwise_correlation_matrix_is_symmetric():
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.03],
            "BBB": [0.02, 0.04, -0.02, 0.06],
            "CCC": [-0.01, 0.00, 0.02, -0.01],
        }
    )

    matrix = pairwise_correlation(returns)

    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert len(correlation_pairs(matrix)) == 3


def test_rolling_pairwise_correlation_returns_long_format():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.03, 0.01, -0.02],
            "BBB": [0.02, 0.04, -0.02, 0.06, 0.02, -0.04],
        },
        index=index,
    )

    rolling = rolling_pairwise_correlation(returns, window=3)

    assert list(rolling.columns) == ["date", "asset_1", "asset_2", "correlation"]
    assert len(rolling) == 4
