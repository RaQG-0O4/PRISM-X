import numpy as np
import pandas as pd
import pytest

from prismx.optimisation import optimise_portfolios


def test_optimisation_requires_multiple_assets():
    returns = pd.DataFrame({"AAA": np.zeros(20)})

    with pytest.raises(ValueError, match="At least two assets"):
        optimise_portfolios(returns)


def test_optimisation_rejects_infeasible_weight_cap():
    index = pd.date_range("2024-01-01", periods=20, freq="B")
    returns = pd.DataFrame({"AAA": np.zeros(20), "BBB": np.zeros(20)}, index=index)

    with pytest.raises(ValueError, match="too small"):
        optimise_portfolios(returns, maximum_weight=0.40)


def test_optimisation_rejects_non_finite_returns():
    index = pd.date_range("2024-01-01", periods=20, freq="B")
    returns = pd.DataFrame({"AAA": np.zeros(20), "BBB": np.zeros(20)}, index=index)
    returns.iloc[5, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        optimise_portfolios(returns)
