import pytest
from pydantic import ValidationError

from prismx.portfolio import PortfolioInput, RiskAppetite, load_portfolio_config


def test_example_portfolio_loads():
    portfolio = load_portfolio_config("config/portfolio.example.yaml")

    assert portfolio.value_inr == 3_700_000
    assert portfolio.risk_appetite is RiskAppetite.MODERATE
    assert len(portfolio.holdings) == 6


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="weights must sum"):
        PortfolioInput(
            value_inr=100_000,
            risk_appetite="moderate",
            holdings=[
                {"name": "Test Co", "weight": 0.40},
                {"name": "Other Co", "weight": 0.40},
            ],
        )


def test_duplicate_holdings_are_rejected():
    with pytest.raises(ValidationError, match="duplicate names"):
        PortfolioInput(
            value_inr=100_000,
            risk_appetite="moderate",
            holdings=[
                {"name": "Test Co", "weight": 0.50},
                {"name": "test co", "weight": 0.50},
            ],
        )
