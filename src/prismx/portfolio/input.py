"""Portfolio input models and validation helpers."""

from __future__ import annotations

from enum import Enum
from math import isclose
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RiskAppetite(str, Enum):
    """Risk profiles supported by the first PRISM-X version."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class HoldingInput(BaseModel):
    """One user-provided portfolio holding."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, description="User-entered company or security name")
    weight: float = Field(gt=0, le=1, description="Portfolio weight as a decimal")
    ticker: str | None = Field(default=None, description="Resolved market ticker, if known")

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Holding name cannot be blank")
        return value.strip()

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value.upper() or None


class PortfolioInput(BaseModel):
    """Validated portfolio input used by later PRISM-X modules."""

    model_config = ConfigDict(str_strip_whitespace=True)

    value_inr: float = Field(gt=0, description="Total portfolio value in INR")
    risk_appetite: RiskAppetite
    holdings: list[HoldingInput] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_holdings(self) -> "PortfolioInput":
        names = [holding.name.casefold() for holding in self.holdings]
        if len(names) != len(set(names)):
            raise ValueError("Holdings must not contain duplicate names")

        total_weight = sum(holding.weight for holding in self.holdings)
        if not isclose(total_weight, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Holding weights must sum to 1.0; received {total_weight:.6f}"
            )

        return self


def load_portfolio_config(path: str | Path) -> PortfolioInput:
    """Load and validate the portfolio section of a YAML configuration file."""

    from prismx.config import load_project_config

    return load_project_config(path).portfolio


def holding_values(portfolio: PortfolioInput) -> dict[str, float]:
    """Return the INR value assigned to each holding."""

    return {
        holding.name: portfolio.value_inr * holding.weight
        for holding in portfolio.holdings
    }
