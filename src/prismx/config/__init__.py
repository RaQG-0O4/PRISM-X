"""Configuration and project settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ProjectMetadata(BaseModel):
    name: str = "PRISM-X"
    reporting_currency: str = "INR"
    timezone: str = "Asia/Kolkata"
    frequency: Literal["daily", "weekly", "monthly"] = "daily"


class MarketSettings(BaseModel):
    benchmark: str = "NIFTY 50"
    history_years: int = Field(default=5, ge=1, le=50)
    rebalance_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly"


class ConstraintSettings(BaseModel):
    long_only: bool = True
    allow_leverage: bool = False
    maximum_single_holding: float = Field(default=0.30, gt=0, le=1)
    maximum_sector_exposure: float = Field(default=0.50, gt=0, le=1)
    transaction_cost_bps: float = Field(default=25.0, ge=0)


class RiskTargetSettings(BaseModel):
    severe_drawdown_horizon_days: int = Field(default=20, ge=1)
    severe_drawdown_threshold: float = Field(default=-0.10, lt=0)
    reverse_stress_target: float = Field(default=-0.20, lt=0)


class ProviderSettings(BaseModel):
    price_provider: str = "yfinance"
    news_provider: str = "yfinance"


class ProjectConfig(BaseModel):
    project: ProjectMetadata = Field(default_factory=ProjectMetadata)
    portfolio: Any
    market: MarketSettings = Field(default_factory=MarketSettings)
    constraints: ConstraintSettings = Field(default_factory=ConstraintSettings)
    risk_targets: RiskTargetSettings = Field(default_factory=RiskTargetSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load the complete validated project configuration."""

    from prismx.portfolio import PortfolioInput

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    if not isinstance(raw_config, dict):
        raise ValueError("Configuration root must be a mapping")
    if not isinstance(raw_config.get("portfolio"), dict):
        raise ValueError("Configuration must contain a 'portfolio' mapping")
    return ProjectConfig(
        project=raw_config.get("project", {}),
        portfolio=PortfolioInput.model_validate(raw_config["portfolio"]),
        market=raw_config.get("market", {}),
        constraints=raw_config.get("constraints", {}),
        risk_targets=raw_config.get("risk_targets", {}),
        providers=raw_config.get("providers", {}),
    )


__all__ = [
    "ConstraintSettings",
    "MarketSettings",
    "ProjectConfig",
    "ProjectMetadata",
    "ProviderSettings",
    "RiskTargetSettings",
    "load_project_config",
]
