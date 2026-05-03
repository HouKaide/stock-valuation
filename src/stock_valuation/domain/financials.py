"""Historical financial statement models for valuation inputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HistoricalFinancials:
    """Chronologically aligned historical financial series for valuation."""

    revenue: pd.Series
    ebit: pd.Series
    tax_rate: pd.Series
    depreciation: pd.Series
    capex: pd.Series
    change_in_working_capital: pd.Series
    cash: pd.Series
    debt: pd.Series
    minority_interest: pd.Series | None
    non_operating_assets: pd.Series | None
