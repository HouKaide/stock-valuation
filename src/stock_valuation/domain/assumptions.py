"""Analyst-controlled valuation assumption models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class ValuationAssumptions:
    """Validated scalar inputs that control the valuation model."""

    symbol: str
    valuation_date: str
    forecast_years: int
    risk_free_rate: Decimal
    equity_risk_premium: Decimal
    country_risk_premium: Decimal
    tax_rate: Decimal
    pre_tax_cost_of_debt: Decimal
    target_debt_to_capital: Decimal
    terminal_growth_rate: Decimal
    sales_to_capital_ratio: Decimal
    initial_revenue_growth_rate: Decimal
    target_revenue_growth_rate: Decimal
    initial_operating_margin: Decimal
    target_operating_margin: Decimal
    failure_probability: Decimal
