"""Valuation result models for enterprise and equity outputs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd


@dataclass
class ValuationResult:
    """Final valuation outputs derived from the forecast and discounting steps."""

    forecast: pd.DataFrame
    cost_of_equity: Decimal
    after_tax_cost_of_debt: Decimal
    wacc: Decimal
    terminal_value: Decimal
    enterprise_value: Decimal
    equity_value: Decimal
    intrinsic_value_per_share: Decimal
    current_price: Decimal
    upside_downside_pct: Decimal
