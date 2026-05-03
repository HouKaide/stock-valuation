"""Forecast domain models produced by valuation calculations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ForecastResult:
    """Explicit yearly forecast outputs used by downstream valuation steps."""

    yearly_forecast: pd.DataFrame
