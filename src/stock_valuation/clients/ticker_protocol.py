"""Protocol definitions for Yahoo Finance ticker objects."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pandas as pd


class TickerProtocol(Protocol):
    """Protocol describing the `yfinance.Ticker` surface used by the client."""

    @property
    def info(self) -> dict[str, Any]:
        """Return ticker metadata."""

    @property
    def fast_info(self) -> Mapping[str, Any]:
        """Return lightweight market data fields."""

    def history(
        self,
        *,
        period: str = "10y",
        interval: str = "1mo",
        auto_adjust: bool = True,
    ) -> "pd.DataFrame":
        """Return price history."""

    def get_income_stmt(self, *, pretty: bool = True, freq: str = "yearly") -> "pd.DataFrame":
        """Return income statement data."""

    def get_balance_sheet(self, *, pretty: bool = True, freq: str = "yearly") -> "pd.DataFrame":
        """Return balance sheet data."""

    def get_cashflow(self, *, pretty: bool = True, freq: str = "yearly") -> "pd.DataFrame":
        """Return cash flow data."""

    def get_shares_full(self, *, start: date | None = None, end: date | None = None) -> "pd.Series | None":
        """Return shares outstanding history."""
