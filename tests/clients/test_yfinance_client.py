"""Tests for the Yahoo Finance client adapter."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from stock_valuation.clients import YFinanceClient
from stock_valuation.errors import (
    MarketDataUnavailableError,
    StatementUnavailableError,
    TickerNotFoundError,
)


class FakeTicker:
    """Simple ticker test double for exercising client behavior."""

    def __init__(
        self,
        *,
        info: dict[str, Any] | None = None,
        fast_info: dict[str, Any] | None = None,
        history: pd.DataFrame | None = None,
        income_statement: pd.DataFrame | None = None,
        balance_sheet: pd.DataFrame | None = None,
        cashflow: pd.DataFrame | None = None,
        shares: pd.Series | None = None,
        history_error: Exception | None = None,
        fast_info_error: Exception | None = None,
        info_error: Exception | None = None,
        statement_error: Exception | None = None,
        shares_error: Exception | None = None,
    ) -> None:
        """Initialize canned responses for the fake ticker."""

        self._info = info or {}
        self._fast_info = fast_info or {}
        self._history = history if history is not None else pd.DataFrame()
        self._income_statement = income_statement if income_statement is not None else pd.DataFrame()
        self._balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        self._cashflow = cashflow if cashflow is not None else pd.DataFrame()
        self._shares = shares
        self._history_error = history_error
        self._fast_info_error = fast_info_error
        self._info_error = info_error
        self._statement_error = statement_error
        self._shares_error = shares_error

    @property
    def info(self) -> dict[str, Any]:
        """Return canned ticker metadata."""

        if self._info_error is not None:
            raise self._info_error
        return self._info

    @property
    def fast_info(self) -> dict[str, Any]:
        """Return canned fast market data."""

        if self._fast_info_error is not None:
            raise self._fast_info_error
        return self._fast_info

    def history(
        self,
        *,
        period: str = "10y",
        interval: str = "1mo",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Return canned history data."""

        del period, interval, auto_adjust
        if self._history_error is not None:
            raise self._history_error
        return self._history

    def get_income_stmt(self, *, pretty: bool = True, freq: str = "yearly") -> pd.DataFrame:
        """Return canned income statement data."""

        del pretty, freq
        if self._statement_error is not None:
            raise self._statement_error
        return self._income_statement

    def get_balance_sheet(self, *, pretty: bool = True, freq: str = "yearly") -> pd.DataFrame:
        """Return canned balance sheet data."""

        del pretty, freq
        if self._statement_error is not None:
            raise self._statement_error
        return self._balance_sheet

    def get_cashflow(self, *, pretty: bool = True, freq: str = "yearly") -> pd.DataFrame:
        """Return canned cash flow data."""

        del pretty, freq
        if self._statement_error is not None:
            raise self._statement_error
        return self._cashflow

    def get_shares_full(self, *, start: date | None = None, end: date | None = None) -> pd.Series | None:
        """Return canned shares history."""

        del start, end
        if self._shares_error is not None:
            raise self._shares_error
        return self._shares


def test_get_ticker_normalizes_symbols_and_caches_instances() -> None:
    """Cache lookups should reuse the same ticker for normalized symbols."""

    calls: list[str] = []

    def ticker_factory(symbol: str) -> FakeTicker:
        calls.append(symbol)
        return FakeTicker()

    client = YFinanceClient(ticker_factory=ticker_factory)

    first_ticker = client.get_ticker(" msft ")
    second_ticker = client.get_ticker("MSFT")

    assert first_ticker is second_ticker
    assert calls == ["MSFT"]


def test_get_ticker_rejects_empty_symbols() -> None:
    """Empty symbols should raise the shared ticker-not-found error."""

    client = YFinanceClient(ticker_factory=lambda symbol: FakeTicker())

    with pytest.raises(TickerNotFoundError):
        client.get_ticker("   ")


def test_get_info_raises_ticker_not_found_when_empty() -> None:
    """Empty metadata should be treated as a missing ticker."""

    client = YFinanceClient(ticker_factory=lambda symbol: FakeTicker(info={}))

    with pytest.raises(TickerNotFoundError):
        client.get_info("AAPL")


def test_get_fast_info_maps_upstream_failure_to_market_data_error() -> None:
    """Fast-info failures should be wrapped in the shared market-data error."""

    client = YFinanceClient(ticker_factory=lambda symbol: FakeTicker(fast_info_error=RuntimeError("boom")))

    with pytest.raises(MarketDataUnavailableError):
        client.get_fast_info("AAPL")


def test_get_history_raises_when_history_is_empty() -> None:
    """Empty price history should raise a market-data error."""

    client = YFinanceClient(ticker_factory=lambda symbol: FakeTicker(history=pd.DataFrame()))

    with pytest.raises(MarketDataUnavailableError):
        client.get_history("AAPL")


def test_get_income_statement_maps_upstream_failure_to_statement_error() -> None:
    """Statement retrieval failures should raise the shared statement error."""

    client = YFinanceClient(ticker_factory=lambda symbol: FakeTicker(statement_error=RuntimeError("boom")))

    with pytest.raises(StatementUnavailableError):
        client.get_income_statement("AAPL")


def test_get_shares_full_returns_series_when_available() -> None:
    """Shares history should pass through the upstream series unchanged."""

    shares = pd.Series([100, 110], index=pd.to_datetime(["2024-01-01", "2025-01-01"]))
    client = YFinanceClient(ticker_factory=lambda symbol: FakeTicker(shares=shares))

    result = client.get_shares_full("AAPL")

    assert result is shares
