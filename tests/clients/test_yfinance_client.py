"""Tests for the ticker-bound Yahoo Finance client."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

import stock_valuation.yfinance_client as yfinance_client_module
from stock_valuation import YFinanceClient as RootYFinanceClient
from stock_valuation.clients import YFinanceClient as ClientsYFinanceClient
from stock_valuation.errors import (
    MarketDataUnavailableError,
    MetricUnavailableError,
    StatementUnavailableError,
    TickerNotFoundError,
)
from stock_valuation.yfinance_client import YFinanceClient


class FakeTicker:
    """Fake yfinance ticker with canned responses and call tracking."""

    def __init__(
        self,
        symbol: str,
        *,
        info: dict[str, Any] | None = None,
        fast_info: dict[str, Any] | None = None,
        history: pd.DataFrame | None = None,
        income_statement: pd.DataFrame | None = None,
        balance_sheet: pd.DataFrame | None = None,
        cashflow: pd.DataFrame | None = None,
        shares: pd.Series | None = None,
        info_error: Exception | None = None,
        fast_info_error: Exception | None = None,
        history_error: Exception | None = None,
        statement_error: Exception | None = None,
        shares_error: Exception | None = None,
    ) -> None:
        self.symbol = symbol
        self._info = info if info is not None else {"symbol": symbol}
        self._fast_info = fast_info if fast_info is not None else {"shares": 100}
        self._history = history if history is not None else pd.DataFrame({"Close": [10]})
        self._income_statement = income_statement if income_statement is not None else pd.DataFrame({"2025": [1]})
        self._balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame({"2025": [2]})
        self._cashflow = cashflow if cashflow is not None else pd.DataFrame({"2025": [3]})
        self._shares = shares
        self._info_error = info_error
        self._fast_info_error = fast_info_error
        self._history_error = history_error
        self._statement_error = statement_error
        self._shares_error = shares_error
        self.history_calls: list[dict[str, Any]] = []
        self.income_statement_calls: list[dict[str, Any]] = []
        self.balance_sheet_calls: list[dict[str, Any]] = []
        self.cashflow_calls: list[dict[str, Any]] = []
        self.shares_calls: list[dict[str, Any]] = []

    def get_info(self) -> dict[str, Any]:
        """Return canned ticker metadata."""

        if self._info_error is not None:
            raise self._info_error
        return self._info

    def get_fast_info(self) -> dict[str, Any]:
        """Return canned fast metadata."""

        if self._fast_info_error is not None:
            raise self._fast_info_error
        return self._fast_info

    def get_income_stmt(self, *, pretty: bool = True, freq: str = "yearly") -> pd.DataFrame:
        """Return canned income statement data."""

        self.income_statement_calls.append({"pretty": pretty, "freq": freq})
        if self._statement_error is not None:
            raise self._statement_error
        return self._income_statement

    def get_balance_sheet(self, *, pretty: bool = True, freq: str = "yearly") -> pd.DataFrame:
        """Return canned balance sheet data."""

        self.balance_sheet_calls.append({"pretty": pretty, "freq": freq})
        if self._statement_error is not None:
            raise self._statement_error
        return self._balance_sheet

    def get_cashflow(self, *, pretty: bool = True, freq: str = "yearly") -> pd.DataFrame:
        """Return canned cash flow statement data."""

        self.cashflow_calls.append({"pretty": pretty, "freq": freq})
        if self._statement_error is not None:
            raise self._statement_error
        return self._cashflow

    def get_shares_full(self, *, start: date | None = None, end: date | None = None) -> pd.Series | None:
        """Return canned shares history."""

        self.shares_calls.append({"start": start, "end": end})
        if self._shares_error is not None:
            raise self._shares_error
        return self._shares

    def history(
        self,
        *,
        period: str = "10y",
        interval: str = "1mo",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Return canned price history."""

        self.history_calls.append({"period": period, "interval": interval, "auto_adjust": auto_adjust})
        if self._history_error is not None:
            raise self._history_error
        return self._history


class FakeLookup:
    """Fake yfinance lookup with per-method canned responses."""

    calls: list[tuple[str, str]] = []
    responses: dict[str, pd.DataFrame] = {}
    error: Exception | None = None

    def __init__(self, query: str) -> None:
        self.query = query

    @classmethod
    def reset(cls) -> None:
        """Reset fake lookup state."""

        cls.calls = []
        cls.responses = {}
        cls.error = None

    def _response(self, method_name: str) -> pd.DataFrame:
        FakeLookup.calls.append((self.query, method_name))
        if FakeLookup.error is not None:
            raise FakeLookup.error
        return FakeLookup.responses.get(method_name, pd.DataFrame({"name": ["Apple Inc."]}, index=["AAPL"]))

    def get_all(self) -> pd.DataFrame:
        """Return all fake instruments."""

        return self._response("get_all")

    def get_stock(self) -> pd.DataFrame:
        """Return fake stock instruments."""

        return self._response("get_stock")

    def get_mutualfund(self) -> pd.DataFrame:
        """Return fake mutual fund instruments."""

        return self._response("get_mutualfund")

    def get_etf(self) -> pd.DataFrame:
        """Return fake ETF instruments."""

        return self._response("get_etf")

    def get_index(self) -> pd.DataFrame:
        """Return fake index instruments."""

        return self._response("get_index")

    def get_future(self) -> pd.DataFrame:
        """Return fake future instruments."""

        return self._response("get_future")

    def get_currency(self) -> pd.DataFrame:
        """Return fake currency instruments."""

        return self._response("get_currency")

    def get_cryptocurrency(self) -> pd.DataFrame:
        """Return fake cryptocurrency instruments."""

        return self._response("get_cryptocurrency")


def patch_ticker(monkeypatch: pytest.MonkeyPatch, ticker: FakeTicker | Exception) -> list[str]:
    """Patch yfinance.Ticker and return the symbols requested."""

    calls: list[str] = []

    def fake_ticker(symbol: str) -> FakeTicker:
        calls.append(symbol)
        if isinstance(ticker, Exception):
            raise ticker
        return ticker

    monkeypatch.setattr(yfinance_client_module.yf, "Ticker", fake_ticker)
    return calls


def test_public_imports_expose_same_client() -> None:
    """The root, top-level module, and clients package should expose the client."""

    assert RootYFinanceClient is YFinanceClient
    assert ClientsYFinanceClient is YFinanceClient


def test_normalized_ticker_trims_and_uppercases() -> None:
    """Ticker normalization should produce the Yahoo lookup symbol."""

    assert YFinanceClient(" aapl ").normalized_ticker() == "AAPL"


def test_normalized_ticker_rejects_empty_ticker() -> None:
    """Empty ticker input should fail before yfinance is called."""

    with pytest.raises(TickerNotFoundError) as error:
        YFinanceClient("   ").normalized_ticker()

    assert error.value.symbol == "   "
    assert error.value.source_attempted == "ticker input"


def test_get_ticker_caches_one_yfinance_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client should construct one yfinance ticker and reuse it."""

    ticker = FakeTicker("AAPL")
    calls = patch_ticker(monkeypatch, ticker)
    client = YFinanceClient(" aapl ")

    assert client.get_ticker() is ticker
    assert client.get_ticker() is ticker
    assert calls == ["AAPL"]


def test_get_ticker_wraps_constructor_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor failures should be hidden behind the project ticker error."""

    patch_ticker(monkeypatch, RuntimeError("upstream exploded"))

    with pytest.raises(TickerNotFoundError) as error:
        YFinanceClient("AAPL").get_ticker()

    assert error.value.symbol == "AAPL"
    assert error.value.source_attempted == "yfinance.Ticker"
    assert "RuntimeError" not in str(error.value)


def test_get_info_returns_raw_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ticker info should pass through unchanged when available."""

    info = {"shortName": "Apple Inc."}
    patch_ticker(monkeypatch, FakeTicker("AAPL", info=info))

    assert YFinanceClient("AAPL").get_info() is info


def test_get_info_raises_ticker_error_for_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty info should be treated as an unresolved ticker."""

    patch_ticker(monkeypatch, FakeTicker("AAPL", info={}))

    with pytest.raises(TickerNotFoundError) as error:
        YFinanceClient("AAPL").get_info()

    assert error.value.source_attempted == "yfinance.Ticker.get_info"


def test_get_info_wraps_upstream_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Info failures should be wrapped without leaking upstream class names."""

    patch_ticker(monkeypatch, FakeTicker("AAPL", info_error=RuntimeError("boom")))

    with pytest.raises(TickerNotFoundError) as error:
        YFinanceClient("AAPL").get_info()

    assert error.value.symbol == "AAPL"
    assert "RuntimeError" not in str(error.value)


def test_get_fast_info_returns_raw_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast info should pass through unchanged when available."""

    fast_info = {"shares": 100}
    patch_ticker(monkeypatch, FakeTicker("AAPL", fast_info=fast_info))

    assert YFinanceClient("AAPL").get_fast_info() is fast_info


def test_get_fast_info_rejects_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty fast info should raise a market-data error."""

    patch_ticker(monkeypatch, FakeTicker("AAPL", fast_info={}))

    with pytest.raises(MarketDataUnavailableError) as error:
        YFinanceClient("AAPL").get_fast_info()

    assert error.value.source_attempted == "yfinance.Ticker.get_fast_info"


def test_get_fast_info_wraps_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast-info failures should raise a project market-data error."""

    patch_ticker(monkeypatch, FakeTicker("AAPL", fast_info_error=RuntimeError("boom")))

    with pytest.raises(MarketDataUnavailableError):
        YFinanceClient("AAPL").get_fast_info()


def test_statement_methods_pass_pretty_and_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Statement calls should use pretty labels and pass through frequency."""

    ticker = FakeTicker("AAPL")
    patch_ticker(monkeypatch, ticker)
    client = YFinanceClient("AAPL")

    assert client.get_income_statement(freq="quarterly") is ticker._income_statement
    assert client.get_balance_sheet(freq="quarterly") is ticker._balance_sheet
    assert client.get_cashflow(freq="quarterly") is ticker._cashflow
    assert ticker.income_statement_calls == [{"pretty": True, "freq": "quarterly"}]
    assert ticker.balance_sheet_calls == [{"pretty": True, "freq": "quarterly"}]
    assert ticker.cashflow_calls == [{"pretty": True, "freq": "quarterly"}]


@pytest.mark.parametrize(
    ("method_name", "statement_name"),
    [
        ("get_income_statement", "income statement"),
        ("get_balance_sheet", "balance sheet"),
        ("get_cashflow", "cash flow statement"),
    ],
)
def test_statement_methods_reject_empty_frames(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    statement_name: str,
) -> None:
    """Empty statement frames should raise statement errors."""

    ticker = FakeTicker(
        "AAPL",
        income_statement=pd.DataFrame(),
        balance_sheet=pd.DataFrame(),
        cashflow=pd.DataFrame(),
    )
    patch_ticker(monkeypatch, ticker)

    with pytest.raises(StatementUnavailableError) as error:
        getattr(YFinanceClient("AAPL"), method_name)()

    assert error.value.statement_name == statement_name
    assert error.value.symbol == "AAPL"


def test_statement_methods_wrap_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Statement exceptions should be wrapped in statement errors."""

    patch_ticker(monkeypatch, FakeTicker("AAPL", statement_error=RuntimeError("boom")))

    with pytest.raises(StatementUnavailableError) as error:
        YFinanceClient("AAPL").get_income_statement()

    assert error.value.source_attempted == "yfinance.Ticker.get_income_stmt"


def test_get_shares_full_passes_dates_and_returns_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shares history should pass dates to yfinance and return the raw series."""

    shares = pd.Series([100, 110], index=pd.to_datetime(["2025-01-01", "2026-01-01"]))
    ticker = FakeTicker("AAPL", shares=shares)
    patch_ticker(monkeypatch, ticker)
    start = date(2025, 1, 1)
    end = date(2026, 1, 1)

    assert YFinanceClient("AAPL").get_shares_full(start=start, end=end) is shares
    assert ticker.shares_calls == [{"start": start, "end": end}]


def test_get_shares_full_passes_none_and_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing shares history should preserve yfinance's None response."""

    ticker = FakeTicker("AAPL", shares=None)
    patch_ticker(monkeypatch, ticker)

    assert YFinanceClient("AAPL").get_shares_full() is None
    assert ticker.shares_calls == [{"start": None, "end": None}]


def test_get_shares_full_rejects_empty_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty shares series should raise a market-data error."""

    patch_ticker(monkeypatch, FakeTicker("AAPL", shares=pd.Series(dtype="int64")))

    with pytest.raises(MarketDataUnavailableError):
        YFinanceClient("AAPL").get_shares_full()


def test_get_history_defaults_and_custom_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """History should use Task 01 defaults and pass custom arguments through."""

    ticker = FakeTicker("AAPL")
    patch_ticker(monkeypatch, ticker)
    client = YFinanceClient("AAPL")

    assert client.get_history() is ticker._history
    assert client.get_history(period="1y", interval="1d", auto_adjust=False) is ticker._history
    assert ticker.history_calls == [
        {"period": "10y", "interval": "1mo", "auto_adjust": True},
        {"period": "1y", "interval": "1d", "auto_adjust": False},
    ]


def test_get_history_rejects_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty price history should raise a market-data error."""

    patch_ticker(monkeypatch, FakeTicker("AAPL", history=pd.DataFrame()))

    with pytest.raises(MarketDataUnavailableError) as error:
        YFinanceClient("AAPL").get_history()

    assert error.value.source_attempted == "yfinance.Ticker.history"


@pytest.mark.parametrize(
    ("instrument_type", "expected_method"),
    [
        (None, "get_all"),
        ("stock", "get_stock"),
        ("equity", "get_stock"),
        ("mutualfund", "get_mutualfund"),
        ("etf", "get_etf"),
        ("index", "get_index"),
        ("future", "get_future"),
        ("currency", "get_currency"),
        ("cryptocurrency", "get_cryptocurrency"),
    ],
)
def test_lookup_instrument_maps_supported_types(
    monkeypatch: pytest.MonkeyPatch,
    instrument_type: str | None,
    expected_method: str,
) -> None:
    """Lookup should call the yfinance method matching the requested type."""

    FakeLookup.reset()
    result = pd.DataFrame({"name": ["Apple Inc."]}, index=["AAPL"])
    FakeLookup.responses[expected_method] = result
    monkeypatch.setattr(yfinance_client_module.yf, "Lookup", FakeLookup)

    assert YFinanceClient("AAPL").lookup_instrument(" apple ", instrument_type) is result
    assert FakeLookup.calls == [("apple", expected_method)]


def test_lookup_instrument_normalizes_instrument_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lookup instrument type should be stripped and lowercased."""

    FakeLookup.reset()
    monkeypatch.setattr(yfinance_client_module.yf, "Lookup", FakeLookup)

    YFinanceClient("AAPL").lookup_instrument("apple", " ETF ")

    assert FakeLookup.calls == [("apple", "get_etf")]


def test_lookup_instrument_rejects_empty_query() -> None:
    """Empty lookup queries should raise a metric error."""

    with pytest.raises(MetricUnavailableError) as error:
        YFinanceClient("AAPL").lookup_instrument("  ")

    assert error.value.metric_name == "instrument lookup"
    assert error.value.source_attempted == "lookup query"


def test_lookup_instrument_rejects_unsupported_type() -> None:
    """Unsupported lookup types should raise a metric error."""

    with pytest.raises(MetricUnavailableError) as error:
        YFinanceClient("AAPL").lookup_instrument("apple", "bond")

    assert error.value.source_attempted == "yfinance.Lookup.bond"


def test_lookup_instrument_rejects_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty lookup results should raise a metric error."""

    FakeLookup.reset()
    FakeLookup.responses["get_stock"] = pd.DataFrame()
    monkeypatch.setattr(yfinance_client_module.yf, "Lookup", FakeLookup)

    with pytest.raises(MetricUnavailableError) as error:
        YFinanceClient("AAPL").lookup_instrument("apple", "stock")

    assert error.value.source_attempted == "yfinance.Lookup.get_stock"


def test_lookup_instrument_wraps_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lookup exceptions should be wrapped without leaking upstream class names."""

    FakeLookup.reset()
    FakeLookup.error = RuntimeError("boom")
    monkeypatch.setattr(yfinance_client_module.yf, "Lookup", FakeLookup)

    with pytest.raises(MetricUnavailableError) as error:
        YFinanceClient("AAPL").lookup_instrument("apple", "stock")

    assert error.value.source_attempted == "yfinance.Lookup.get_stock"
    assert "RuntimeError" not in str(error.value)
