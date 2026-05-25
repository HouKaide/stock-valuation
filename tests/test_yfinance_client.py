"""Tests for the stock valuation yfinance client."""

from datetime import date

import pandas as pd
import pytest

import stock_valuation.yfinance_client as yfinance_client_module
from stock_valuation import (
    MetricUnavailableError,
    StatementUnavailableError,
    TickerNotFoundError,
    YFinanceClient,
)


class FakeYFinanceError(Exception):
    """Fake yfinance exception used to verify exception wrapping."""


class FakeTicker:
    """Fake ticker with the yfinance methods used by the client."""

    def __init__(self) -> None:
        self.info: dict[str, object] = {"longName": "Example Corp"}
        self.fast_info: dict[str, object] = {"currency": "USD"}
        self.income_statement = pd.DataFrame({"2024": [1]}, index=["EBIT"])
        self.balance_sheet = pd.DataFrame({"2024": [1]}, index=["Total Debt"])
        self.cashflow = pd.DataFrame({"2024": [1]}, index=["Capital Expenditure"])
        self.shares_full: pd.Series | None = pd.Series([100], index=pd.to_datetime(["2024-01-01"]))
        self.history_frame = pd.DataFrame({"Close": [10]}, index=pd.to_datetime(["2024-01-01"]))
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_info(self) -> dict[str, object]:
        """Return fake company information."""
        self.calls.append(("get_info", {}))
        return self.info

    def get_fast_info(self) -> dict[str, object]:
        """Return fake fast market information."""
        self.calls.append(("get_fast_info", {}))
        return self.fast_info

    def get_income_stmt(self, *, pretty: bool, freq: str) -> pd.DataFrame:
        """Return a fake income statement."""
        self.calls.append(("get_income_stmt", {"pretty": pretty, "freq": freq}))
        return self.income_statement

    def get_balance_sheet(self, *, pretty: bool, freq: str) -> pd.DataFrame:
        """Return a fake balance sheet."""
        self.calls.append(("get_balance_sheet", {"pretty": pretty, "freq": freq}))
        return self.balance_sheet

    def get_cashflow(self, *, pretty: bool, freq: str) -> pd.DataFrame:
        """Return a fake cash-flow statement."""
        self.calls.append(("get_cashflow", {"pretty": pretty, "freq": freq}))
        return self.cashflow

    def get_shares_full(self, *, start: date | None, end: date | None) -> pd.Series | None:
        """Return fake shares history."""
        self.calls.append(("get_shares_full", {"start": start, "end": end}))
        return self.shares_full

    def history(self, *, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        """Return fake price history."""
        self.calls.append(("history", {"period": period, "interval": interval, "auto_adjust": auto_adjust}))
        return self.history_frame


class RaisingTicker(FakeTicker):
    """Fake ticker that raises from every data method."""

    def get_info(self) -> dict[str, object]:
        """Raise a fake yfinance exception."""
        raise FakeYFinanceError("hidden yfinance detail")

    def get_fast_info(self) -> dict[str, object]:
        """Raise a fake yfinance exception."""
        raise FakeYFinanceError("hidden yfinance detail")

    def get_income_stmt(self, *, pretty: bool, freq: str) -> pd.DataFrame:
        """Raise a fake yfinance exception."""
        raise FakeYFinanceError("hidden yfinance detail")

    def get_balance_sheet(self, *, pretty: bool, freq: str) -> pd.DataFrame:
        """Raise a fake yfinance exception."""
        raise FakeYFinanceError("hidden yfinance detail")

    def get_cashflow(self, *, pretty: bool, freq: str) -> pd.DataFrame:
        """Raise a fake yfinance exception."""
        raise FakeYFinanceError("hidden yfinance detail")

    def get_shares_full(self, *, start: date | None, end: date | None) -> pd.Series | None:
        """Raise a fake yfinance exception."""
        raise FakeYFinanceError("hidden yfinance detail")

    def history(self, *, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        """Raise a fake yfinance exception."""
        raise FakeYFinanceError("hidden yfinance detail")


def test_errors_store_diagnostic_attributes() -> None:
    """Confirm typed errors expose diagnostic context without yfinance internals."""
    ticker_error = TickerNotFoundError(
        "Ticker is required.",
        ticker="",
        metric_name="ticker",
        source="YFinanceClient.normalized_ticker",
        suggested_override="ticker",
    )
    statement_error = StatementUnavailableError(
        "income_statement is unavailable.",
        ticker="AAPL",
        statement_name="income_statement",
        source="yfinance.Ticker.get_income_stmt",
        fallbacks=("none",),
    )
    metric_error = MetricUnavailableError(
        "info is unavailable.",
        ticker="AAPL",
        metric_name="info",
        source="yfinance.Ticker.get_info",
        fallbacks=("fast_info",),
    )

    assert ticker_error.ticker == ""
    assert ticker_error.metric_name == "ticker"
    assert ticker_error.suggested_override == "ticker"
    assert statement_error.statement_name == "income_statement"
    assert statement_error.metric_name == "income_statement"
    assert statement_error.fallbacks == ("none",)
    assert metric_error.metric_name == "info"
    assert metric_error.source == "yfinance.Ticker.get_info"
    assert "FakeYFinanceError" not in str(metric_error)


def test_normalized_ticker_uppercases_and_strips_whitespace() -> None:
    """Confirm ticker normalization trims whitespace and uppercases."""
    assert YFinanceClient(" aapl ").normalized_ticker() == "AAPL"


@pytest.mark.parametrize("ticker", ["", " ", "\n"])
def test_normalized_ticker_rejects_empty_values(ticker: str) -> None:
    """Confirm empty ticker values raise a typed error."""
    with pytest.raises(TickerNotFoundError):
        YFinanceClient(ticker).normalized_ticker()


def test_normalization_error_does_not_create_yfinance_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm invalid ticker input fails before constructing yfinance objects."""
    calls: list[str] = []

    def fake_ticker(symbol: str) -> FakeTicker:
        calls.append(symbol)
        return FakeTicker()

    monkeypatch.setattr(yfinance_client_module.yf, "Ticker", fake_ticker)

    with pytest.raises(TickerNotFoundError):
        YFinanceClient(" ").get_ticker()

    assert calls == []


def test_get_ticker_constructs_normalized_ticker_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm the yfinance ticker is cached per client instance."""
    calls: list[str] = []
    fake = FakeTicker()

    def fake_ticker(symbol: str) -> FakeTicker:
        calls.append(symbol)
        return fake

    monkeypatch.setattr(yfinance_client_module.yf, "Ticker", fake_ticker)
    client = YFinanceClient(" msft ")

    assert client.get_ticker() is fake
    assert client.get_ticker() is fake
    assert calls == ["MSFT"]
    assert "_ticker" not in repr(client)


def test_get_ticker_wraps_constructor_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm yfinance ticker construction failures become project errors."""

    def fake_ticker(symbol: str) -> FakeTicker:
        raise FakeYFinanceError(f"{symbol} failed")

    monkeypatch.setattr(yfinance_client_module.yf, "Ticker", fake_ticker)

    with pytest.raises(TickerNotFoundError) as exc_info:
        YFinanceClient("msft").get_ticker()

    assert exc_info.value.ticker == "MSFT"
    assert "FakeYFinanceError" not in str(exc_info.value)


def test_get_info_returns_non_empty_dictionary() -> None:
    """Confirm company info is returned unchanged."""
    fake = FakeTicker()
    client = YFinanceClient("AAPL")
    client._ticker = fake

    assert client.get_info() is fake.info


def test_get_info_rejects_empty_dictionary() -> None:
    """Confirm empty company info raises a metric error."""
    fake = FakeTicker()
    fake.info = {}
    client = YFinanceClient("AAPL")
    client._ticker = fake

    with pytest.raises(MetricUnavailableError) as exc_info:
        client.get_info()

    assert exc_info.value.metric_name == "info"
    assert exc_info.value.source == "yfinance.Ticker.get_info"


def test_get_fast_info_returns_non_empty_mapping() -> None:
    """Confirm fast info is returned unchanged."""
    fake = FakeTicker()
    client = YFinanceClient("AAPL")
    client._ticker = fake

    assert client.get_fast_info() is fake.fast_info


def test_get_fast_info_rejects_empty_mapping() -> None:
    """Confirm empty fast info raises a metric error."""
    fake = FakeTicker()
    fake.fast_info = {}
    client = YFinanceClient("AAPL")
    client._ticker = fake

    with pytest.raises(MetricUnavailableError) as exc_info:
        client.get_fast_info()

    assert exc_info.value.metric_name == "fast_info"
    assert exc_info.value.source == "yfinance.Ticker.get_fast_info"


@pytest.mark.parametrize("method_name", ["get_info", "get_fast_info"])
def test_metadata_methods_wrap_yfinance_exceptions(method_name: str) -> None:
    """Confirm metadata yfinance exceptions become metric errors."""
    client = YFinanceClient("AAPL")
    client._ticker = RaisingTicker()

    with pytest.raises(MetricUnavailableError) as exc_info:
        getattr(client, method_name)()

    assert "FakeYFinanceError" not in str(exc_info.value)


def test_statement_methods_return_raw_dataframes_and_pass_frequencies() -> None:
    """Confirm statement methods call yfinance with pretty row labels and requested frequencies."""
    fake = FakeTicker()
    client = YFinanceClient("AAPL")
    client._ticker = fake

    assert client.get_income_statement() is fake.income_statement
    assert client.get_income_statement(freq="quarterly") is fake.income_statement
    assert client.get_income_statement(freq="trailing") is fake.income_statement
    assert client.get_balance_sheet() is fake.balance_sheet
    assert client.get_balance_sheet(freq="quarterly") is fake.balance_sheet
    assert client.get_cashflow() is fake.cashflow
    assert client.get_cashflow(freq="quarterly") is fake.cashflow
    assert ("get_income_stmt", {"pretty": True, "freq": "yearly"}) in fake.calls
    assert ("get_income_stmt", {"pretty": True, "freq": "quarterly"}) in fake.calls
    assert ("get_income_stmt", {"pretty": True, "freq": "trailing"}) in fake.calls
    assert ("get_balance_sheet", {"pretty": True, "freq": "yearly"}) in fake.calls
    assert ("get_balance_sheet", {"pretty": True, "freq": "quarterly"}) in fake.calls
    assert ("get_cashflow", {"pretty": True, "freq": "yearly"}) in fake.calls
    assert ("get_cashflow", {"pretty": True, "freq": "quarterly"}) in fake.calls


@pytest.mark.parametrize(
    ("method_name", "statement_name", "source"),
    [
        ("get_income_statement", "income_statement", "yfinance.Ticker.get_income_stmt"),
        ("get_balance_sheet", "balance_sheet", "yfinance.Ticker.get_balance_sheet"),
        ("get_cashflow", "cashflow", "yfinance.Ticker.get_cashflow"),
    ],
)
def test_statement_methods_reject_empty_dataframes(method_name: str, statement_name: str, source: str) -> None:
    """Confirm empty statement responses raise statement errors."""
    fake = FakeTicker()
    fake.income_statement = pd.DataFrame()
    fake.balance_sheet = pd.DataFrame()
    fake.cashflow = pd.DataFrame()
    client = YFinanceClient("AAPL")
    client._ticker = fake

    with pytest.raises(StatementUnavailableError) as exc_info:
        getattr(client, method_name)()

    assert exc_info.value.statement_name == statement_name
    assert exc_info.value.source == source


@pytest.mark.parametrize("method_name", ["get_income_statement", "get_balance_sheet", "get_cashflow"])
def test_statement_methods_wrap_yfinance_exceptions(method_name: str) -> None:
    """Confirm statement yfinance exceptions become statement errors."""
    client = YFinanceClient("AAPL")
    client._ticker = RaisingTicker()

    with pytest.raises(StatementUnavailableError) as exc_info:
        getattr(client, method_name)()

    assert "FakeYFinanceError" not in str(exc_info.value)


def test_get_shares_full_passes_dates_and_returns_series() -> None:
    """Confirm shares history date arguments pass through to yfinance."""
    fake = FakeTicker()
    client = YFinanceClient("AAPL")
    client._ticker = fake
    start = date(2023, 1, 1)
    end = date(2024, 1, 1)

    assert client.get_shares_full(start=start, end=end) is fake.shares_full
    assert ("get_shares_full", {"start": start, "end": end}) in fake.calls


def test_get_shares_full_returns_none_unchanged() -> None:
    """Confirm missing yfinance shares history is preserved as None."""
    fake = FakeTicker()
    fake.shares_full = None
    client = YFinanceClient("AAPL")
    client._ticker = fake

    assert client.get_shares_full() is None


def test_get_shares_full_rejects_empty_series() -> None:
    """Confirm empty shares history raises a metric error."""
    fake = FakeTicker()
    fake.shares_full = pd.Series(dtype="int64")
    client = YFinanceClient("AAPL")
    client._ticker = fake

    with pytest.raises(MetricUnavailableError) as exc_info:
        client.get_shares_full()

    assert exc_info.value.metric_name == "shares_full"
    assert exc_info.value.source == "yfinance.Ticker.get_shares_full"


def test_get_shares_full_wraps_yfinance_exceptions() -> None:
    """Confirm shares history yfinance exceptions become metric errors."""
    client = YFinanceClient("AAPL")
    client._ticker = RaisingTicker()

    with pytest.raises(MetricUnavailableError):
        client.get_shares_full()


def test_get_history_uses_defaults_and_returns_dataframe() -> None:
    """Confirm price history defaults and raw return value."""
    fake = FakeTicker()
    client = YFinanceClient("AAPL")
    client._ticker = fake

    assert client.get_history() is fake.history_frame
    assert ("history", {"period": "10y", "interval": "1mo", "auto_adjust": True}) in fake.calls


def test_get_history_passes_custom_arguments() -> None:
    """Confirm custom price history arguments pass through."""
    fake = FakeTicker()
    client = YFinanceClient("AAPL")
    client._ticker = fake

    assert client.get_history(period="1y", interval="1d", auto_adjust=False) is fake.history_frame
    assert ("history", {"period": "1y", "interval": "1d", "auto_adjust": False}) in fake.calls


def test_get_history_rejects_empty_dataframe() -> None:
    """Confirm empty price history raises a metric error."""
    fake = FakeTicker()
    fake.history_frame = pd.DataFrame()
    client = YFinanceClient("AAPL")
    client._ticker = fake

    with pytest.raises(MetricUnavailableError) as exc_info:
        client.get_history()

    assert exc_info.value.metric_name == "history"
    assert exc_info.value.source == "yfinance.Ticker.history"


def test_get_history_wraps_yfinance_exceptions() -> None:
    """Confirm price history yfinance exceptions become metric errors."""
    client = YFinanceClient("AAPL")
    client._ticker = RaisingTicker()

    with pytest.raises(MetricUnavailableError):
        client.get_history()


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
        (" Stock ", "get_stock"),
    ],
)
def test_lookup_instrument_calls_mapped_method(
    monkeypatch: pytest.MonkeyPatch,
    instrument_type: str | None,
    expected_method: str,
) -> None:
    """Confirm lookup dispatches each supported instrument type to yfinance."""
    calls: list[tuple[str, str]] = []
    expected = pd.DataFrame({"name": ["US Treasury"]}, index=pd.Index(["^TNX"], name="symbol"))

    class FakeLookup:
        """Fake lookup object with all yfinance lookup methods."""

        def __init__(self, query: str) -> None:
            self.query = query

        def _return(self, method_name: str) -> pd.DataFrame:
            calls.append((self.query, method_name))
            return expected

        def get_all(self) -> pd.DataFrame:
            return self._return("get_all")

        def get_stock(self) -> pd.DataFrame:
            return self._return("get_stock")

        def get_mutualfund(self) -> pd.DataFrame:
            return self._return("get_mutualfund")

        def get_etf(self) -> pd.DataFrame:
            return self._return("get_etf")

        def get_index(self) -> pd.DataFrame:
            return self._return("get_index")

        def get_future(self) -> pd.DataFrame:
            return self._return("get_future")

        def get_currency(self) -> pd.DataFrame:
            return self._return("get_currency")

        def get_cryptocurrency(self) -> pd.DataFrame:
            return self._return("get_cryptocurrency")

    monkeypatch.setattr(yfinance_client_module.yf, "Lookup", FakeLookup)

    result = YFinanceClient("AAPL").lookup_instrument("  treasury  ", instrument_type=instrument_type)

    assert result is expected
    assert calls == [("treasury", expected_method)]


def test_lookup_instrument_rejects_empty_query() -> None:
    """Confirm blank lookup queries fail before yfinance lookup construction."""
    with pytest.raises(MetricUnavailableError) as exc_info:
        YFinanceClient("AAPL").lookup_instrument("  ")

    assert exc_info.value.metric_name == "query"


def test_lookup_instrument_rejects_unsupported_instrument_type() -> None:
    """Confirm unsupported lookup instrument types raise metric errors."""
    with pytest.raises(MetricUnavailableError) as exc_info:
        YFinanceClient("AAPL").lookup_instrument("treasury", instrument_type="bond")

    assert exc_info.value.metric_name == "instrument_type"


def test_lookup_instrument_rejects_empty_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm empty lookup results raise metric errors."""

    class FakeLookup:
        """Fake lookup that returns no candidates."""

        def __init__(self, query: str) -> None:
            self.query = query

        def get_all(self) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(yfinance_client_module.yf, "Lookup", FakeLookup)

    with pytest.raises(MetricUnavailableError) as exc_info:
        YFinanceClient("AAPL").lookup_instrument("treasury")

    assert exc_info.value.metric_name == "lookup_instrument"
    assert exc_info.value.source == "yfinance.Lookup.get_all"


def test_lookup_instrument_wraps_yfinance_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm yfinance lookup exceptions become metric errors."""

    class FakeLookup:
        """Fake lookup that raises from the selected method."""

        def __init__(self, query: str) -> None:
            self.query = query

        def get_all(self) -> pd.DataFrame:
            raise FakeYFinanceError("hidden yfinance detail")

    monkeypatch.setattr(yfinance_client_module.yf, "Lookup", FakeLookup)

    with pytest.raises(MetricUnavailableError) as exc_info:
        YFinanceClient("AAPL").lookup_instrument("treasury")

    assert "FakeYFinanceError" not in str(exc_info.value)


def test_public_api_exports_work() -> None:
    """Confirm public imports work from the package root."""
    assert YFinanceClient("AAPL").normalized_ticker() == "AAPL"
