"""Tests for the valuation-ready stock model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from stock_valuation import Stock as RootStock
from stock_valuation.errors import MetricUnavailableError
from stock_valuation.stock import Stock


@dataclass
class FakeTaxRateProvider:
    """Fake tax-rate provider with call tracking."""

    tax_rate: Decimal
    calls: list[tuple[str, date]]

    def get_corporate_tax_rate(self, country: str, valuation_date: date) -> Decimal:
        """Return a canned tax rate."""

        self.calls.append((country, valuation_date))
        return self.tax_rate


class FakeYFinanceClient:
    """Fake yfinance client exposing the raw surfaces Stock needs."""

    def __init__(
        self,
        *,
        ticker: str = "AAPL",
        info: dict[str, Any] | None = None,
        fast_info: dict[str, Any] | None = None,
        income_statement: pd.DataFrame | None = None,
        balance_sheet: pd.DataFrame | None = None,
        cashflow: pd.DataFrame | None = None,
        shares: pd.Series | None = None,
        history: pd.DataFrame | None = None,
    ) -> None:
        self.ticker = ticker
        self.info = info if info is not None else self.default_info()
        self.fast_info = fast_info if fast_info is not None else self.default_fast_info()
        self.income_statement = income_statement if income_statement is not None else self.default_income_statement()
        self.balance_sheet = balance_sheet if balance_sheet is not None else self.default_balance_sheet()
        self.cashflow = cashflow if cashflow is not None else self.default_cashflow()
        self.shares = shares
        self.history = history if history is not None else pd.DataFrame({"Close": [Decimal("190.50")]})
        self.statement_calls: list[tuple[str, str]] = []

    def normalized_ticker(self) -> str:
        """Return normalized ticker."""

        return self.ticker.strip().upper()

    def get_info(self) -> dict[str, Any]:
        """Return fake ticker metadata."""

        return self.info

    def get_fast_info(self) -> dict[str, Any]:
        """Return fake fast metadata."""

        return self.fast_info

    def get_income_statement(self, freq: str = "yearly") -> pd.DataFrame:
        """Return fake annual income statement."""

        self.statement_calls.append(("income", freq))
        return self.income_statement

    def get_balance_sheet(self, freq: str = "yearly") -> pd.DataFrame:
        """Return fake annual balance sheet."""

        self.statement_calls.append(("balance", freq))
        return self.balance_sheet

    def get_cashflow(self, freq: str = "yearly") -> pd.DataFrame:
        """Return fake annual cash flow statement."""

        self.statement_calls.append(("cashflow", freq))
        return self.cashflow

    def get_shares_full(self) -> pd.Series | None:
        """Return fake shares history."""

        return self.shares

    def get_history(self, *, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
        """Return fake price history."""

        return self.history

    @staticmethod
    def default_info() -> dict[str, Any]:
        """Return default metadata."""

        return {
            "longName": "Apple Inc.",
            "country": "United States",
            "financialCurrency": "USD",
            "currency": "USD",
            "currentPrice": 189.1,
            "marketCap": 3_000_000_000_000,
            "beta": 1.2,
            "sharesOutstanding": 15_000_000_000,
        }

    @staticmethod
    def default_fast_info() -> dict[str, Any]:
        """Return default fast metadata."""

        return {
            "currency": "USD",
            "last_price": 190.25,
            "market_cap": 3_100_000_000_000,
            "shares": 15_500_000_000,
        }

    @staticmethod
    def default_income_statement() -> pd.DataFrame:
        """Return default income statement rows."""

        return pd.DataFrame(
            {
                pd.Timestamp("2023-12-31"): [380, 110, -4],
                pd.Timestamp("2024-12-31"): [400, 120, -5],
                pd.Timestamp("2025-12-31"): [450, 150, -6],
            },
            index=["Total Revenue", "EBIT", "Interest Expense"],
        )

    @staticmethod
    def default_balance_sheet() -> pd.DataFrame:
        """Return default balance sheet rows."""

        return pd.DataFrame(
            {
                pd.Timestamp("2023-12-31"): [90, 30, 180, 70],
                pd.Timestamp("2024-12-31"): [95, 35, 190, 75],
                pd.Timestamp("2025-12-31"): [100, 40, 200, 80],
            },
            index=["Total Debt", "Cash And Cash Equivalents", "Stockholders Equity", "Invested Capital"],
        )

    @staticmethod
    def default_cashflow() -> pd.DataFrame:
        """Return default cash flow rows."""

        return pd.DataFrame(
            {
                pd.Timestamp("2023-12-31"): [10, -20, -3],
                pd.Timestamp("2024-12-31"): [11, -22, -4],
                pd.Timestamp("2025-12-31"): [12, -24, -5],
            },
            index=["Depreciation And Amortization", "Capital Expenditure", "Change In Working Capital"],
        )


def test_root_package_exports_stock_model() -> None:
    """The stock model should be available from the package root."""

    assert RootStock is Stock


def test_stock_creates_default_client_when_not_injected() -> None:
    """Stock should create a ticker-bound yfinance client by default."""

    stock = Stock(" aapl ")

    assert stock.yfinance_client.normalized_ticker() == "AAPL"


def test_metadata_methods_use_documented_yfinance_fallbacks() -> None:
    """Metadata methods should normalize yfinance info and fast-info values."""

    client = FakeYFinanceClient(info={"shortName": "Apple", "country": "US", "financialCurrency": "USD", "beta": 1.1})
    stock = Stock("AAPL", yfinance_client=client)

    assert stock.company_name() == "Apple"
    assert stock.headquarters_country() == "US"
    assert stock.valuation_currency() == "USD"
    assert stock.trading_currency() == "USD"
    assert stock.current_price() == Decimal("190.25")
    assert stock.market_cap() == Decimal("3100000000000")
    assert stock.beta() == Decimal("1.1")
    assert stock.shares_outstanding() == Decimal("15500000000")


def test_market_data_falls_back_to_info_history_and_share_history() -> None:
    """Scalar values should use the documented fallback chain."""

    shares = pd.Series([100, 120], index=pd.to_datetime(["2024-01-01", "2025-01-01"]))
    client = FakeYFinanceClient(
        fast_info={"currency": "USD"},
        info={"country": "US", "financialCurrency": "USD", "currentPrice": 189, "marketCap": 1000},
        shares=shares,
    )
    stock = Stock("AAPL", yfinance_client=client)

    assert stock.current_price() == Decimal("189")
    assert stock.market_cap() == Decimal("1000")
    assert stock.shares_outstanding() == Decimal("120")


def test_current_price_falls_back_to_latest_history_close() -> None:
    """A latest close should be used when metadata does not expose a price."""

    client = FakeYFinanceClient(
        fast_info={"currency": "USD"},
        info={"country": "US", "financialCurrency": "USD"},
        history=pd.DataFrame({"Close": [180.5, 181.25]}),
    )

    assert Stock("AAPL", yfinance_client=client).current_price() == Decimal("181.25")


def test_currency_mismatch_is_recorded_in_diagnostics() -> None:
    """Trading and financial currency differences should be observable."""

    client = FakeYFinanceClient(
        fast_info={"currency": "EUR", "last_price": 10, "shares": 100},
        info={"country": "Germany", "financialCurrency": "USD"},
    )
    stock = Stock("SAP", yfinance_client=client)

    assert stock.trading_currency() == "EUR"
    assert stock.diagnostics == ["Trading currency EUR differs from valuation currency USD."]


def test_statement_accessors_are_cached_and_use_yearly_frequency() -> None:
    """Annual statement accessors should cache raw client frames."""

    client = FakeYFinanceClient()
    stock = Stock("AAPL", yfinance_client=client)

    assert stock.annual_income_statement() is stock.annual_income_statement()
    assert stock.annual_balance_sheet() is stock.annual_balance_sheet()
    assert stock.annual_cashflow() is stock.annual_cashflow()
    assert client.statement_calls == [("income", "yearly"), ("balance", "yearly"), ("cashflow", "yearly")]


def test_statement_metric_series_are_chronological_and_normalized() -> None:
    """Statement rows should become chronological Decimal series."""

    stock = Stock("AAPL", yfinance_client=FakeYFinanceClient())

    assert stock.revenue_series().tolist() == [Decimal("380"), Decimal("400"), Decimal("450")]
    assert stock.ebit_series().tolist() == [Decimal("110"), Decimal("120"), Decimal("150")]
    assert stock.interest_expense_series().tolist() == [Decimal("4"), Decimal("5"), Decimal("6")]
    assert stock.depreciation_series().tolist() == [Decimal("10"), Decimal("11"), Decimal("12")]
    assert stock.capex_series().tolist() == [Decimal("20"), Decimal("22"), Decimal("24")]
    assert stock.change_in_non_cash_working_capital_series().tolist() == [
        Decimal("3"),
        Decimal("4"),
        Decimal("5"),
    ]
    assert stock.debt_series().tolist() == [Decimal("90"), Decimal("95"), Decimal("100")]
    assert stock.cash_series().tolist() == [Decimal("30"), Decimal("35"), Decimal("40")]


def test_ebit_can_be_derived_from_revenue_and_operating_expense() -> None:
    """EBIT should fall back to revenue less operating expenses."""

    income_statement = pd.DataFrame(
        {
            pd.Timestamp("2024-12-31"): [400, 280],
            pd.Timestamp("2025-12-31"): [450, 300],
        },
        index=["Total Revenue", "Operating Expense"],
    )

    stock = Stock("AAPL", yfinance_client=FakeYFinanceClient(income_statement=income_statement))

    assert stock.ebit_series().tolist() == [Decimal("120"), Decimal("150")]


def test_working_capital_can_be_derived_from_balance_sheet_rows() -> None:
    """Working-capital changes should fall back to balance-sheet derivation."""

    balance_sheet = pd.DataFrame(
        {
            pd.Timestamp("2023-12-31"): [120, 30, 70, 10],
            pd.Timestamp("2024-12-31"): [150, 35, 80, 10],
            pd.Timestamp("2025-12-31"): [170, 40, 90, 12],
        },
        index=["Current Assets", "Cash And Cash Equivalents", "Current Liabilities", "Current Debt"],
    )
    cashflow = FakeYFinanceClient.default_cashflow().drop(index="Change In Working Capital")
    stock = Stock("AAPL", yfinance_client=FakeYFinanceClient(balance_sheet=balance_sheet, cashflow=cashflow))

    assert stock.change_in_non_cash_working_capital_series().tolist() == [Decimal("15"), Decimal("7")]


def test_latest_fcff_inputs_use_tax_provider_and_latest_common_period() -> None:
    """FCFF inputs should bundle the latest aligned normalized values."""

    provider = FakeTaxRateProvider(Decimal("0.21"), [])
    stock = Stock("AAPL", yfinance_client=FakeYFinanceClient(), tax_rate_provider=provider)

    inputs = stock.latest_fcff_inputs()

    assert inputs.period == pd.Timestamp("2025-12-31")
    assert inputs.ebit == Decimal("150")
    assert inputs.tax_rate == Decimal("0.21")
    assert inputs.nopat == Decimal("118.50")
    assert inputs.depreciation_amortization == Decimal("12")
    assert inputs.capex == Decimal("24")
    assert inputs.change_in_non_cash_working_capital == Decimal("5")
    assert provider.calls[0][0] == "United States"


def test_invested_capital_return_on_capital_and_reinvestment_rates() -> None:
    """Aggregate capital metrics should align statement series."""

    stock = Stock("AAPL", yfinance_client=FakeYFinanceClient(), tax_rate=Decimal("0.20"))

    assert stock.invested_capital_series().tolist() == [Decimal("70"), Decimal("75"), Decimal("80")]
    assert stock.return_on_capital_series().tolist() == [
        Decimal("96.0") / Decimal("70"),
        Decimal("120.0") / Decimal("75"),
    ]
    assert stock.reinvestment_rate_series().tolist() == [
        Decimal("13.0") / Decimal("88.0"),
        Decimal("15.0") / Decimal("96.0"),
        Decimal("17.0") / Decimal("120.0"),
    ]


def test_missing_required_metric_raises_typed_error() -> None:
    """Missing required rows should raise project metric errors."""

    stock = Stock(
        "AAPL",
        yfinance_client=FakeYFinanceClient(
            income_statement=FakeYFinanceClient.default_income_statement().drop(index="Total Revenue")
        ),
    )

    with pytest.raises(MetricUnavailableError) as error:
        stock.revenue_series()

    assert error.value.metric_name == "revenue"
    assert error.value.symbol == "AAPL"
