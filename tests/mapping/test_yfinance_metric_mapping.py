"""Tests for deterministic yfinance metric mapping."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from stock_valuation.errors import MetricUnavailableError, UnsupportedCurrencyError
from stock_valuation.mapping import (
    SourceMetadata,
    align_series,
    chronological_series,
    map_beta,
    map_capex_series,
    map_cash_series,
    map_company_name,
    map_current_price,
    map_debt_series,
    map_depreciation_series,
    map_ebit_series,
    map_headquarters_country,
    map_interest_expense_series,
    map_invested_capital_series,
    map_market_cap,
    map_revenue_series,
    map_shares_outstanding,
    map_trading_currency,
    map_valuation_currency,
    map_working_capital_change_series,
    normalize_currency,
    optional_metric,
    require_metric,
    select_mapping_value,
    select_statement_row,
    to_decimal,
)


@pytest.fixture(params=["USD", "eur", "GBP", "GBp", "chf"])
def supported_currency(request: pytest.FixtureRequest) -> str:
    """Return supported ISO and Yahoo unit currency examples."""

    return str(request.param)


def test_generic_mapping_helpers_record_primary_and_fallback_sources() -> None:
    """Generic selectors should expose selected source and attempted fallbacks."""

    metadata: list[SourceMetadata] = []
    value = select_mapping_value(
        {"primary": None, "fallback": 42},
        ("primary", "fallback"),
        "example",
        source_name="info",
        metadata=metadata,
    )
    statement = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [None, 10]},
        index=["Primary Row", "Fallback Row"],
    )
    row = select_statement_row(
        statement,
        ("Primary Row", "Fallback Row"),
        "row example",
        statement_name="income_statement",
        metadata=metadata,
    )

    assert value == 42
    assert row is not None
    assert metadata[0] == SourceMetadata("example", "info.fallback", ("info.primary",))
    assert metadata[1].selected_source == "income_statement.Fallback Row"


def test_required_and_optional_metric_helpers_handle_missing_values() -> None:
    """Required values should fail while optional values record unavailability."""

    metadata: list[SourceMetadata] = []

    with pytest.raises(MetricUnavailableError) as error:
        require_metric(None, "AAPL", "beta", ("info.beta", "override"))

    assert optional_metric(
        None,
        "market capitalization",
        ("fast_info.market_cap", "info.marketCap"),
        metadata=metadata,
    ) is None
    assert error.value.fallbacks_attempted == ("override",)
    assert metadata == [
        SourceMetadata(
            "market capitalization",
            None,
            ("fast_info.market_cap", "info.marketCap"),
        )
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, Decimal("123")),
        (123.5, Decimal("123.5")),
        (Decimal("0.21"), Decimal("0.21")),
        (-5, Decimal("5")),
    ],
)
def test_scalar_decimal_conversion(value: object, expected: Decimal) -> None:
    """Scalar conversion should preserve exact values and normalize signs on request."""

    assert to_decimal(value, "AAPL", "metric", absolute=value == -5) == expected


@pytest.mark.parametrize("value", [None, "invalid", float("nan"), float("inf"), True])
def test_scalar_decimal_conversion_rejects_invalid_values(value: object) -> None:
    """Required numeric metrics should reject invalid scalar values."""

    with pytest.raises(MetricUnavailableError):
        to_decimal(value, "AAPL", "metric")


def test_currency_normalization(supported_currency: str) -> None:
    """Supported currencies should normalize to uppercase ISO codes."""

    expected = "GBP" if supported_currency == "GBp" else supported_currency.upper()

    assert normalize_currency(supported_currency, "currency") == expected


@pytest.mark.parametrize("value", [None, "", "XYZ", "BTC"])
def test_currency_normalization_rejects_missing_or_unsupported_values(value: object) -> None:
    """Missing and unsupported currencies should raise typed errors."""

    with pytest.raises(UnsupportedCurrencyError):
        normalize_currency(value, "currency")


def test_company_name_mapping_covers_all_sources() -> None:
    """Company names should fall back from long name to short name to ticker."""

    metadata: list[SourceMetadata] = []

    assert map_company_name({"longName": "Apple Inc."}, "AAPL", metadata=metadata) == "Apple Inc."
    assert map_company_name({"shortName": "Apple"}, "AAPL") == "Apple"
    assert map_company_name({}, "AAPL", metadata=metadata) == "AAPL"
    assert metadata[-1].selected_source == "ticker"


def test_country_mapping_uses_source_override_and_typed_failure() -> None:
    """Country mapping should support metadata and explicit override sources."""

    metadata: list[SourceMetadata] = []

    assert map_headquarters_country({"country": "United States"}, "AAPL") == "United States"
    assert map_headquarters_country({}, "AAPL", override="US", metadata=metadata) == "US"
    assert metadata[-1].used_override is True
    with pytest.raises(MetricUnavailableError):
        map_headquarters_country({}, "AAPL")


def test_valuation_currency_mapping_covers_fallbacks_and_override() -> None:
    """Valuation currency should use info, fast info, then explicit override."""

    metadata: list[SourceMetadata] = []

    assert map_valuation_currency({"financialCurrency": "usd"}, {}) == "USD"
    assert map_valuation_currency({}, {"currency": "GBp"}) == "GBP"
    assert map_valuation_currency({}, {}, override="eur", metadata=metadata) == "EUR"
    assert metadata[-1].used_override is True
    with pytest.raises(UnsupportedCurrencyError):
        map_valuation_currency({}, {})


def test_trading_currency_mapping_covers_primary_fallback_and_missing() -> None:
    """Trading currency should prefer fast info and fall back to info."""

    assert map_trading_currency({"currency": "GBp"}, {"currency": "USD"}) == "GBP"
    assert map_trading_currency({}, {"currency": "eur"}) == "EUR"
    with pytest.raises(UnsupportedCurrencyError):
        map_trading_currency({}, {})


def test_current_price_mapping_covers_all_fallbacks() -> None:
    """Current price should fall back from fast info to info to latest close."""

    history = pd.DataFrame({"Close": [Decimal("180"), None, Decimal("181.25")]})

    assert map_current_price({"last_price": 190.25}, {}, None, "AAPL") == Decimal("190.25")
    assert map_current_price({}, {"currentPrice": 189}, None, "AAPL") == Decimal("189")
    assert map_current_price({}, {}, history, "AAPL") == Decimal("181.25")
    with pytest.raises(MetricUnavailableError):
        map_current_price({}, {}, pd.DataFrame(), "AAPL")


def test_market_cap_mapping_covers_primary_fallback_and_missing() -> None:
    """Market capitalization should map primary, fallback, and optional missing values."""

    metadata: list[SourceMetadata] = []

    assert map_market_cap({"market_cap": 1000}, {}, "AAPL") == Decimal("1000")
    assert map_market_cap({}, {"marketCap": 900}, "AAPL") == Decimal("900")
    assert map_market_cap({}, {}, "AAPL", metadata=metadata) is None
    assert metadata[-1].selected_source is None


def test_beta_mapping_covers_source_override_optional_and_required_missing() -> None:
    """Beta should support source, override, and consumer-specific missing behavior."""

    metadata: list[SourceMetadata] = []

    assert map_beta({"beta": 1.2}, "AAPL") == Decimal("1.2")
    assert map_beta({}, "AAPL", override=Decimal("1.1"), metadata=metadata) == Decimal("1.1")
    assert metadata[-1].used_override is True
    assert map_beta({}, "AAPL") is None
    with pytest.raises(MetricUnavailableError):
        map_beta({}, "AAPL", required=True)


def test_shares_mapping_covers_every_fallback() -> None:
    """Shares outstanding should preserve exact values across every fallback."""

    history = pd.Series([100, None, 120], index=pd.date_range("2024-01-01", periods=3))

    assert map_shares_outstanding({"shares": 150}, {}, None, "AAPL") == Decimal("150")
    assert map_shares_outstanding({}, {"sharesOutstanding": 140}, None, "AAPL") == Decimal("140")
    assert map_shares_outstanding({}, {}, history, "AAPL") == Decimal("120")
    assert map_shares_outstanding({}, {}, None, "AAPL", override=Decimal("130")) == Decimal("130")
    with pytest.raises(MetricUnavailableError):
        map_shares_outstanding({}, {}, None, "AAPL")


def test_income_statement_mapping_fallbacks_and_signs() -> None:
    """Income mappings should support row fallbacks, derivation, and positive interest."""

    statement = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [450, 150, -6],
            pd.Timestamp("2024-12-31"): [400, 120, -5],
        },
        index=["Operating Revenue", "Operating Income", "Interest Expense Non Operating"],
    )

    assert map_revenue_series(statement, "AAPL").tolist() == [Decimal("400"), Decimal("450")]
    assert map_ebit_series(statement, "AAPL").tolist() == [Decimal("120"), Decimal("150")]
    assert map_interest_expense_series(statement, "AAPL").tolist() == [Decimal("5"), Decimal("6")]

    derived = statement.drop(index="Operating Income")
    derived.loc["Operating Expense"] = [300, 280]
    assert map_ebit_series(derived, "AAPL").tolist() == [Decimal("120"), Decimal("150")]

    with pytest.raises(MetricUnavailableError):
        map_revenue_series(statement.drop(index="Operating Revenue"), "AAPL")


def test_cashflow_mapping_fallbacks_signs_and_derivations() -> None:
    """Cash-flow mappings should support fallbacks, signs, and balance-sheet derivation."""

    cashflow = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [-12, -24],
            pd.Timestamp("2024-12-31"): [-11, -22],
        },
        index=["Depreciation", "Purchase Of PPE"],
    )
    income = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [12]},
        index=["Reconciled Depreciation"],
    )
    balance = pd.DataFrame(
        {
            pd.Timestamp("2023-12-31"): [120, 30, 70, 10],
            pd.Timestamp("2024-12-31"): [150, 35, 80, 10],
            pd.Timestamp("2025-12-31"): [170, 40, 90, 12],
        },
        index=["Current Assets", "Cash And Cash Equivalents", "Current Liabilities", "Current Debt"],
    )

    assert map_depreciation_series(cashflow, "AAPL").tolist() == [Decimal("11"), Decimal("12")]
    assert map_capex_series(cashflow, "AAPL").tolist() == [Decimal("22"), Decimal("24")]
    assert map_working_capital_change_series(cashflow, balance, "AAPL").tolist() == [
        Decimal("15"),
        Decimal("7"),
    ]
    assert map_depreciation_series(pd.DataFrame(), "AAPL", income_statement=income).tolist() == [Decimal("12")]
    with pytest.raises(MetricUnavailableError):
        map_capex_series(pd.DataFrame(), "AAPL")


def test_balance_sheet_mapping_primary_fallbacks_and_derivations() -> None:
    """Balance mappings should support primary rows and documented derivations."""

    primary = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [100, 40, 80],
            pd.Timestamp("2024-12-31"): [95, 35, 75],
        },
        index=["Total Debt", "Cash And Cash Equivalents", "Invested Capital"],
    )
    derived = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [80, 20, 40, 200],
            pd.Timestamp("2024-12-31"): [75, 20, 35, 190],
        },
        index=["Long Term Debt", "Current Debt", "Cash And Cash Equivalents", "Stockholders Equity"],
    )

    assert map_debt_series(primary, "AAPL").tolist() == [Decimal("95"), Decimal("100")]
    assert map_cash_series(primary, "AAPL").tolist() == [Decimal("35"), Decimal("40")]
    assert map_invested_capital_series(primary, "AAPL").tolist() == [Decimal("75"), Decimal("80")]
    assert map_debt_series(derived, "AAPL").tolist() == [Decimal("95"), Decimal("100")]
    assert map_invested_capital_series(derived, "AAPL").tolist() == [Decimal("250"), Decimal("260")]
    with pytest.raises(MetricUnavailableError):
        map_debt_series(pd.DataFrame(), "AAPL")


def test_chronological_ordering_preserves_period_labels_and_values() -> None:
    """Descending yfinance periods should become chronological without relabeling."""

    periods = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
    raw = pd.Series([20, None], index=periods)

    result = chronological_series(raw, "AAPL", "metric")

    assert result.index.tolist() == [pd.Timestamp("2025-12-31")]
    assert result.tolist() == [Decimal("20")]


def test_series_alignment_preserves_common_chronological_periods() -> None:
    """Alignment should retain common periods and reject no-overlap inputs."""

    left = pd.Series(
        [Decimal("3"), Decimal("2"), Decimal("1")],
        index=pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31"]),
    )
    right = pd.Series(
        [Decimal("20"), Decimal("30")],
        index=pd.to_datetime(["2024-12-31", "2026-12-31"]),
    )

    aligned_left, aligned_right = align_series(left, right, "AAPL", "alignment")

    assert aligned_left.index.tolist() == [pd.Timestamp("2024-12-31")]
    assert aligned_right.tolist() == [Decimal("20")]
    with pytest.raises(MetricUnavailableError):
        align_series(left, right.loc[[pd.Timestamp("2026-12-31")]], "AAPL", "alignment")
