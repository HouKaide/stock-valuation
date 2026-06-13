"""Tests for typed valuation errors."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stock_valuation.errors import (
    InvalidAssumptionsError,
    MarketDataUnavailableError,
    MetricUnavailableError,
    NormalizationError,
    ProviderUnavailableError,
    StatementUnavailableError,
    StockValuationError,
    TickerNotFoundError,
    UnsupportedCurrencyError,
)
from stock_valuation.mapping import to_decimal


def test_base_error_preserves_common_safe_context() -> None:
    """Base errors should expose all shared fields without leaking credentials."""

    error = StockValuationError(
        "Provider failed with token=secret-value",
        ticker="TEST",
        metric="yield",
        provider="macro",
        source_attempted="https://user:password@example.com/rates",
        fallbacks_attempted=("cache",),
        suggested_override="Set api_key=secret-value",
        metadata={"api_key": "secret-value", "status": 503},
    )

    assert error.ticker == "TEST"
    assert error.metric == "yield"
    assert error.provider == "macro"
    assert error.fallbacks_attempted == ("cache",)
    assert "secret-value" not in str(error)
    assert "password" not in error.source_attempted
    assert error.metadata == {"api_key": "[REDACTED]", "status": 503}


def test_ticker_not_found_error_handles_empty_and_wrapped_failures() -> None:
    """Ticker errors should retain safe input and source context."""

    empty = TickerNotFoundError("   ", source_attempted="ticker input")
    wrapped = TickerNotFoundError("TEST", source_attempted="yfinance.Ticker")

    assert empty.symbol == "   "
    assert empty.ticker == "   "
    assert wrapped.source_attempted == "yfinance.Ticker"
    assert "yfinance" not in str(wrapped).lower()


@pytest.mark.parametrize(
    "statement_name",
    ["income statement", "balance sheet", "cash flow statement"],
)
def test_statement_unavailable_error_preserves_statement_context(
    statement_name: str,
) -> None:
    """Statement failures should expose ticker, statement, and fallback context."""

    error = StatementUnavailableError(
        "TEST",
        statement_name,
        source_attempted="yfinance statement method",
        fallbacks_attempted=("quarterly",),
    )

    assert error.statement_name == statement_name
    assert error.metric == statement_name
    assert error.fallbacks_attempted == ("quarterly",)


@pytest.mark.parametrize(
    "metric",
    ["beta", "shares outstanding", "EBIT", "valuation currency", "current price"],
)
def test_metric_unavailable_error_preserves_metric_context(metric: str) -> None:
    """Required metric failures should carry actionable source information."""

    error = MetricUnavailableError(
        "TEST",
        metric,
        source_attempted="normalized stock data",
        fallbacks_attempted=("override",),
        suggested_override=f"Provide {metric}.",
    )

    assert error.symbol == "TEST"
    assert error.metric_name == metric
    assert error.source_attempted == "normalized stock data"
    assert error.suggested_override == f"Provide {metric}."


def test_provider_error_redacts_runtime_and_configuration_secrets() -> None:
    """Provider failures should never expose API credentials."""

    error = ProviderUnavailableError(
        "macro",
        "government yield",
        source_attempted="https://user:password@example.com/rates",
        suggested_override="Set API_KEY=top-secret",
    )

    assert error.provider == "macro"
    assert error.input_name == "government yield"
    assert "password" not in error.source_attempted
    assert "top-secret" not in error.suggested_override
    assert "top-secret" not in repr(error)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("terminal_growth_rate", Decimal("0.10")),
        ("fcff", Decimal("0")),
        ("nopat", Decimal("-1")),
        ("invested_capital", Decimal("0")),
        ("shares_outstanding", Decimal("-10")),
    ],
)
def test_invalid_assumptions_error_preserves_safe_values(
    field_name: str,
    value: Decimal,
) -> None:
    """Invalid mathematical inputs should expose safe comparison context."""

    error = InvalidAssumptionsError(
        field_name,
        "Value is mathematically invalid.",
        value=value,
        wacc=Decimal("0.08") if field_name == "terminal_growth_rate" else None,
        suggested_override="Provide a valid override.",
    )

    assert error.metric == field_name
    assert error.value == value
    assert error.suggested_override == "Provide a valid override."


def test_invalid_assumptions_error_redacts_secret_named_values() -> None:
    """Secret-bearing invalid fields should not render their values."""

    error = InvalidAssumptionsError(
        "provider_password",
        "Invalid credential.",
        value="plain-secret",
    )

    assert error.value == "[REDACTED]"
    assert "plain-secret" not in str(error)


@pytest.mark.parametrize("currency", [None, "XYZ"])
def test_unsupported_currency_error_includes_override_guidance(
    currency: object,
) -> None:
    """Currency errors should expose raw context and override guidance."""

    error = UnsupportedCurrencyError(
        currency,
        "valuation currency",
        source_attempted="info.financialCurrency",
    )

    assert error.currency is currency
    assert error.metric == "valuation currency"
    assert "valuation_currency_override" in error.suggested_override


def test_unsupported_currency_error_supports_fx_context() -> None:
    """Currency failures should preserve FX conversion source context."""

    error = UnsupportedCurrencyError(
        "XYZ",
        "FX conversion",
        ticker="TEST",
        source_attempted="FxRateProvider",
        fallbacks_attempted=("valuation_currency_override",),
    )

    assert error.ticker == "TEST"
    assert error.fallbacks_attempted == ("valuation_currency_override",)


@pytest.mark.parametrize(
    ("source", "raw_value"),
    [
        ("numeric scalar conversion", "not-a-number"),
        ("date conversion", "not-a-date"),
        ("statement shape validation", {"unexpected": "columns"}),
    ],
)
def test_normalization_error_preserves_safe_raw_context(
    source: str,
    raw_value: object,
) -> None:
    """Normalization failures should identify conversion and safe raw context."""

    error = NormalizationError(
        "test metric",
        ticker="TEST",
        source_attempted=source,
        raw_value=raw_value,
    )

    assert isinstance(error, MetricUnavailableError)
    assert error.metric == "test metric"
    assert error.source_attempted == source
    assert error.raw_value == raw_value


def test_decimal_conversion_raises_normalization_error() -> None:
    """Invalid numeric source values should use the normalization error type."""

    with pytest.raises(NormalizationError) as captured:
        to_decimal("invalid", "TEST", "market capitalization")

    assert captured.value.ticker == "TEST"
    assert captured.value.raw_value == "invalid"


def test_market_data_error_uses_common_base_fields() -> None:
    """Existing market-data errors should participate in the common model."""

    error = MarketDataUnavailableError(
        "TEST",
        "price history",
        source_attempted="yfinance.Ticker.history",
    )

    assert isinstance(error, StockValuationError)
    assert error.metric == "price history"
    assert error.ticker == "TEST"
