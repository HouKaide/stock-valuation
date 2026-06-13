"""Contract tests for external valuation providers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from decimal import Decimal

import pytest

from stock_valuation.contracts import Diagnostic
from stock_valuation.errors import ProviderUnavailableError
from stock_valuation.providers import (
    EquityRiskPremiumProvider,
    FxRateProvider,
    MacroRateProvider,
    MarketDebtProvider,
    ProviderConfig,
    SovereignYieldCandidate,
    SovereignYieldProvider,
    SovereignYieldResult,
    TaxRateProvider,
    raise_provider_unavailable,
)
from stock_valuation.stock import TaxRateProvider as StockTaxRateProvider

VALUATION_DATE = date(2026, 1, 31)


class FakeTaxRateProvider:
    """Return a deterministic corporate tax rate."""

    def get_corporate_tax_rate(self, country: str, valuation_date: date) -> Decimal:
        """Return a canned corporate tax rate."""

        return Decimal("0.21")


class FakeEquityRiskPremiumProvider:
    """Return a deterministic equity risk premium."""

    def get_equity_risk_premium(self, country: str, valuation_date: date) -> Decimal:
        """Return a canned equity risk premium."""

        return Decimal("0.05")


class FakeMacroRateProvider:
    """Return deterministic government yields."""

    def get_long_term_government_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> Decimal:
        """Return a canned currency-matched yield."""

        return Decimal("0.025")

    def get_us_10y_treasury_yield(self, valuation_date: date) -> Decimal:
        """Return a canned US Treasury yield."""

        return Decimal("0.04")


class FakeSovereignYieldProvider:
    """Return a deterministic sovereign-yield selection."""

    def find_10y_sovereign_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> SovereignYieldResult:
        """Return a canned sovereign-yield result."""

        selected = SovereignYieldCandidate(
            symbol="DE10Y",
            currency=currency,
            country=country,
            maturity_years=Decimal("10"),
            instrument_type="government-bond",
            confidence=Decimal("0.95"),
            yield_value=Decimal("0.026"),
        )
        return SovereignYieldResult(
            selected=selected,
            provider="fake-sovereign",
            valuation_date=valuation_date,
        )


class FakeFxRateProvider:
    """Implement deterministic FX identity behavior."""

    def convert(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        valuation_date: date,
    ) -> Decimal:
        """Return the input amount for an identity conversion."""

        assert from_currency == to_currency
        return amount

    def get_rate(
        self,
        from_currency: str,
        to_currency: str,
        valuation_date: date,
    ) -> Decimal:
        """Return one for an identity conversion."""

        assert from_currency == to_currency
        return Decimal("1")


class FakeMarketDebtProvider:
    """Represent unavailable market debt."""

    def get_market_value_of_debt(
        self, ticker: str, valuation_date: date
    ) -> Decimal | None:
        """Return no market debt value."""

        return None


def test_provider_protocols_accept_structural_implementations() -> None:
    """Runtime-checkable protocols should accept matching fake providers."""

    assert isinstance(FakeTaxRateProvider(), TaxRateProvider)
    assert isinstance(FakeEquityRiskPremiumProvider(), EquityRiskPremiumProvider)
    assert isinstance(FakeMacroRateProvider(), MacroRateProvider)
    assert isinstance(FakeSovereignYieldProvider(), SovereignYieldProvider)
    assert isinstance(FakeFxRateProvider(), FxRateProvider)
    assert isinstance(FakeMarketDebtProvider(), MarketDebtProvider)
    assert StockTaxRateProvider is TaxRateProvider


def test_provider_scalar_contracts_use_decimal_values() -> None:
    """Provider scalar results should use Decimal and documented absence semantics."""

    assert FakeTaxRateProvider().get_corporate_tax_rate(
        "US", VALUATION_DATE
    ) == Decimal("0.21")
    assert FakeEquityRiskPremiumProvider().get_equity_risk_premium(
        "US", VALUATION_DATE
    ) == Decimal("0.05")
    assert FakeMacroRateProvider().get_long_term_government_yield(
        "EUR", "Germany", VALUATION_DATE
    ) == Decimal("0.025")
    assert FakeMacroRateProvider().get_us_10y_treasury_yield(VALUATION_DATE) == Decimal(
        "0.04"
    )
    assert (
        FakeMarketDebtProvider().get_market_value_of_debt("AAPL", VALUATION_DATE)
        is None
    )


def test_fx_identity_conversion_returns_input_and_unit_rate() -> None:
    """Identity FX operations should preserve exact Decimal values."""

    provider = FakeFxRateProvider()
    amount = Decimal("123.45")

    assert provider.convert(amount, "USD", "USD", VALUATION_DATE) is amount
    assert provider.get_rate("USD", "USD", VALUATION_DATE) == Decimal("1")


def test_provider_config_stores_references_without_secret_values() -> None:
    """Provider configuration should contain an environment variable name only."""

    secret = "do-not-store-this-api-key"
    config = ProviderConfig(
        name="example",
        base_url="https://provider.example",
        api_key_env_var="EXAMPLE_API_KEY",
    )

    assert asdict(config)["api_key_env_var"] == "EXAMPLE_API_KEY"
    assert secret not in repr(config)
    assert secret not in str(asdict(config))


def test_provider_error_is_distinct_actionable_and_secret_safe() -> None:
    """Provider failures should retain context without exposing their cause."""

    secret = "credential-value"
    cause = RuntimeError(f"request failed with token {secret}")

    with pytest.raises(ProviderUnavailableError) as captured:
        raise_provider_unavailable(
            "example",
            "corporate tax rate for US",
            source_attempted="tax-rate endpoint",
            fallbacks_attempted=("cached tax table",),
            api_key_env_var="EXAMPLE_API_KEY",
            cause=cause,
        )

    error = captured.value
    assert error.provider_name == "example"
    assert error.input_name == "corporate tax rate for US"
    assert error.source_attempted == "tax-rate endpoint"
    assert error.fallbacks_attempted == ("cached tax table",)
    assert error.suggested_override == (
        "Set environment variable EXAMPLE_API_KEY or supply the documented valuation override."
    )
    assert secret not in str(error)
    assert error.__cause__ is cause


def test_sovereign_result_records_selection_factors_and_rejections() -> None:
    """Sovereign selection should expose deterministic factors and diagnostics."""

    selected = SovereignYieldCandidate(
        symbol="DE10Y",
        currency="EUR",
        country="Germany",
        maturity_years=Decimal("10"),
        instrument_type="government-bond",
        confidence=Decimal("0.95"),
        yield_value=Decimal("0.026"),
    )
    rejected = SovereignYieldCandidate(
        symbol="FR10Y",
        currency="EUR",
        country="France",
        maturity_years=Decimal("10"),
        instrument_type="government-bond",
        confidence=Decimal("0.90"),
        yield_value=Decimal("0.031"),
        rejection_reason="Country mismatch.",
    )
    diagnostic = Diagnostic(
        kind="provider",
        message="Selected DE10Y; rejected FR10Y because the country did not match.",
        provider="example",
        source_attempted="instrument discovery",
    )

    result = SovereignYieldResult(
        selected=selected,
        provider="example",
        valuation_date=VALUATION_DATE,
        rejected=(rejected,),
        diagnostics=[diagnostic],
    )

    assert result.yield_value == Decimal("0.026")
    assert (
        result.selected.currency,
        result.selected.country,
        result.selected.maturity_years,
        result.selected.instrument_type,
        result.selected.confidence,
    ) == ("EUR", "Germany", Decimal("10"), "government-bond", Decimal("0.95"))
    assert result.rejected == (rejected,)
    assert result.diagnostics == [diagnostic]
