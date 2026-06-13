"""Tests for terminal growth, terminal value, and discounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pytest

from stock_valuation import DamodaranValuationProcessor
from stock_valuation.contracts import ValuationAssumptions, to_json_safe
from stock_valuation.errors import InvalidAssumptionsError, ProviderUnavailableError
from stock_valuation.processor import discount_factor, validate_terminal_growth
from stock_valuation.providers import (
    SovereignYieldCandidate,
    SovereignYieldResult,
)

VALUATION_DATE = date(2026, 1, 31)


@dataclass
class FakeStock:
    """Expose currency and country without external calls."""

    currency: str = "USD"
    country: str = "United States"

    def normalized_ticker(self) -> str:
        """Return a deterministic ticker."""

        return "TEST"

    def valuation_currency(self) -> str:
        """Return valuation currency."""

        return self.currency

    def headquarters_country(self) -> str:
        """Return company country."""

        return self.country


@dataclass
class FakeSovereignYieldProvider:
    """Return a deterministic sovereign-yield selection."""

    yield_value: Decimal = Decimal("0.03")
    fail: bool = False
    ambiguous: bool = False
    calls: list[tuple[str, str | None, date]] = field(default_factory=list)

    def find_10y_sovereign_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> SovereignYieldResult:
        """Return a selected candidate with optional rejected alternatives."""

        self.calls.append((currency, country, valuation_date))
        if self.fail:
            raise ProviderUnavailableError(
                "fake sovereign",
                "10-year sovereign yield",
                suggested_override="Provide terminal_growth_rate_override.",
            )
        if self.ambiguous:
            raise ProviderUnavailableError(
                "fake sovereign",
                "deterministic sovereign-yield selection",
                source_attempted=f"{currency}/{country}",
                suggested_override="Provide terminal_growth_rate_override.",
            )
        selected = SovereignYieldCandidate(
            symbol=f"{currency}10Y",
            currency=currency,
            country=country,
            maturity_years=Decimal("10"),
            instrument_type="government-bond",
            confidence=Decimal("0.95"),
            yield_value=self.yield_value,
        )
        rejected = SovereignYieldCandidate(
            symbol=f"{currency}ALT",
            currency=currency,
            country=country,
            maturity_years=Decimal("10"),
            instrument_type="government-bond",
            confidence=Decimal("0.80"),
            yield_value=self.yield_value + Decimal("0.001"),
            rejection_reason="Lower provider confidence.",
        )
        return SovereignYieldResult(
            selected=selected,
            provider="fake sovereign",
            valuation_date=valuation_date,
            rejected=(rejected,),
        )


def _processor(
    *,
    stock: FakeStock | None = None,
    assumptions: ValuationAssumptions | None = None,
    provider: FakeSovereignYieldProvider | None = None,
) -> DamodaranValuationProcessor:
    return DamodaranValuationProcessor(
        stock=stock or FakeStock(),  # type: ignore[arg-type]
        assumptions=assumptions or ValuationAssumptions(valuation_date=VALUATION_DATE),
        sovereign_yield_provider=provider,
    )


def test_terminal_growth_override_prevents_provider_call() -> None:
    """A terminal-growth override should bypass sovereign discovery."""

    provider = FakeSovereignYieldProvider(fail=True)
    processor = _processor(
        assumptions=ValuationAssumptions(
            valuation_date=VALUATION_DATE,
            terminal_growth_rate_override=Decimal("0.025"),
        ),
        provider=provider,
    )

    result = processor.terminal_growth_rate()

    assert result.selected_instrument == "override"
    assert result.yield_value == Decimal("0.025")
    assert result.provider == "user override"
    assert result.valuation_date == VALUATION_DATE
    assert result.diagnostics[0].kind == "override"
    assert provider.calls == []


@pytest.mark.parametrize(
    ("currency", "country"),
    [
        ("USD", "United States"),
        ("EUR", "Germany"),
        ("GBP", "United Kingdom"),
        ("CHF", "Switzerland"),
    ],
)
def test_terminal_growth_passes_any_currency_to_provider(
    currency: str,
    country: str,
) -> None:
    """Sovereign discovery should pass currency and country unchanged."""

    provider = FakeSovereignYieldProvider()
    processor = _processor(
        assumptions=ValuationAssumptions(
            valuation_date=VALUATION_DATE,
            valuation_currency_override=currency,
            company_country_override=country,
        ),
        provider=provider,
    )

    result = processor.terminal_growth_rate()

    assert provider.calls == [(currency, country, VALUATION_DATE)]
    assert result.selected_instrument == f"{currency}10Y"
    assert result.yield_value == Decimal("0.03")
    assert result.provider == "fake sovereign"
    assert result.fallbacks_attempted == (f"{currency}ALT",)
    assert result.diagnostics[-1].kind == "provider"
    assert f"{currency}ALT" in result.diagnostics[-1].message


def test_terminal_growth_preserves_deterministic_candidate_diagnostics() -> None:
    """Selected and rejected candidates should remain observable."""

    result = _processor(provider=FakeSovereignYieldProvider()).terminal_growth_rate()

    diagnostic = result.diagnostics[-1]
    assert diagnostic.source_attempted == "USD10Y"
    assert diagnostic.fallbacks_attempted == ("USDALT",)
    assert "Selected USD10Y" in diagnostic.message
    assert "rejected alternatives: USDALT" in diagnostic.message


def test_terminal_growth_raises_for_ambiguous_or_missing_yield() -> None:
    """Ambiguous selection and unusable yields should raise typed errors."""

    with pytest.raises(ProviderUnavailableError):
        _processor(
            provider=FakeSovereignYieldProvider(ambiguous=True)
        ).terminal_growth_rate()

    with pytest.raises(ProviderUnavailableError) as captured:
        _processor(
            provider=FakeSovereignYieldProvider(yield_value=Decimal("NaN"))
        ).terminal_growth_rate()

    assert captured.value.suggested_override == "Provide terminal_growth_rate_override."


def test_terminal_growth_requires_provider_without_override() -> None:
    """Missing sovereign provider should produce actionable setup guidance."""

    with pytest.raises(ProviderUnavailableError) as captured:
        _processor().terminal_growth_rate()

    assert "terminal_growth_rate_override" in captured.value.suggested_override


def test_validate_terminal_growth_accepts_growth_below_wacc() -> None:
    """Terminal growth below WACC should pass validation."""

    validate_terminal_growth(Decimal("0.08"), Decimal("0.03"))


@pytest.mark.parametrize(
    "growth",
    [Decimal("0.08"), Decimal("0.09")],
)
def test_validate_terminal_growth_rejects_equal_or_higher_growth(
    growth: Decimal,
) -> None:
    """Terminal growth equal to or above WACC should fail with context."""

    with pytest.raises(InvalidAssumptionsError) as captured:
        validate_terminal_growth(Decimal("0.08"), growth)

    error = captured.value
    assert error.field_name == "terminal_growth_rate"
    assert error.wacc == Decimal("0.08")
    assert error.terminal_growth_rate == growth
    assert "terminal_growth_rate_override" in error.suggested_override


@pytest.mark.parametrize(
    ("wacc", "growth", "field_name"),
    [
        (Decimal("NaN"), Decimal("0.03"), "wacc"),
        (Decimal("0.08"), Decimal("Infinity"), "terminal_growth_rate"),
    ],
)
def test_validate_terminal_growth_rejects_non_finite_values(
    wacc: Decimal,
    growth: Decimal,
    field_name: str,
) -> None:
    """Non-finite terminal inputs should fail before comparison."""

    with pytest.raises(InvalidAssumptionsError) as captured:
        validate_terminal_growth(wacc, growth)

    assert captured.value.field_name == field_name


def test_terminal_value_calculates_all_formula_fields() -> None:
    """Terminal value should calculate next FCFF, value, and present value."""

    result = _processor().terminal_value(
        Decimal("100"),
        Decimal("0.10"),
        Decimal("0.03"),
        5,
    )
    expected_next_fcff = Decimal("103")
    expected_terminal_value = expected_next_fcff / Decimal("0.07")
    expected_present_value = expected_terminal_value / (Decimal("1.10") ** 5)

    assert result.final_forecast_year_fcff == Decimal("100")
    assert result.terminal_growth_rate == Decimal("0.03")
    assert result.next_year_fcff == expected_next_fcff
    assert result.terminal_value == expected_terminal_value
    assert result.present_value_terminal_value == expected_present_value
    assert len(result.calculation_steps) == 3
    assert "Validated terminal growth" in result.diagnostics[0].message


def test_terminal_value_rejects_invalid_growth_and_forecast_years() -> None:
    """Terminal value should enforce growth and forecast-year boundaries."""

    with pytest.raises(InvalidAssumptionsError):
        _processor().terminal_value(
            Decimal("100"),
            Decimal("0.08"),
            Decimal("0.08"),
            5,
        )
    with pytest.raises(InvalidAssumptionsError):
        _processor().terminal_value(
            Decimal("100"),
            Decimal("0.08"),
            Decimal("0.03"),
            0,
        )


@pytest.mark.parametrize("year", [1, 2, 5])
def test_discount_factor_uses_decimal_exponent(year: int) -> None:
    """Discount factor should match exact Decimal exponent behavior."""

    assert discount_factor(Decimal("0.10"), year) == Decimal("1") / (
        Decimal("1.10") ** year
    )


@pytest.mark.parametrize(
    ("wacc", "year"),
    [
        (Decimal("-1"), 1),
        (Decimal("-2"), 1),
        (Decimal("NaN"), 1),
        (Decimal("0.1"), 0),
    ],
)
def test_discount_factor_rejects_invalid_inputs(
    wacc: Decimal,
    year: int,
) -> None:
    """Invalid WACC bases and years should fail."""

    with pytest.raises(InvalidAssumptionsError):
        discount_factor(wacc, year)


def test_forecast_discount_table_preserves_order() -> None:
    """Forecast discount table should preserve input order and formulas."""

    forecasts = [Decimal("100"), Decimal("110"), Decimal("120")]
    table = _processor().forecast_fcff_discount_table(
        forecasts,
        Decimal("0.10"),
    )

    assert table["year"].tolist() == [1, 2, 3]
    assert table["fcff"].tolist() == forecasts
    assert table["cash_flow_type"].tolist() == ["forecast"] * 3
    assert table.loc[1, "discount_factor"] == Decimal("1") / Decimal("1.10") ** 2
    assert table.loc[2, "present_value"] == (Decimal("120") / Decimal("1.10") ** 3)


def test_forecast_discount_table_supports_one_year() -> None:
    """A one-year forecast should produce one deterministic row."""

    table = _processor().forecast_fcff_discount_table(
        [Decimal("100")],
        Decimal("0.10"),
    )

    assert len(table) == 1
    assert table.iloc[0].to_dict() == {
        "year": 1,
        "fcff": Decimal("100"),
        "discount_factor": Decimal("1") / Decimal("1.10"),
        "present_value": Decimal("100") / Decimal("1.10"),
        "cash_flow_type": "forecast",
    }


def test_discount_to_today_builds_terminal_row_and_enterprise_value() -> None:
    """Discount result should sum forecast and terminal present values."""

    forecasts = [Decimal("100"), Decimal("110")]
    terminal_value = Decimal("1000")
    result = _processor().discount_to_today(
        forecasts,
        Decimal("0.10"),
        terminal_value,
    )
    expected_forecast_pv = (
        Decimal("100") / Decimal("1.10") + Decimal("110") / Decimal("1.10") ** 2
    )
    expected_terminal_pv = terminal_value / Decimal("1.10") ** 2

    assert len(result.discount_table) == 3
    assert result.discount_table["cash_flow_type"].tolist() == [
        "forecast",
        "forecast",
        "terminal",
    ]
    assert result.present_value_forecast_fcffs == expected_forecast_pv
    assert result.present_value_terminal_value == expected_terminal_pv
    assert result.enterprise_value == expected_forecast_pv + expected_terminal_pv
    assert "2 forecast rows and one terminal row" in result.diagnostics[0].message


def test_terminal_and_discount_results_are_json_safe_and_deterministic() -> None:
    """Terminal contracts should serialize Decimals and rows deterministically."""

    processor = _processor(provider=FakeSovereignYieldProvider())
    growth = processor.terminal_growth_rate()
    terminal = processor.terminal_value(
        Decimal("100"),
        Decimal("0.10"),
        growth.yield_value,
        2,
    )
    discount = processor.discount_to_today(
        [Decimal("90"), Decimal("100")],
        Decimal("0.10"),
        terminal.terminal_value,
    )

    growth_json = to_json_safe(growth)
    terminal_json = to_json_safe(terminal)
    discount_json = to_json_safe(discount)

    assert growth_json["yield_value"] == "0.03"
    assert terminal_json["next_year_fcff"] == "103.00"
    assert [row["cash_flow_type"] for row in discount_json["discount_table"]] == [
        "forecast",
        "forecast",
        "terminal",
    ]
    assert discount_json["discount_table"][0]["year"] == 1
    assert discount_json["diagnostics"]
