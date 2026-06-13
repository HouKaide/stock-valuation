"""Tests for the equity bridge and intrinsic value per share."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from stock_valuation import DamodaranValuationProcessor
from stock_valuation.contracts import (
    CostOfDebtResult,
    CostOfEquityResult,
    Diagnostic,
    DiscountResult,
    EstimatedGrowthResult,
    FcffInputs,
    FcffResult,
    TerminalValueResult,
    ValuationAssumptions,
    WaccResult,
    to_json_safe,
)
from stock_valuation.errors import InvalidAssumptionsError, MetricUnavailableError
from stock_valuation.mapping import (
    map_minority_interest_series,
    map_non_operating_assets_series,
)

VALUATION_DATE = date(2026, 1, 31)


def _series(*values: str) -> pd.Series:
    return pd.Series(
        [Decimal(value) for value in values],
        index=pd.date_range("2024-12-31", periods=len(values), freq="YE"),
        dtype=object,
    )


@dataclass
class FakeStock:
    """Expose normalized equity bridge inputs without external calls."""

    debt: pd.Series = field(default_factory=lambda: _series("100", "120"))
    cash: pd.Series | None = field(default_factory=lambda: _series("30", "40"))
    other_assets: pd.Series | None = None
    minority: pd.Series | None = None
    shares: Decimal = Decimal("10")
    price: Decimal | None = Decimal("90")

    def normalized_ticker(self) -> str:
        """Return a deterministic ticker."""

        return "TEST"

    def valuation_currency(self) -> str:
        """Return a deterministic valuation currency."""

        return "USD"

    def debt_series(self) -> pd.Series:
        """Return normalized debt."""

        return self.debt

    def cash_series(self) -> pd.Series:
        """Return normalized cash or a typed missing-metric error."""

        if self.cash is None:
            raise MetricUnavailableError("TEST", "cash")
        return self.cash

    def non_operating_assets_series(self) -> pd.Series | None:
        """Return normalized non-operating assets when available."""

        return self.other_assets

    def minority_interest_series(self) -> pd.Series | None:
        """Return normalized minority interest when available."""

        return self.minority

    def shares_outstanding(self) -> Decimal:
        """Return normalized shares outstanding."""

        return self.shares

    def current_price(self) -> Decimal:
        """Return current price or a typed missing-metric error."""

        if self.price is None:
            raise MetricUnavailableError("TEST", "current price")
        return self.price


def _wacc(debt: str = "150") -> WaccResult:
    return WaccResult(
        market_value_of_equity=Decimal("850"),
        market_value_of_debt=Decimal(debt),
        equity_weight=Decimal("0.85"),
        debt_weight=Decimal("0.15"),
        cost_of_equity=Decimal("0.10"),
        pretax_cost_of_debt=Decimal("0.05"),
        tax_rate=Decimal("0.20"),
        wacc=Decimal("0.091"),
        calculation_steps=("WACC",),
        diagnostics=[],
    )


def _processor(
    *,
    stock: FakeStock | None = None,
    assumptions: ValuationAssumptions | None = None,
) -> DamodaranValuationProcessor:
    return DamodaranValuationProcessor(
        stock=stock or FakeStock(),  # type: ignore[arg-type]
        assumptions=assumptions or ValuationAssumptions(valuation_date=VALUATION_DATE),
    )


def test_debt_adjustment_prefers_override_then_wacc_then_book_debt() -> None:
    """Debt resolution should follow the documented precedence."""

    override_processor = _processor(
        assumptions=ValuationAssumptions(
            valuation_date=VALUATION_DATE,
            market_debt_override=Decimal("175"),
        )
    )

    assert override_processor.debt_adjustment(_wacc()) == Decimal("175")
    assert _processor().debt_adjustment(_wacc()) == Decimal("150")
    assert _processor().debt_adjustment() == Decimal("120")


def test_debt_adjustment_fails_when_book_debt_is_missing() -> None:
    """Missing book debt should remain a typed failure."""

    stock = FakeStock(debt=pd.Series(dtype=object))

    with pytest.raises(MetricUnavailableError):
        _processor(stock=stock).debt_adjustment()


def test_cash_adjustment_supports_source_zero_fallback_and_strict_failure() -> None:
    """Cash should use the latest value or the configured missing-data policy."""

    assert _processor().cash_adjustment() == Decimal("40")

    processor = _processor(stock=FakeStock(cash=None))
    assert processor.cash_adjustment() == Decimal("0")
    assert processor.diagnostics[-1].kind == "fallback"

    with pytest.raises(MetricUnavailableError):
        _processor(stock=FakeStock(cash=None)).cash_adjustment(
            allow_zero_when_missing=False
        )


def test_optional_bridge_claims_use_explicit_rows_and_zero_defaults() -> None:
    """Optional bridge claims should expose both sourced and fallback behavior."""

    stock = FakeStock(
        other_assets=_series("15", "20"),
        minority=_series("4", "5"),
    )
    processor = _processor(stock=stock)

    assert processor.non_operating_assets(Decimal("25")) == Decimal("25")
    assert processor.non_operating_assets() == Decimal("20")
    assert processor.minority_interest() == Decimal("5")

    fallback = _processor()
    assert fallback.non_operating_assets() == Decimal("0")
    assert fallback.minority_interest() == Decimal("0")
    assert [item.kind for item in fallback.diagnostics] == ["fallback", "fallback"]


def test_optional_balance_sheet_rows_are_normalized_as_positive_values() -> None:
    """Identifiable optional rows should map through normalized Stock inputs."""

    balance_sheet = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): {
                "Investments And Other Financial Assets": Decimal("-20"),
                "Minority Interest": Decimal("-5"),
            }
        }
    )

    other_assets = map_non_operating_assets_series(balance_sheet, "TEST")
    minority = map_minority_interest_series(balance_sheet, "TEST")

    assert other_assets is not None
    assert minority is not None
    assert other_assets.iloc[-1] == Decimal("20")
    assert minority.iloc[-1] == Decimal("5")


def test_equity_bridge_calculates_all_explicit_components() -> None:
    """The bridge result should preserve every adjustment and formula."""

    processor = _processor(
        stock=FakeStock(
            cash=_series("40"),
            minority=_series("5"),
        )
    )

    result = processor.equity_bridge(
        Decimal("1000"),
        wacc_result=_wacc("150"),
        non_operating_assets=Decimal("25"),
    )

    assert result.enterprise_value == Decimal("1000")
    assert result.debt == Decimal("150")
    assert result.cash == Decimal("40")
    assert result.non_operating_assets == Decimal("25")
    assert result.minority_interest == Decimal("5")
    assert result.equity_value == Decimal("910")
    assert result.calculation_steps == (
        "Equity Value = Enterprise Value - Debt + Cash "
        "+ Non-Operating Assets - Minority Interest",
    )
    assert len(result.diagnostics) == 5


@pytest.mark.parametrize("shares", [Decimal("0"), Decimal("-1")])
def test_shares_outstanding_rejects_non_positive_stock_values(
    shares: Decimal,
) -> None:
    """Normalized stock shares must be positive."""

    with pytest.raises(MetricUnavailableError):
        _processor(stock=FakeStock(shares=shares)).shares_outstanding()


def test_shares_outstanding_uses_positive_override() -> None:
    """A positive explicit share count should bypass normalized stock data."""

    processor = _processor(
        stock=FakeStock(shares=Decimal("0")),
        assumptions=ValuationAssumptions(
            valuation_date=VALUATION_DATE,
            shares_outstanding_override=Decimal("20"),
        ),
    )

    assert processor.shares_outstanding() == Decimal("20")
    assert processor.diagnostics[-1].kind == "override"


@pytest.mark.parametrize("shares", [Decimal("0"), Decimal("-1")])
def test_intrinsic_value_rejects_invalid_explicit_shares(shares: Decimal) -> None:
    """Intrinsic value division should reject invalid denominators."""

    with pytest.raises(InvalidAssumptionsError):
        _processor().intrinsic_value_per_share(Decimal("900"), shares)


def test_intrinsic_value_uses_exact_decimal_division() -> None:
    """Intrinsic value should preserve the project's Decimal policy."""

    assert _processor().intrinsic_value_per_share(
        Decimal("100"),
        Decimal("3"),
    ) == Decimal("100") / Decimal("3")


def test_current_price_is_optional_and_diagnostic() -> None:
    """Unavailable market price should not block intrinsic value."""

    assert _processor().current_price() == Decimal("90")

    processor = _processor(stock=FakeStock(price=None))
    assert processor.current_price() is None
    assert processor.diagnostics[-1].metric == "current price"


@pytest.mark.parametrize("price", [Decimal("0"), Decimal("-1")])
def test_upside_downside_rejects_non_positive_price(price: Decimal) -> None:
    """Present current prices must be positive."""

    with pytest.raises(InvalidAssumptionsError):
        _processor().upside_downside(Decimal("100"), price)


def test_upside_downside_calculates_or_omits_market_comparison() -> None:
    """Upside should use the documented formula and remain optional."""

    processor = _processor()

    assert processor.upside_downside(
        Decimal("120"),
        Decimal("100"),
    ) == Decimal("0.2")
    assert processor.upside_downside(Decimal("120"), None) is None
    assert processor.diagnostics[-1].metric == "upside/downside"


def _upstream_results() -> tuple[
    pd.DataFrame,
    FcffResult,
    EstimatedGrowthResult,
    CostOfEquityResult,
    CostOfDebtResult,
    WaccResult,
    TerminalValueResult,
    DiscountResult,
]:
    diagnostic = Diagnostic(kind="calculation", message="upstream")
    inputs = FcffInputs(
        period=pd.Timestamp("2025-12-31"),
        ebit=Decimal("100"),
        tax_rate=Decimal("0.20"),
        depreciation_amortization=Decimal("10"),
        capex=Decimal("15"),
        change_in_non_cash_working_capital=Decimal("5"),
    )
    fcff = FcffResult(
        inputs=inputs,
        nopat=Decimal("80"),
        fcff=Decimal("70"),
        calculation_steps=("FCFF",),
        diagnostics=[diagnostic],
    )
    growth = EstimatedGrowthResult(
        reinvestment_rate=Decimal("0.25"),
        return_on_capital=Decimal("0.20"),
        estimated_growth=Decimal("0.05"),
        source_method="fundamental",
        diagnostics=[diagnostic],
    )
    cost_of_equity = CostOfEquityResult(
        risk_free_rate=Decimal("0.04"),
        beta=Decimal("1.1"),
        equity_risk_premium=Decimal("0.05"),
        cost_of_equity=Decimal("0.095"),
        source_details=("test",),
        diagnostics=[diagnostic],
    )
    cost_of_debt = CostOfDebtResult(
        interest_expense=Decimal("5"),
        average_debt=Decimal("100"),
        pretax_cost_of_debt=Decimal("0.05"),
        tax_rate=Decimal("0.20"),
        after_tax_cost_of_debt=Decimal("0.04"),
        diagnostics=[diagnostic],
    )
    wacc = _wacc("150")
    terminal = TerminalValueResult(
        final_forecast_year_fcff=Decimal("100"),
        terminal_growth_rate=Decimal("0.03"),
        next_year_fcff=Decimal("103"),
        terminal_value=Decimal("1000"),
        present_value_terminal_value=Decimal("800"),
        calculation_steps=("Terminal",),
        diagnostics=[diagnostic],
    )
    discount_table = pd.DataFrame(
        [
            {
                "year": 1,
                "fcff": Decimal("100"),
                "discount_factor": Decimal("0.9"),
                "present_value": Decimal("90"),
                "cash_flow_type": "forecast",
            }
        ]
    )
    discounting = DiscountResult(
        discount_table=discount_table,
        present_value_forecast_fcffs=Decimal("200"),
        present_value_terminal_value=Decimal("800"),
        enterprise_value=Decimal("1000"),
        diagnostics=[diagnostic],
    )
    return (
        discount_table,
        fcff,
        growth,
        cost_of_equity,
        cost_of_debt,
        wacc,
        terminal,
        discounting,
    )


def test_valuation_result_includes_bridge_per_share_and_json_safe_values() -> None:
    """Complete result assembly should serialize bridge outputs consistently."""

    processor = _processor(
        stock=FakeStock(
            cash=_series("40"),
            minority=_series("5"),
            shares=Decimal("10"),
            price=Decimal("100"),
        )
    )
    (
        forecast_table,
        fcff,
        growth,
        cost_of_equity,
        cost_of_debt,
        wacc,
        terminal,
        discounting,
    ) = _upstream_results()

    result = processor.assemble_valuation_result(
        forecast_table=forecast_table,
        fcff=fcff,
        growth=growth,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        wacc=wacc,
        terminal_value=terminal,
        discounting=discounting,
        non_operating_assets=Decimal("25"),
    )
    serialized = to_json_safe(result)

    assert result.enterprise_value == Decimal("1000")
    assert result.equity_value == Decimal("910")
    assert result.intrinsic_value_per_share == Decimal("91")
    assert result.current_price == Decimal("100")
    assert result.upside_downside_pct == Decimal("-0.09")
    assert serialized["enterprise_value"] == "1000"
    assert serialized["equity_value"] == "910"
    assert serialized["intrinsic_value_per_share"] == "91"
    assert serialized["current_price"] == "100"
    assert serialized["upside_downside_pct"] == "-0.09"
    json.dumps(serialized)


def test_missing_price_serializes_optional_comparison_as_null() -> None:
    """Missing market comparison should serialize consistently as null."""

    processor = _processor(stock=FakeStock(price=None))
    assert to_json_safe(
        {
            "current_price": processor.current_price(),
            "upside_downside_pct": processor.upside_downside(Decimal("91"), None),
        }
    ) == {"current_price": None, "upside_downside_pct": None}
