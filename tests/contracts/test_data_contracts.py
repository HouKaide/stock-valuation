"""Tests for shared valuation data contracts."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from stock_valuation import Diagnostic as RootDiagnostic
from stock_valuation import FcffInputs as RootFcffInputs
from stock_valuation.contracts import (
    CostOfDebtResult,
    CostOfEquityResult,
    Diagnostic,
    DiscountResult,
    EquityBridgeResult,
    EstimatedGrowthResult,
    FcffInputs,
    FcffResult,
    GrowthRegressionResult,
    TerminalGrowthResult,
    TerminalValueResult,
    ValuationAssumptions,
    ValuationResult,
    WaccResult,
    to_json_safe,
)
from stock_valuation.errors import InvalidAssumptionsError


def test_public_contracts_are_exported() -> None:
    """Contracts should be importable from the package root and contracts package."""

    assert RootDiagnostic is Diagnostic
    assert RootFcffInputs is FcffInputs


def test_valuation_assumptions_defaults_and_overrides() -> None:
    """Assumptions should keep defaults and explicit override values."""

    assumptions = ValuationAssumptions(
        valuation_date=date(2026, 1, 31),
        company_country_override="United States",
        valuation_currency_override="USD",
        beta_override=Decimal("1.15"),
        tax_rate_override=Decimal("0.21"),
        equity_risk_premium_override=Decimal("0.055"),
        risk_free_rate_override=Decimal("0.04"),
        terminal_growth_rate_override=Decimal("0.025"),
        shares_outstanding_override=Decimal("15000000000"),
        market_debt_override=Decimal("100000000"),
    )

    assert assumptions.forecast_years == 5
    assert assumptions.tax_rate_override == Decimal("0.21")
    assert assumptions.terminal_growth_rate_override == Decimal("0.025")
    assert assumptions.market_debt_override == Decimal("100000000")


def test_valuation_assumptions_allow_missing_optional_overrides() -> None:
    """Optional overrides should not be required."""

    assumptions = ValuationAssumptions(valuation_date=date(2026, 1, 31))

    assert assumptions.company_country_override is None
    assert assumptions.tax_rate_override is None


def test_valuation_assumptions_validate_boundaries() -> None:
    """Invalid forecast years, rate notation, and non-finite decimals should fail."""

    with pytest.raises(InvalidAssumptionsError) as forecast_error:
        ValuationAssumptions(valuation_date=date(2026, 1, 31), forecast_years=0)

    with pytest.raises(InvalidAssumptionsError) as rate_error:
        ValuationAssumptions(valuation_date=date(2026, 1, 31), tax_rate_override=Decimal("21"))

    with pytest.raises(InvalidAssumptionsError) as finite_error:
        ValuationAssumptions(valuation_date=date(2026, 1, 31), beta_override=Decimal("NaN"))

    assert forecast_error.value.field_name == "forecast_years"
    assert rate_error.value.field_name == "tax_rate_override"
    assert finite_error.value.field_name == "beta_override"


def test_diagnostic_covers_source_fallback_provider_override_and_failure() -> None:
    """Diagnostic should represent the source, fallback, provider, override, and failure fields."""

    diagnostic = Diagnostic(
        kind="failure",
        message="Beta unavailable.",
        ticker="AAPL",
        metric="beta",
        provider="yfinance",
        source_attempted="yfinance.Ticker.get_info.beta",
        fallbacks_attempted=("beta_override",),
        suggested_override="Provide beta_override.",
    )

    assert diagnostic.kind == "failure"
    assert diagnostic.provider == "yfinance"
    assert diagnostic.fallbacks_attempted == ("beta_override",)
    assert diagnostic.suggested_override == "Provide beta_override."


def test_fcff_inputs_and_result_are_manual_calculation_contracts() -> None:
    """FCFF contracts should carry all values needed to reproduce the calculation."""

    diagnostic = Diagnostic(kind="source", message="Resolved from yearly statements.")
    inputs = FcffInputs(
        period=pd.Timestamp("2025-12-31"),
        ebit=Decimal("150"),
        tax_rate=Decimal("0.21"),
        depreciation_amortization=Decimal("12"),
        capex=Decimal("24"),
        change_in_non_cash_working_capital=Decimal("5"),
        diagnostics=[diagnostic],
    )
    result = FcffResult(
        inputs=inputs,
        nopat=Decimal("118.50"),
        fcff=Decimal("101.50"),
        calculation_steps=("NOPAT = EBIT * (1 - tax_rate)", "FCFF = NOPAT + D&A - CAPEX - NWC"),
        diagnostics=[diagnostic],
    )

    assert inputs.nopat == Decimal("118.50")
    assert result.inputs is inputs
    assert result.fcff == Decimal("101.50")


def test_growth_result_contracts_accept_decimal_rates() -> None:
    """Growth contracts should hold regression and estimated-growth outputs."""

    diagnostic = Diagnostic(kind="source", message="Calculated from revenue series.")
    regression = GrowthRegressionResult(
        slope=Decimal("0.01"),
        intercept=Decimal("0.03"),
        sample_size=5,
        predicted_next_year_growth=Decimal("0.08"),
        diagnostics=[diagnostic],
    )
    estimated = EstimatedGrowthResult(
        reinvestment_rate=Decimal("0.20"),
        return_on_capital=Decimal("0.15"),
        estimated_growth=Decimal("0.03"),
        source_method="reinvestment_rate_x_return_on_capital",
        diagnostics=[diagnostic],
    )

    assert regression.predicted_next_year_growth == Decimal("0.08")
    assert estimated.estimated_growth == Decimal("0.03")


def test_cost_of_capital_result_contracts_accept_decimal_rates() -> None:
    """Cost-of-capital contracts should hold decimal rates and calculation steps."""

    diagnostic = Diagnostic(kind="provider", message="Provider returned rate.")
    cost_of_equity = CostOfEquityResult(
        risk_free_rate=Decimal("0.04"),
        beta=Decimal("1.10"),
        equity_risk_premium=Decimal("0.055"),
        cost_of_equity=Decimal("0.1005"),
        source_details=("risk-free provider", "ERP provider", "yfinance beta"),
        diagnostics=[diagnostic],
    )
    cost_of_debt = CostOfDebtResult(
        interest_expense=Decimal("5"),
        average_debt=Decimal("100"),
        pretax_cost_of_debt=Decimal("0.05"),
        tax_rate=Decimal("0.21"),
        after_tax_cost_of_debt=Decimal("0.0395"),
        diagnostics=[diagnostic],
    )
    wacc = WaccResult(
        market_value_of_equity=Decimal("900"),
        market_value_of_debt=Decimal("100"),
        equity_weight=Decimal("0.90"),
        debt_weight=Decimal("0.10"),
        cost_of_equity=cost_of_equity.cost_of_equity,
        pretax_cost_of_debt=cost_of_debt.pretax_cost_of_debt,
        tax_rate=Decimal("0.21"),
        wacc=Decimal("0.0944"),
        calculation_steps=("WACC = E/V * Re + D/V * Rd * (1 - T)",),
        diagnostics=[diagnostic],
    )

    assert cost_of_equity.cost_of_equity == Decimal("0.1005")
    assert cost_of_debt.after_tax_cost_of_debt == Decimal("0.0395")
    assert wacc.wacc == Decimal("0.0944")


def test_terminal_discount_and_equity_bridge_contracts() -> None:
    """Terminal, discounting, and equity bridge contracts should expose all adjustments."""

    diagnostic = Diagnostic(kind="fallback", message="Used book debt fallback.")
    terminal_growth = TerminalGrowthResult(
        selected_instrument="US10Y",
        yield_value=Decimal("0.035"),
        valuation_date=date(2026, 1, 31),
        provider="sovereign-yield-provider",
        fallbacks_attempted=("US Treasury fallback",),
        diagnostics=[diagnostic],
    )
    terminal_value = TerminalValueResult(
        final_forecast_year_fcff=Decimal("100"),
        terminal_growth_rate=terminal_growth.yield_value,
        next_year_fcff=Decimal("103.5"),
        terminal_value=Decimal("1725"),
        present_value_terminal_value=Decimal("1300"),
        calculation_steps=("FCF_n+1 = FCF_n * (1 + g)",),
        diagnostics=[diagnostic],
    )
    discount = DiscountResult(
        discount_table=pd.DataFrame(
            [{"year": 1, "fcff": Decimal("100"), "discount_factor": Decimal("0.95"), "present_value": Decimal("95")}]
        ),
        present_value_forecast_fcffs=Decimal("95"),
        present_value_terminal_value=terminal_value.present_value_terminal_value,
        enterprise_value=Decimal("1395"),
        diagnostics=[diagnostic],
    )
    bridge = EquityBridgeResult(
        enterprise_value=discount.enterprise_value,
        debt=Decimal("100"),
        cash=Decimal("50"),
        non_operating_assets=Decimal("10"),
        minority_interest=Decimal("5"),
        equity_value=Decimal("1350"),
        diagnostics=[diagnostic],
    )

    assert terminal_growth.yield_value == Decimal("0.035")
    assert discount.enterprise_value == Decimal("1395")
    assert bridge.equity_value == Decimal("1350")


def test_complete_valuation_result_constructs_without_calculation() -> None:
    """A complete valuation result should be constructible from result contracts."""

    diagnostic = Diagnostic(kind="source", message="Complete result assembled.")
    fcff_inputs = FcffInputs(
        period=pd.Timestamp("2025-12-31"),
        ebit=Decimal("150"),
        tax_rate=Decimal("0.21"),
        depreciation_amortization=Decimal("12"),
        capex=Decimal("24"),
        change_in_non_cash_working_capital=Decimal("5"),
    )
    fcff = FcffResult(fcff_inputs, Decimal("118.50"), Decimal("101.50"), ("FCFF calculation",), [diagnostic])
    growth = EstimatedGrowthResult(Decimal("0.20"), Decimal("0.15"), Decimal("0.03"), "default", [diagnostic])
    cost_of_equity = CostOfEquityResult(
        Decimal("0.04"), Decimal("1.1"), Decimal("0.055"), Decimal("0.1005"), ("sources",), [diagnostic]
    )
    cost_of_debt = CostOfDebtResult(
        Decimal("5"), Decimal("100"), Decimal("0.05"), Decimal("0.21"), Decimal("0.0395"), [diagnostic]
    )
    wacc = WaccResult(
        Decimal("900"),
        Decimal("100"),
        Decimal("0.90"),
        Decimal("0.10"),
        Decimal("0.1005"),
        Decimal("0.05"),
        Decimal("0.21"),
        Decimal("0.0944"),
        ("WACC calculation",),
        [diagnostic],
    )
    terminal_value = TerminalValueResult(
        Decimal("100"), Decimal("0.035"), Decimal("103.5"), Decimal("1725"), Decimal("1300"), ("TV",), [diagnostic]
    )
    discounting = DiscountResult(
        pd.DataFrame([{"year": 1, "fcff": Decimal("100")}]),
        Decimal("95"),
        Decimal("1300"),
        Decimal("1395"),
        [diagnostic],
    )

    result = ValuationResult(
        ticker="AAPL",
        valuation_date=date(2026, 1, 31),
        valuation_currency="USD",
        forecast_table=pd.DataFrame([{"year": 1, "revenue": Decimal("450")}]),
        fcff=fcff,
        growth=growth,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        wacc=wacc,
        terminal_value=terminal_value,
        discounting=discounting,
        enterprise_value=Decimal("1395"),
        equity_value=Decimal("1350"),
        intrinsic_value_per_share=Decimal("90"),
        current_price=Decimal("100"),
        upside_downside_pct=Decimal("-0.10"),
        diagnostics=[diagnostic],
    )

    assert result.ticker == "AAPL"
    assert result.intrinsic_value_per_share == Decimal("90")


def test_json_safe_serialization_uses_stable_decimal_date_and_diagnostic_values() -> None:
    """Serialization should convert Decimals, dates, pandas objects, and diagnostics explicitly."""

    diagnostic = Diagnostic(kind="override", message="Used terminal growth override.")
    assumptions = ValuationAssumptions(
        valuation_date=date(2026, 1, 31),
        tax_rate_override=Decimal("0.21"),
    )
    frame = pd.DataFrame([{"valuation_date": date(2026, 1, 31), "value": Decimal("12.34")}])

    serialized = to_json_safe(
        {
            "assumptions": assumptions,
            "frame": frame,
            "diagnostics": [diagnostic],
        }
    )

    assert serialized["assumptions"]["valuation_date"] == "2026-01-31"
    assert serialized["assumptions"]["tax_rate_override"] == "0.21"
    assert serialized["frame"] == [{"valuation_date": "2026-01-31", "value": "12.34"}]
    assert serialized["diagnostics"][0]["kind"] == "override"
