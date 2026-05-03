"""Tests for the valuation domain dataclasses and package exports."""

from __future__ import annotations

from dataclasses import is_dataclass
from decimal import Decimal

import pandas as pd

from stock_valuation.domain import (
    ForecastResult,
    HistoricalFinancials,
    MarketSnapshot,
    ValuationAssumptions,
    ValuationResult,
)
from stock_valuation.domain.assumptions import ValuationAssumptions as ModuleValuationAssumptions
from stock_valuation.domain.financials import HistoricalFinancials as ModuleHistoricalFinancials
from stock_valuation.domain.forecasts import ForecastResult as ModuleForecastResult
from stock_valuation.domain.market import MarketSnapshot as ModuleMarketSnapshot
from stock_valuation.domain.valuation import ValuationResult as ModuleValuationResult


def test_domain_package_reexports_core_types() -> None:
    """The domain package should expose its public dataclass types."""

    assert MarketSnapshot is ModuleMarketSnapshot
    assert HistoricalFinancials is ModuleHistoricalFinancials
    assert ValuationAssumptions is ModuleValuationAssumptions
    assert ForecastResult is ModuleForecastResult
    assert ValuationResult is ModuleValuationResult


def test_market_snapshot_construction() -> None:
    """Market snapshots should accept Decimal-based scalar values."""

    snapshot = MarketSnapshot(
        symbol="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        current_price=Decimal("185.42"),
        market_cap=Decimal("2750000000000"),
        beta_reference=Decimal("1.12"),
        shares_outstanding=Decimal("15234000000"),
    )

    assert is_dataclass(snapshot)
    assert snapshot.current_price == Decimal("185.42")
    assert snapshot.beta_reference == Decimal("1.12")


def test_historical_financials_construction() -> None:
    """Historical financials should keep aligned pandas series fields."""

    periods = pd.to_datetime(["2022-12-31", "2023-12-31"])
    revenue = pd.Series([1000, 1100], index=periods)
    ebit = pd.Series([200, 230], index=periods)
    tax_rate = pd.Series([0.21, 0.22], index=periods)
    depreciation = pd.Series([50, 55], index=periods)
    capex = pd.Series([70, 75], index=periods)
    change_in_working_capital = pd.Series([10, 12], index=periods)
    cash = pd.Series([100, 105], index=periods)
    debt = pd.Series([300, 280], index=periods)

    financials = HistoricalFinancials(
        revenue=revenue,
        ebit=ebit,
        tax_rate=tax_rate,
        depreciation=depreciation,
        capex=capex,
        change_in_working_capital=change_in_working_capital,
        cash=cash,
        debt=debt,
        minority_interest=None,
        non_operating_assets=None,
    )

    assert is_dataclass(financials)
    assert financials.revenue is revenue
    assert financials.debt is debt


def test_valuation_assumptions_construction() -> None:
    """Valuation assumptions should keep Decimal scalar inputs and the symbol."""

    assumptions = ValuationAssumptions(
        symbol="MSFT",
        valuation_date="2026-05-03",
        forecast_years=10,
        risk_free_rate=Decimal("0.04"),
        equity_risk_premium=Decimal("0.05"),
        country_risk_premium=Decimal("0.00"),
        tax_rate=Decimal("0.21"),
        pre_tax_cost_of_debt=Decimal("0.045"),
        target_debt_to_capital=Decimal("0.20"),
        terminal_growth_rate=Decimal("0.025"),
        sales_to_capital_ratio=Decimal("1.80"),
        initial_revenue_growth_rate=Decimal("0.10"),
        target_revenue_growth_rate=Decimal("0.04"),
        initial_operating_margin=Decimal("0.32"),
        target_operating_margin=Decimal("0.35"),
        failure_probability=Decimal("0.00"),
    )

    assert is_dataclass(assumptions)
    assert assumptions.symbol == "MSFT"
    assert assumptions.target_operating_margin == Decimal("0.35")


def test_forecast_and_valuation_result_construction() -> None:
    """Forecast and valuation results should keep tabular and Decimal outputs."""

    forecast_table = pd.DataFrame(
        {
            "revenue": [1000, 1100],
            "fcff": [120, 135],
        },
        index=[1, 2],
    )

    forecast_result = ForecastResult(yearly_forecast=forecast_table)
    valuation_result = ValuationResult(
        forecast=forecast_table,
        cost_of_equity=Decimal("0.09"),
        after_tax_cost_of_debt=Decimal("0.03"),
        wacc=Decimal("0.08"),
        terminal_value=Decimal("2500"),
        enterprise_value=Decimal("3200"),
        equity_value=Decimal("3000"),
        intrinsic_value_per_share=Decimal("198.75"),
        current_price=Decimal("185.42"),
        upside_downside_pct=Decimal("0.0724"),
    )

    assert is_dataclass(forecast_result)
    assert forecast_result.yearly_forecast is forecast_table
    assert is_dataclass(valuation_result)
    assert valuation_result.forecast is forecast_table
    assert valuation_result.intrinsic_value_per_share == Decimal("198.75")
