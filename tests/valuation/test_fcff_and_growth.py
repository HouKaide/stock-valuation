"""Tests for FCFF and growth calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from stock_valuation import DamodaranValuationProcessor
from stock_valuation.contracts import (
    EstimatedGrowthResult,
    FcffInputs,
    FcffResult,
    GrowthRegressionResult,
    to_json_safe,
)
from stock_valuation.errors import InvalidAssumptionsError, MetricUnavailableError
from stock_valuation.processor import calculate_nopat, validate_positive_operating_base


@dataclass
class FakeStock:
    """Expose normalized stock metrics without external calls."""

    fcff_inputs: FcffInputs = field(
        default_factory=lambda: FcffInputs(
            period=pd.Timestamp("2025-12-31"),
            ebit=Decimal("120"),
            tax_rate=Decimal("0.20"),
            depreciation_amortization=Decimal("10"),
            capex=Decimal("30"),
            change_in_non_cash_working_capital=Decimal("5"),
        )
    )
    revenue: pd.Series = field(
        default_factory=lambda: pd.Series(
            [Decimal("132"), Decimal("100"), Decimal("110")],
            index=pd.to_datetime(["2025-12-31", "2023-12-31", "2024-12-31"]),
            dtype=object,
        )
    )
    ebit: pd.Series = field(
        default_factory=lambda: pd.Series(
            [Decimal("80"), Decimal("100"), Decimal("120")],
            index=pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
            dtype=object,
        )
    )
    invested_capital: pd.Series = field(
        default_factory=lambda: pd.Series(
            [Decimal("200"), Decimal("250"), Decimal("300")],
            index=pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
            dtype=object,
        )
    )

    def normalized_ticker(self) -> str:
        """Return a deterministic ticker."""

        return "TEST"

    def latest_fcff_inputs(self) -> FcffInputs:
        """Return normalized FCFF inputs."""

        return self.fcff_inputs

    def revenue_series(self) -> pd.Series:
        """Return normalized annual revenue."""

        return self.revenue

    def ebit_series(self) -> pd.Series:
        """Return normalized annual EBIT."""

        return self.ebit

    def invested_capital_series(self) -> pd.Series:
        """Return normalized annual invested capital."""

        return self.invested_capital


def _processor(stock: FakeStock | None = None) -> DamodaranValuationProcessor:
    return DamodaranValuationProcessor(stock or FakeStock())  # type: ignore[arg-type]


def test_calculate_nopat_uses_decimal_formula() -> None:
    """NOPAT should equal EBIT multiplied by one minus the tax rate."""

    assert calculate_nopat(Decimal("100"), Decimal("0.21")) == Decimal("79")


@pytest.mark.parametrize("ebit", [Decimal("0"), Decimal("-1")])
def test_calculate_nopat_rejects_non_positive_results(ebit: Decimal) -> None:
    """Zero and negative NOPAT should fail with a typed error."""

    with pytest.raises(InvalidAssumptionsError) as captured:
        calculate_nopat(ebit, Decimal("0"))

    assert captured.value.field_name == "nopat"
    assert captured.value.value == ebit


@pytest.mark.parametrize(
    ("ebit", "tax_rate", "field_name"),
    [
        (None, Decimal("0.20"), "ebit"),
        (Decimal("NaN"), Decimal("0.20"), "ebit"),
        (Decimal("100"), None, "tax_rate"),
        (Decimal("100"), Decimal("Infinity"), "tax_rate"),
    ],
)
def test_calculate_nopat_rejects_malformed_inputs(
    ebit: Any,
    tax_rate: Any,
    field_name: str,
) -> None:
    """Missing and non-finite inputs should fail before calculation."""

    with pytest.raises(InvalidAssumptionsError) as captured:
        calculate_nopat(ebit, tax_rate)

    assert captured.value.field_name == field_name


def test_calculate_fcff_uses_positive_capex_and_working_capital_outflows() -> None:
    """CAPEX and working-capital increases should reduce FCFF."""

    result = _processor().calculate_fcff()

    assert result.nopat == Decimal("96")
    assert result.fcff == Decimal("71")
    assert result.fcff == Decimal("96") + Decimal("10") - Decimal("30") - Decimal("5")
    assert result.diagnostics[0].metric == "fcff"
    assert len(result.calculation_steps) == 2


def test_calculate_fcff_reflects_each_reinvestment_outflow() -> None:
    """Increasing either CAPEX or working capital should lower FCFF."""

    base = FakeStock()
    base_result = _processor(base).calculate_fcff()
    capex_result = _processor(
        FakeStock(
            fcff_inputs=FcffInputs(
                **{
                    **base.fcff_inputs.__dict__,
                    "capex": Decimal("31"),
                }
            )
        )
    ).calculate_fcff()
    working_capital_result = _processor(
        FakeStock(
            fcff_inputs=FcffInputs(
                **{
                    **base.fcff_inputs.__dict__,
                    "change_in_non_cash_working_capital": Decimal("6"),
                }
            )
        )
    ).calculate_fcff()

    assert capex_result.fcff == base_result.fcff - Decimal("1")
    assert working_capital_result.fcff == base_result.fcff - Decimal("1")


def test_calculate_fcff_rejects_non_positive_fcff_with_context() -> None:
    """Invalid FCFF should report metric, period, value, and guidance."""

    stock = FakeStock(
        fcff_inputs=FcffInputs(
            period="2025",
            ebit=Decimal("100"),
            tax_rate=Decimal("0"),
            depreciation_amortization=Decimal("0"),
            capex=Decimal("100"),
            change_in_non_cash_working_capital=Decimal("0"),
        )
    )

    with pytest.raises(InvalidAssumptionsError) as captured:
        _processor(stock).calculate_fcff()

    error = captured.value
    assert error.field_name == "fcff"
    assert error.period == "2025"
    assert error.value == Decimal("0")
    assert error.suggested_override is not None


def test_historical_revenue_growth_is_chronological_and_uses_all_years() -> None:
    """Revenue growth should sort periods before calculating adjacent changes."""

    growth = _processor().historical_revenue_growth()

    assert list(growth.index) == list(pd.to_datetime(["2024-12-31", "2025-12-31"]))
    assert list(growth) == [Decimal("0.1"), Decimal("0.2")]


def test_historical_revenue_growth_reports_sample_and_dropped_periods() -> None:
    """Revenue diagnostics should identify source and invalid periods."""

    stock = FakeStock(
        revenue=pd.Series(
            [Decimal("100"), Decimal("0"), Decimal("121")],
            index=pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
            dtype=object,
        )
    )
    processor = _processor(stock)

    growth = processor.historical_revenue_growth()

    assert list(growth) == [Decimal("0.21")]
    message = processor.diagnostics[-1].message
    assert "1 revenue growth observations" in message
    assert "2023-12-31" in message
    assert "2025-12-31" in message
    assert "Dropped invalid periods: 2024-12-31" in message


@pytest.mark.parametrize(
    "revenue",
    [
        pd.Series([], dtype=object),
        pd.Series([Decimal("100")], index=[pd.Timestamp("2025-12-31")], dtype=object),
    ],
)
def test_historical_revenue_growth_requires_two_valid_points(
    revenue: pd.Series,
) -> None:
    """Fewer than two valid revenue observations should fail."""

    with pytest.raises(MetricUnavailableError):
        _processor(FakeStock(revenue=revenue)).historical_revenue_growth()


def test_revenue_growth_regression_is_deterministic() -> None:
    """Regression should return exact Decimal coefficients and prediction."""

    result = _processor().forecast_revenue_growth_regression()

    assert result.slope == Decimal("0.1")
    assert result.intercept == Decimal("0.1")
    assert result.sample_size == 2
    assert result.predicted_next_year_growth == Decimal("0.3")
    assert result.diagnostics[0].metric == "historical revenue growth"
    assert result.diagnostics[-1].metric == "revenue growth regression"


def test_revenue_growth_regression_requires_two_growth_points() -> None:
    """A single growth observation cannot support a regression."""

    stock = FakeStock(
        revenue=pd.Series(
            [Decimal("100"), Decimal("110")],
            index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
            dtype=object,
        )
    )

    with pytest.raises(MetricUnavailableError) as captured:
        _processor(stock).forecast_revenue_growth_regression()

    assert captured.value.metric_name == "revenue growth regression"


def test_next_year_regression_helper_delegates_to_result() -> None:
    """The convenience helper should return the regression result field."""

    processor = _processor()

    assert (
        processor.next_year_revenue_growth_from_regression()
        == processor.forecast_revenue_growth_regression().predicted_next_year_growth
    )


def test_reinvestment_rate_uses_latest_normalized_inputs() -> None:
    """Reinvestment rate should use CAPEX minus D&A plus working capital."""

    assert _processor().reinvestment_rate() == Decimal("25") / Decimal("96")


@pytest.mark.parametrize("ebit", [Decimal("0"), Decimal("-10")])
def test_reinvestment_rate_requires_positive_nopat(ebit: Decimal) -> None:
    """Reinvestment rate should reject zero and negative NOPAT."""

    stock = FakeStock(
        fcff_inputs=FcffInputs(
            period="2025",
            ebit=ebit,
            tax_rate=Decimal("0"),
            depreciation_amortization=Decimal("10"),
            capex=Decimal("20"),
            change_in_non_cash_working_capital=Decimal("5"),
        )
    )

    with pytest.raises(InvalidAssumptionsError) as captured:
        _processor(stock).reinvestment_rate()

    assert captured.value.field_name == "nopat"


def test_return_on_capital_uses_prior_period_invested_capital() -> None:
    """Return on capital should divide NOPAT by prior-period capital."""

    assert _processor().return_on_capital() == Decimal("96") / Decimal("250")


@pytest.mark.parametrize("capital", [Decimal("0"), Decimal("-1")])
def test_return_on_capital_rejects_invalid_prior_capital(capital: Decimal) -> None:
    """Non-positive prior-period capital should fail with source context."""

    stock = FakeStock(
        invested_capital=pd.Series(
            [Decimal("200"), capital, Decimal("300")],
            index=pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]),
            dtype=object,
        )
    )

    with pytest.raises(InvalidAssumptionsError) as captured:
        _processor(stock).return_on_capital()

    assert captured.value.field_name == "invested_capital"
    assert captured.value.period == pd.Timestamp("2025-12-31")
    assert captured.value.value == capital


def test_estimated_growth_is_default_aligned_growth_method() -> None:
    """Estimated growth should multiply latest aligned reinvestment and ROC."""

    result = _processor().estimated_growth()

    assert result.reinvestment_rate == Decimal("25") / Decimal("96")
    assert result.return_on_capital == Decimal("96") / Decimal("250")
    assert result.estimated_growth == Decimal("0.1")
    assert result.source_method == "reinvestment_rate_times_return_on_capital"
    assert "default growth" in result.diagnostics[0].message


@pytest.mark.parametrize(
    ("metric_name", "value"),
    [
        ("fcff", Decimal("0")),
        ("nopat", Decimal("-1")),
        ("invested_capital", Decimal("NaN")),
    ],
)
def test_operating_base_validator_covers_all_invalid_metrics(
    metric_name: str,
    value: Decimal,
) -> None:
    """The shared validator should reject every invalid operating-base metric."""

    with pytest.raises(InvalidAssumptionsError) as captured:
        validate_positive_operating_base(metric_name, value, period="2025")

    assert captured.value.field_name == metric_name
    assert captured.value.period == "2025"
    if value.is_nan():
        assert captured.value.value.is_nan()
    else:
        assert captured.value.value == value


def test_fcff_and_growth_results_are_json_safe() -> None:
    """Result contracts should serialize Decimals as strings with diagnostics."""

    processor = _processor()
    results: tuple[FcffResult | GrowthRegressionResult | EstimatedGrowthResult, ...] = (
        processor.calculate_fcff(),
        processor.forecast_revenue_growth_regression(),
        processor.estimated_growth(),
    )

    serialized = [to_json_safe(result) for result in results]

    assert Decimal(serialized[0]["fcff"]) == Decimal("71")
    assert serialized[1]["slope"] == "0.1"
    assert Decimal(serialized[2]["estimated_growth"]) == Decimal("0.1")
    assert all(result["diagnostics"] for result in serialized)


def test_processor_import_does_not_construct_external_dependencies() -> None:
    """Constructing the processor should require only a normalized Stock object."""

    processor = _processor()

    assert processor.stock.normalized_ticker() == "TEST"
