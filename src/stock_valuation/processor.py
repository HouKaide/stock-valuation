"""FCFF and growth calculations over normalized stock metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from stock_valuation.contracts import (
    Diagnostic,
    EstimatedGrowthResult,
    FcffResult,
    GrowthRegressionResult,
)
from stock_valuation.errors import InvalidAssumptionsError, MetricUnavailableError
from stock_valuation.stock import Stock


def validate_positive_operating_base(
    metric_name: str,
    value: Decimal,
    *,
    period: Any | None = None,
    suggested_override: str | None = None,
) -> None:
    """Require a finite, positive Decimal operating-base value.

    Parameters
    ----------
    metric_name:
        Name of the operating metric being validated.
    value:
        Decimal value to validate.
    period:
        Optional source period associated with the value.
    suggested_override:
        Optional action that can resolve the invalid operating base.

    Raises
    ------
    InvalidAssumptionsError
        If the value is missing, not a Decimal, non-finite, zero, or negative.
    """

    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise InvalidAssumptionsError(
            metric_name,
            "Operating-base values must be finite and greater than zero.",
            suggested_override=suggested_override,
            period=period,
            value=value,
        )


def calculate_nopat(
    ebit: Decimal,
    tax_rate: Decimal,
    *,
    period: Any | None = None,
) -> Decimal:
    """Calculate and validate net operating profit after tax.

    Parameters
    ----------
    ebit:
        Earnings before interest and taxes.
    tax_rate:
        Marginal corporate tax rate in decimal representation.
    period:
        Optional source period associated with the inputs.

    Returns
    -------
    Decimal
        Positive net operating profit after tax.

    Raises
    ------
    InvalidAssumptionsError
        If an input is malformed or the calculated NOPAT is non-positive.
    """

    _validate_decimal_input("ebit", ebit, period=period)
    _validate_decimal_input("tax_rate", tax_rate, period=period)
    nopat = ebit * (Decimal("1") - tax_rate)
    validate_positive_operating_base(
        "nopat",
        nopat,
        period=period,
        suggested_override="Provide positive EBIT and a valid tax-rate override.",
    )
    return nopat


@dataclass
class DamodaranValuationProcessor:
    """Calculate FCFF and growth from normalized stock data.

    Parameters
    ----------
    stock:
        Stock model exposing normalized annual metrics. The processor never
        calls yfinance or external providers directly.
    """

    stock: Stock
    diagnostics: list[Diagnostic] = field(default_factory=list, init=False)

    def calculate_fcff(self) -> FcffResult:
        """Calculate free cash flow to the firm from normalized inputs.

        Returns
        -------
        FcffResult
            Reproducible FCFF result with calculation steps and diagnostics.

        Raises
        ------
        InvalidAssumptionsError
            If NOPAT or FCFF is non-positive.
        """

        inputs = self.stock.latest_fcff_inputs()
        nopat = calculate_nopat(inputs.ebit, inputs.tax_rate, period=inputs.period)
        fcff = (
            nopat
            + inputs.depreciation_amortization
            - inputs.capex
            - inputs.change_in_non_cash_working_capital
        )
        validate_positive_operating_base(
            "fcff",
            fcff,
            period=inputs.period,
            suggested_override="Review normalized FCFF inputs or provide valid valuation assumptions.",
        )
        diagnostic = Diagnostic(
            kind="calculation",
            message=f"Calculated FCFF from normalized inputs for period {inputs.period}.",
            ticker=self._ticker(),
            metric="fcff",
            source_attempted="Stock.latest_fcff_inputs",
        )
        self._record_diagnostic(diagnostic)
        return FcffResult(
            inputs=inputs,
            nopat=nopat,
            fcff=fcff,
            calculation_steps=(
                "NOPAT = EBIT * (1 - tax_rate)",
                "FCFF = NOPAT + depreciation_and_amortization - capex "
                "- change_in_non_cash_working_capital",
            ),
            diagnostics=[*inputs.diagnostics, diagnostic],
        )

    def historical_revenue_growth(self) -> pd.Series:
        """Return chronological annual revenue growth using all valid periods.

        Returns
        -------
        pandas.Series
            Decimal annual revenue-growth rates ordered oldest to newest.

        Raises
        ------
        MetricUnavailableError
            If fewer than two valid positive annual revenue points exist.
        """

        revenue = self.stock.revenue_series().sort_index()
        valid_periods: list[Any] = []
        valid_values: list[Decimal] = []
        dropped_periods: list[Any] = []
        for period, value in revenue.items():
            if isinstance(value, Decimal) and value.is_finite() and value > 0:
                valid_periods.append(period)
                valid_values.append(value)
            else:
                dropped_periods.append(period)

        if len(valid_values) < 2:
            raise MetricUnavailableError(
                self._ticker(),
                "historical revenue growth",
                source_attempted="Stock.revenue_series",
                suggested_override="Provide at least two positive annual revenue observations.",
            )

        growth_values = [
            (current / previous) - Decimal("1")
            for previous, current in zip(
                valid_values[:-1], valid_values[1:], strict=True
            )
        ]
        growth = pd.Series(growth_values, index=valid_periods[1:], dtype=object)
        source_periods = ", ".join(str(period) for period in valid_periods)
        message = f"Calculated {len(growth)} revenue growth observations from periods: {source_periods}."
        if dropped_periods:
            message += f" Dropped invalid periods: {', '.join(str(period) for period in dropped_periods)}."
        self._record_diagnostic(
            Diagnostic(
                kind="calculation",
                message=message,
                ticker=self._ticker(),
                metric="historical revenue growth",
                source_attempted="Stock.revenue_series",
            )
        )
        return growth

    def forecast_revenue_growth_regression(self) -> GrowthRegressionResult:
        """Fit a simple Decimal regression over historical revenue growth.

        Returns
        -------
        GrowthRegressionResult
            Regression coefficients, sample size, next-year prediction, and
            structured diagnostics.

        Raises
        ------
        MetricUnavailableError
            If fewer than two growth observations are available.
        """

        growth = self.historical_revenue_growth()
        historical_diagnostic = self.diagnostics[-1]
        sample_size = len(growth)
        if sample_size < 2:
            raise MetricUnavailableError(
                self._ticker(),
                "revenue growth regression",
                source_attempted="historical revenue growth",
                suggested_override="Provide at least three valid annual revenue observations.",
            )

        x_values = [Decimal(index) for index in range(sample_size)]
        y_values = list(growth)
        x_mean = sum(x_values, Decimal("0")) / Decimal(sample_size)
        y_mean = sum(y_values, Decimal("0")) / Decimal(sample_size)
        numerator = sum(
            (
                (x_value - x_mean) * (y_value - y_mean)
                for x_value, y_value in zip(x_values, y_values, strict=True)
            ),
            Decimal("0"),
        )
        denominator = sum(
            ((x_value - x_mean) ** 2 for x_value in x_values), Decimal("0")
        )
        slope = numerator / denominator
        intercept = y_mean - (slope * x_mean)
        predicted_growth = intercept + (slope * Decimal(sample_size))
        diagnostic = Diagnostic(
            kind="calculation",
            message=f"Fit revenue growth regression using {sample_size} observations.",
            ticker=self._ticker(),
            metric="revenue growth regression",
            source_attempted="historical revenue growth",
        )
        self._record_diagnostic(diagnostic)
        return GrowthRegressionResult(
            slope=slope,
            intercept=intercept,
            sample_size=sample_size,
            predicted_next_year_growth=predicted_growth,
            diagnostics=[historical_diagnostic, diagnostic],
        )

    def next_year_revenue_growth_from_regression(self) -> Decimal:
        """Return the regression estimate for next-year revenue growth.

        Returns
        -------
        Decimal
            Predicted next-year revenue growth.
        """

        return self.forecast_revenue_growth_regression().predicted_next_year_growth

    def reinvestment_rate(self) -> Decimal:
        """Return the latest reinvestment rate from aligned normalized inputs.

        Returns
        -------
        Decimal
            Latest reinvestment rate.

        Raises
        ------
        InvalidAssumptionsError
            If the latest NOPAT is non-positive.
        MetricUnavailableError
            If normalized series have no common period.
        """

        rates = self._reinvestment_rate_series()
        return rates.iloc[-1]

    def return_on_capital(self) -> Decimal:
        """Return the latest return on capital using prior-period capital.

        Returns
        -------
        Decimal
            Latest return on capital.

        Raises
        ------
        InvalidAssumptionsError
            If prior-period invested capital or NOPAT is non-positive.
        MetricUnavailableError
            If normalized series have no aligned period.
        """

        rates = self._return_on_capital_series()
        return rates.iloc[-1]

    def estimated_growth(self) -> EstimatedGrowthResult:
        """Return default forecast growth from reinvestment and capital return.

        Returns
        -------
        EstimatedGrowthResult
            Latest aligned estimated growth and structured diagnostics.

        Raises
        ------
        MetricUnavailableError
            If reinvestment and return-on-capital series do not align.
        """

        reinvestment = self._reinvestment_rate_series()
        return_on_capital = self._return_on_capital_series()
        common_periods = reinvestment.index.intersection(
            return_on_capital.index
        ).sort_values()
        if common_periods.empty:
            raise MetricUnavailableError(
                self._ticker(),
                "estimated growth",
                source_attempted="reinvestment rate and return on capital",
                suggested_override="Provide aligned annual operating metrics.",
            )
        period = common_periods[-1]
        reinvestment_rate = reinvestment.loc[period]
        capital_return = return_on_capital.loc[period]
        estimated_growth = reinvestment_rate * capital_return
        diagnostic = Diagnostic(
            kind="calculation",
            message=f"Used reinvestment rate times return on capital for default growth at {period}.",
            ticker=self._ticker(),
            metric="estimated growth",
            source_attempted="reinvestment rate * return on capital",
        )
        self._record_diagnostic(diagnostic)
        return EstimatedGrowthResult(
            reinvestment_rate=reinvestment_rate,
            return_on_capital=capital_return,
            estimated_growth=estimated_growth,
            source_method="reinvestment_rate_times_return_on_capital",
            diagnostics=[diagnostic],
        )

    def _reinvestment_rate_series(self) -> pd.Series:
        inputs = self.stock.latest_fcff_inputs()
        nopat = calculate_nopat(inputs.ebit, inputs.tax_rate, period=inputs.period)
        rate = (
            inputs.capex
            - inputs.depreciation_amortization
            + inputs.change_in_non_cash_working_capital
        ) / nopat
        return pd.Series([rate], index=[inputs.period], dtype=object)

    def _return_on_capital_series(self) -> pd.Series:
        inputs = self.stock.latest_fcff_inputs()
        ebit = self.stock.ebit_series().sort_index()
        invested_capital = self.stock.invested_capital_series().sort_index().shift(1)
        common_periods = ebit.index.intersection(
            invested_capital.dropna().index
        ).sort_values()
        if common_periods.empty:
            raise MetricUnavailableError(
                self._ticker(),
                "return on capital",
                source_attempted="prior-period invested capital",
                suggested_override="Provide at least two aligned annual invested-capital observations.",
            )

        values: list[Decimal] = []
        for period in common_periods:
            capital = invested_capital.loc[period]
            validate_positive_operating_base(
                "invested_capital",
                capital,
                period=period,
                suggested_override="Provide positive prior-period invested capital.",
            )
            nopat = calculate_nopat(ebit.loc[period], inputs.tax_rate, period=period)
            values.append(nopat / capital)
        return pd.Series(values, index=common_periods, dtype=object)

    def _ticker(self) -> str:
        return self.stock.normalized_ticker()

    def _record_diagnostic(self, diagnostic: Diagnostic) -> None:
        if diagnostic not in self.diagnostics:
            self.diagnostics.append(diagnostic)


def _validate_decimal_input(
    field_name: str,
    value: Decimal,
    *,
    period: Any | None,
) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise InvalidAssumptionsError(
            field_name,
            "Calculation inputs must be finite Decimal values.",
            suggested_override=f"Provide {field_name} as a finite Decimal.",
            period=period,
            value=value,
        )


__all__ = [
    "DamodaranValuationProcessor",
    "calculate_nopat",
    "validate_positive_operating_base",
]
