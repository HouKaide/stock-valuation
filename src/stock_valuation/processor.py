"""FCFF and growth calculations over normalized stock metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from stock_valuation.contracts import (
    CostOfDebtResult,
    CostOfEquityResult,
    Diagnostic,
    DiscountResult,
    EquityBridgeResult,
    EstimatedGrowthResult,
    FcffResult,
    GrowthRegressionResult,
    TerminalGrowthResult,
    TerminalValueResult,
    ValuationAssumptions,
    ValuationResult,
    WaccResult,
)
from stock_valuation.errors import (
    InvalidAssumptionsError,
    MetricUnavailableError,
    ProviderUnavailableError,
)
from stock_valuation.providers import (
    EquityRiskPremiumProvider,
    FxRateProvider,
    MacroRateProvider,
    MarketDebtProvider,
    SovereignYieldProvider,
    TaxRateProvider,
)
from stock_valuation.stock import Stock


def validate_positive_operating_base(
    metric_name: str,
    value: Decimal,
    *,
    period: Any | None = None,
    suggested_override: str | None = None,
    source_rows: Sequence[str] = (),
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
    source_rows:
        Normalized source rows used to calculate the operating base.

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
            source_rows=source_rows,
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
        source_rows=("EBIT", "tax_rate"),
    )
    return nopat


def validate_terminal_growth(
    wacc: Decimal,
    terminal_growth_rate: Decimal,
) -> None:
    """Require finite terminal growth strictly below WACC.

    Parameters
    ----------
    wacc:
        Weighted average cost of capital.
    terminal_growth_rate:
        Perpetual terminal growth rate.

    Raises
    ------
    InvalidAssumptionsError
        If either rate is malformed or terminal growth is not below WACC.
    """

    _validate_decimal_input("wacc", wacc, period=None)
    _validate_decimal_input(
        "terminal_growth_rate",
        terminal_growth_rate,
        period=None,
    )
    if terminal_growth_rate >= wacc:
        raise InvalidAssumptionsError(
            "terminal_growth_rate",
            "Terminal growth must be strictly lower than WACC.",
            suggested_override="Provide terminal_growth_rate_override below WACC.",
            value=terminal_growth_rate,
            wacc=wacc,
            terminal_growth_rate=terminal_growth_rate,
        )


def discount_factor(wacc: Decimal, year: int) -> Decimal:
    """Calculate the discount factor for a forecast year.

    Parameters
    ----------
    wacc:
        Weighted average cost of capital.
    year:
        Positive forecast year.

    Returns
    -------
    Decimal
        Discount factor ``1 / (1 + WACC) ** year``.

    Raises
    ------
    InvalidAssumptionsError
        If WACC is malformed, ``1 + WACC`` is non-positive, or year is not a
        positive integer.
    """

    _validate_decimal_input("wacc", wacc, period=year)
    if Decimal("1") + wacc <= 0:
        raise InvalidAssumptionsError(
            "wacc",
            "One plus WACC must be positive for discounting.",
            value=wacc,
        )
    if isinstance(year, bool) or not isinstance(year, int) or year <= 0:
        raise InvalidAssumptionsError(
            "year",
            "Discount year must be a positive integer.",
            value=year,
        )
    return Decimal("1") / ((Decimal("1") + wacc) ** year)


@dataclass
class DamodaranValuationProcessor:
    """Calculate FCFF and growth from normalized stock data.

    Parameters
    ----------
    stock:
        Stock model exposing normalized annual metrics. The processor never
        calls yfinance directly.
    assumptions:
        Optional valuation assumptions and overrides.
    macro_provider:
        Optional provider for government yields.
    erp_provider:
        Optional provider for equity risk premiums.
    tax_rate_provider:
        Optional provider for corporate tax rates.
    market_debt_provider:
        Optional provider for market debt values.
    sovereign_yield_provider:
        Optional provider for deterministic sovereign-yield discovery.
    fx_rate_provider:
        Optional provider for converting trading-currency market values into
        the valuation currency.
    """

    stock: Stock
    assumptions: ValuationAssumptions | None = None
    macro_provider: MacroRateProvider | None = None
    erp_provider: EquityRiskPremiumProvider | None = None
    tax_rate_provider: TaxRateProvider | None = None
    market_debt_provider: MarketDebtProvider | None = None
    sovereign_yield_provider: SovereignYieldProvider | None = None
    fx_rate_provider: FxRateProvider | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list, init=False)
    _stock_metadata_count: int = field(default=0, init=False, repr=False)

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
        self._record_stock_mapping_diagnostics()
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
            source_rows=(
                "EBIT",
                "tax_rate",
                "Depreciation And Amortization",
                "Capital Expenditure",
                "Change In Working Capital",
            ),
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
        self._record_stock_mapping_diagnostics()
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

    def risk_free_rate(self) -> Decimal:
        """Resolve the valuation-currency risk-free rate.

        Returns
        -------
        Decimal
            Risk-free rate in decimal representation.

        Raises
        ------
        ProviderUnavailableError
            If neither the currency-matched yield nor US 10-year fallback can
            be resolved.
        """

        assumptions = self._require_assumptions()
        if assumptions.risk_free_rate_override is not None:
            diagnostic = Diagnostic(
                kind="override",
                message="Used risk-free-rate override.",
                ticker=self._ticker(),
                metric="risk-free rate",
                source_attempted="ValuationAssumptions.risk_free_rate_override",
            )
            self._record_diagnostic(diagnostic)
            return assumptions.risk_free_rate_override
        if self.macro_provider is None:
            raise ProviderUnavailableError(
                "macro rate provider",
                "risk-free rate",
                suggested_override="Provide risk_free_rate_override or configure a macro rate provider.",
            )

        currency = (
            assumptions.valuation_currency_override or self.stock.valuation_currency()
        )
        country = (
            assumptions.company_country_override or self.stock.headquarters_country()
        )
        provider_name = type(self.macro_provider).__name__
        try:
            rate = self.macro_provider.get_long_term_government_yield(
                currency,
                country,
                assumptions.valuation_date,
            )
        except ProviderUnavailableError:
            try:
                rate = self.macro_provider.get_us_10y_treasury_yield(
                    assumptions.valuation_date
                )
            except ProviderUnavailableError as fallback_error:
                raise ProviderUnavailableError(
                    provider_name,
                    "risk-free rate",
                    source_attempted="currency-matched government yield",
                    fallbacks_attempted=("US 10-year Treasury yield",),
                    suggested_override="Provide risk_free_rate_override.",
                ) from fallback_error
            diagnostic = Diagnostic(
                kind="fallback",
                message="Used US 10-year Treasury yield after currency-matched yield was unavailable.",
                ticker=self._ticker(),
                metric="risk-free rate",
                provider=provider_name,
                source_attempted="US 10-year Treasury yield",
                fallbacks_attempted=("currency-matched government yield",),
            )
            self._record_diagnostic(diagnostic)
            return rate
        diagnostic = Diagnostic(
            kind="provider",
            message=f"Resolved {currency} risk-free rate from macro provider.",
            ticker=self._ticker(),
            metric="risk-free rate",
            provider=provider_name,
            source_attempted="currency-matched government yield",
        )
        self._record_diagnostic(diagnostic)
        return rate

    def equity_risk_premium(self) -> Decimal:
        """Resolve the company-country equity risk premium.

        Returns
        -------
        Decimal
            Equity risk premium in decimal representation.

        Raises
        ------
        ProviderUnavailableError
            If no override or ERP provider value is available.
        """

        assumptions = self._require_assumptions()
        if assumptions.equity_risk_premium_override is not None:
            diagnostic = Diagnostic(
                kind="override",
                message="Used equity-risk-premium override.",
                ticker=self._ticker(),
                metric="equity risk premium",
                source_attempted="ValuationAssumptions.equity_risk_premium_override",
            )
            self._record_diagnostic(diagnostic)
            return assumptions.equity_risk_premium_override
        if self.erp_provider is None:
            raise ProviderUnavailableError(
                "equity risk premium provider",
                "equity risk premium",
                suggested_override="Provide equity_risk_premium_override or configure an ERP provider.",
            )

        country = (
            assumptions.company_country_override or self.stock.headquarters_country()
        )
        premium = self.erp_provider.get_equity_risk_premium(
            country,
            assumptions.valuation_date,
        )
        self._record_diagnostic(
            Diagnostic(
                kind="provider",
                message=f"Resolved equity risk premium for {country}.",
                ticker=self._ticker(),
                metric="equity risk premium",
                provider=type(self.erp_provider).__name__,
                source_attempted="country equity risk premium",
            )
        )
        return premium

    def beta(self) -> Decimal:
        """Resolve equity beta from an override or normalized stock data.

        Returns
        -------
        Decimal
            Equity beta.

        Raises
        ------
        MetricUnavailableError
            If beta is absent and no override is supplied.
        """

        assumptions = self._require_assumptions()
        if assumptions.beta_override is not None:
            self._record_diagnostic(
                Diagnostic(
                    kind="override",
                    message="Used beta override.",
                    ticker=self._ticker(),
                    metric="beta",
                    source_attempted="ValuationAssumptions.beta_override",
                )
            )
            return assumptions.beta_override
        beta = self.stock.beta()
        if beta is None:
            raise MetricUnavailableError(
                self._ticker(),
                "beta",
                source_attempted="Stock.beta",
                suggested_override="Provide beta_override.",
            )
        self._record_diagnostic(
            Diagnostic(
                kind="source",
                message="Resolved beta from normalized stock metadata.",
                ticker=self._ticker(),
                metric="beta",
                source_attempted="Stock.beta",
            )
        )
        return beta

    def cost_of_equity(self) -> CostOfEquityResult:
        """Calculate cost of equity using risk-free rate, beta, and ERP.

        Returns
        -------
        CostOfEquityResult
            Cost-of-equity inputs, result, sources, steps, and diagnostics.
        """

        start = len(self.diagnostics)
        risk_free_rate = self.risk_free_rate()
        beta = self.beta()
        equity_risk_premium = self.equity_risk_premium()
        cost_of_equity = risk_free_rate + (beta * equity_risk_premium)
        diagnostics = self.diagnostics[start:]
        return CostOfEquityResult(
            risk_free_rate=risk_free_rate,
            beta=beta,
            equity_risk_premium=equity_risk_premium,
            cost_of_equity=cost_of_equity,
            source_details=tuple(
                diagnostic.source_attempted or diagnostic.kind
                for diagnostic in diagnostics
            ),
            diagnostics=diagnostics,
            calculation_steps=(
                "Cost of Equity = Risk-Free Rate + Beta * Equity Risk Premium",
            ),
        )

    def market_value_of_equity(self) -> Decimal:
        """Return positive market capitalization for WACC weighting.

        Returns
        -------
        Decimal
            Market value of equity.

        Raises
        ------
        MetricUnavailableError
            If market capitalization is missing or non-positive.
        """

        market_cap = self.stock.market_cap()
        self._record_stock_mapping_diagnostics()
        if market_cap is None or market_cap <= 0:
            raise MetricUnavailableError(
                self._ticker(),
                "market value of equity",
                source_attempted="Stock.market_cap",
                suggested_override="Provide a positive market capitalization source.",
            )
        market_cap = self._convert_market_amount(
            market_cap,
            metric_name="market value of equity",
        )
        self._record_diagnostic(
            Diagnostic(
                kind="source",
                message="Resolved market value of equity from market capitalization.",
                ticker=self._ticker(),
                metric="market value of equity",
                source_attempted="Stock.market_cap",
            )
        )
        return market_cap

    def market_value_of_debt(self) -> Decimal:
        """Resolve market debt with an override, provider, or book fallback.

        Returns
        -------
        Decimal
            Positive market or book value of debt.

        Raises
        ------
        MetricUnavailableError
            If neither provider nor normalized book debt yields a positive
            value.
        """

        assumptions = self._require_assumptions()
        if assumptions.market_debt_override is not None:
            debt = assumptions.market_debt_override
            source_kind = "override"
            message = "Used market-debt override."
            source = "ValuationAssumptions.market_debt_override"
        elif self.market_debt_provider is not None:
            debt = self.market_debt_provider.get_market_value_of_debt(
                self._ticker(),
                assumptions.valuation_date,
            )
            if debt is not None:
                source_kind = "provider"
                message = "Resolved market value of debt from provider."
                source = "MarketDebtProvider.get_market_value_of_debt"
            else:
                debt = self._latest_book_debt()
                source_kind = "fallback"
                message = (
                    "Market debt provider returned no value; used latest book debt."
                )
                source = "Stock.debt_series"
        else:
            debt = self._latest_book_debt()
            source_kind = "fallback"
            message = "No market debt provider configured; used latest book debt."
            source = "Stock.debt_series"
        if debt <= 0:
            raise MetricUnavailableError(
                self._ticker(),
                "market value of debt",
                source_attempted=source,
                suggested_override="Provide a positive market_debt_override.",
            )
        self._record_diagnostic(
            Diagnostic(
                kind=source_kind,
                message=message,
                ticker=self._ticker(),
                metric="market value of debt",
                provider=(
                    type(self.market_debt_provider).__name__
                    if self.market_debt_provider is not None
                    else None
                ),
                source_attempted=source,
            )
        )
        return debt

    def average_debt(self) -> Decimal:
        """Calculate positive average debt from the latest adjacent periods.

        Returns
        -------
        Decimal
            Average of the latest two annual debt values.

        Raises
        ------
        MetricUnavailableError
            If fewer than two periods exist or average debt is non-positive.
        """

        average_debt, periods = self._average_debt_details()
        self._record_diagnostic(
            Diagnostic(
                kind="calculation",
                message=f"Calculated average debt from periods {periods[0]} and {periods[1]}.",
                ticker=self._ticker(),
                metric="average debt",
                source_attempted="Stock.debt_series",
            )
        )
        return average_debt

    def tax_rate(self) -> Decimal:
        """Resolve the marginal corporate tax rate.

        Returns
        -------
        Decimal
            Corporate tax rate in decimal representation.

        Raises
        ------
        ProviderUnavailableError
            If no override or provider value is available.
        """

        assumptions = self._require_assumptions()
        if assumptions.tax_rate_override is not None:
            self._record_diagnostic(
                Diagnostic(
                    kind="override",
                    message="Used tax-rate override.",
                    ticker=self._ticker(),
                    metric="tax rate",
                    source_attempted="ValuationAssumptions.tax_rate_override",
                )
            )
            return assumptions.tax_rate_override
        provider = self.tax_rate_provider or getattr(
            self.stock, "tax_rate_provider", None
        )
        if provider is None:
            raise ProviderUnavailableError(
                "tax rate provider",
                "corporate tax rate",
                suggested_override="Provide tax_rate_override or configure a tax-rate provider.",
            )
        country = (
            assumptions.company_country_override or self.stock.headquarters_country()
        )
        tax_rate = provider.get_corporate_tax_rate(
            country,
            assumptions.valuation_date,
        )
        self._record_diagnostic(
            Diagnostic(
                kind="provider",
                message=f"Resolved corporate tax rate for {country}.",
                ticker=self._ticker(),
                metric="tax rate",
                provider=type(provider).__name__,
                source_attempted="country corporate tax rate",
            )
        )
        return tax_rate

    def cost_of_debt(self) -> CostOfDebtResult:
        """Calculate pre-tax and after-tax historical cost of debt.

        Returns
        -------
        CostOfDebtResult
            Interest expense, average debt, tax rate, rates, periods, steps,
            and diagnostics.

        Raises
        ------
        MetricUnavailableError
            If debt or interest expense cannot be aligned.
        """

        start = len(self.diagnostics)
        average_debt, debt_periods = self._average_debt_details()
        interest = self.stock.interest_expense_series().sort_index()
        self._record_stock_mapping_diagnostics()
        current_period = debt_periods[-1]
        if current_period not in interest.index:
            raise MetricUnavailableError(
                self._ticker(),
                "interest expense",
                source_attempted=f"Stock.interest_expense_series at {current_period}",
            )
        interest_expense = interest.loc[current_period]
        if not isinstance(interest_expense, Decimal) or interest_expense <= 0:
            raise MetricUnavailableError(
                self._ticker(),
                "interest expense",
                source_attempted=f"Stock.interest_expense_series at {current_period}",
            )
        tax_rate = self.tax_rate()
        pretax_cost = interest_expense / average_debt
        after_tax_cost = pretax_cost * (Decimal("1") - tax_rate)
        diagnostic = Diagnostic(
            kind="calculation",
            message=f"Calculated cost of debt using periods {debt_periods[0]} and {debt_periods[1]}.",
            ticker=self._ticker(),
            metric="cost of debt",
            source_attempted="interest expense / average debt",
        )
        self._record_diagnostic(diagnostic)
        return CostOfDebtResult(
            interest_expense=interest_expense,
            average_debt=average_debt,
            pretax_cost_of_debt=pretax_cost,
            tax_rate=tax_rate,
            after_tax_cost_of_debt=after_tax_cost,
            diagnostics=self.diagnostics[start:],
            source_periods=debt_periods,
            calculation_steps=(
                "Average Debt = (Debt_t + Debt_t-1) / 2",
                "Cost of Debt = Interest Expense / Average Debt",
                "After-Tax Cost of Debt = Cost of Debt * (1 - tax_rate)",
            ),
        )

    def capital_weights(
        self,
        market_value_of_equity: Decimal,
        market_value_of_debt: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Calculate total capital and exact equity and debt weights.

        Parameters
        ----------
        market_value_of_equity:
            Market value of equity.
        market_value_of_debt:
            Market value of debt.

        Returns
        -------
        tuple[Decimal, Decimal, Decimal]
            Total capital, equity weight, and debt weight.

        Raises
        ------
        InvalidAssumptionsError
            If total capital is zero or negative.
        """

        total_capital = market_value_of_equity + market_value_of_debt
        validate_positive_operating_base(
            "total_capital",
            total_capital,
            suggested_override="Provide positive market values of debt and equity.",
        )
        return (
            total_capital,
            market_value_of_equity / total_capital,
            market_value_of_debt / total_capital,
        )

    def wacc(
        self,
        *,
        cost_of_equity_result: CostOfEquityResult | None = None,
        cost_of_debt_result: CostOfDebtResult | None = None,
    ) -> WaccResult:
        """Calculate weighted average cost of capital with full breakdown.

        Parameters
        ----------
        cost_of_equity_result:
            Optional precomputed cost-of-equity result.
        cost_of_debt_result:
            Optional precomputed cost-of-debt result.

        Returns
        -------
        WaccResult
            Capital values, weights, component costs, WACC, steps, and
            diagnostics.
        """

        start = len(self.diagnostics)
        equity_cost = cost_of_equity_result or self.cost_of_equity()
        market_equity = self.market_value_of_equity()
        market_debt = self.market_value_of_debt()
        debt_cost = cost_of_debt_result or self.cost_of_debt()
        total_capital, equity_weight, debt_weight = self.capital_weights(
            market_equity,
            market_debt,
        )
        wacc = (equity_weight * equity_cost.cost_of_equity) + (
            debt_weight * debt_cost.after_tax_cost_of_debt
        )
        diagnostic = Diagnostic(
            kind="calculation",
            message="Calculated WACC from market capital weights and component costs.",
            ticker=self._ticker(),
            metric="wacc",
            source_attempted="capital-weighted cost of equity and after-tax debt",
        )
        self._record_diagnostic(diagnostic)
        return WaccResult(
            market_value_of_equity=market_equity,
            market_value_of_debt=market_debt,
            equity_weight=equity_weight,
            debt_weight=debt_weight,
            cost_of_equity=equity_cost.cost_of_equity,
            pretax_cost_of_debt=debt_cost.pretax_cost_of_debt,
            tax_rate=debt_cost.tax_rate,
            wacc=wacc,
            calculation_steps=(
                "Total Capital = Market Value of Equity + Market Value of Debt",
                "Equity Weight = E / (D + E)",
                "Debt Weight = D / (D + E)",
                "WACC = Equity Weight * Re + Debt Weight * Rd * (1 - T)",
            ),
            diagnostics=self.diagnostics[start:],
            total_capital=total_capital,
            after_tax_cost_of_debt=debt_cost.after_tax_cost_of_debt,
        )

    def terminal_growth_rate(self) -> TerminalGrowthResult:
        """Resolve terminal growth from an override or sovereign yield.

        Returns
        -------
        TerminalGrowthResult
            Selected instrument, yield, date, provider, fallbacks, and
            diagnostics.

        Raises
        ------
        ProviderUnavailableError
            If no override or usable deterministic provider result exists.
        """

        assumptions = self._require_assumptions()
        if assumptions.terminal_growth_rate_override is not None:
            diagnostic = Diagnostic(
                kind="override",
                message="Used terminal-growth-rate override.",
                ticker=self._ticker(),
                metric="terminal growth rate",
                source_attempted="ValuationAssumptions.terminal_growth_rate_override",
            )
            self._record_diagnostic(diagnostic)
            return TerminalGrowthResult(
                selected_instrument="override",
                yield_value=assumptions.terminal_growth_rate_override,
                valuation_date=assumptions.valuation_date,
                provider="user override",
                fallbacks_attempted=(),
                diagnostics=[diagnostic],
            )
        if self.sovereign_yield_provider is None:
            raise ProviderUnavailableError(
                "sovereign yield provider",
                "terminal growth rate",
                suggested_override="Provide terminal_growth_rate_override or configure a sovereign-yield provider.",
            )

        currency = (
            assumptions.valuation_currency_override or self.stock.valuation_currency()
        )
        country = (
            assumptions.company_country_override or self.stock.headquarters_country()
        )
        result = self.sovereign_yield_provider.find_10y_sovereign_yield(
            currency,
            country,
            assumptions.valuation_date,
        )
        selected = result.selected
        if (
            not isinstance(selected.yield_value, Decimal)
            or not selected.yield_value.is_finite()
        ):
            raise ProviderUnavailableError(
                result.provider or type(self.sovereign_yield_provider).__name__,
                "terminal growth rate",
                source_attempted=selected.symbol,
                suggested_override="Provide terminal_growth_rate_override.",
            )
        rejected_symbols = tuple(candidate.symbol for candidate in result.rejected)
        selection_diagnostic = Diagnostic(
            kind="provider",
            message=(
                f"Selected {selected.symbol} for {currency}/{country}; "
                f"rejected alternatives: {', '.join(rejected_symbols) or 'none'}."
            ),
            ticker=self._ticker(),
            metric="terminal growth rate",
            provider=result.provider,
            source_attempted=selected.symbol,
            fallbacks_attempted=rejected_symbols,
        )
        diagnostics = [*result.diagnostics, selection_diagnostic]
        for diagnostic in diagnostics:
            self._record_diagnostic(diagnostic)
        return TerminalGrowthResult(
            selected_instrument=selected.symbol,
            yield_value=selected.yield_value,
            valuation_date=result.valuation_date,
            provider=result.provider,
            fallbacks_attempted=rejected_symbols,
            diagnostics=diagnostics,
        )

    def terminal_value(
        self,
        final_forecast_year_fcff: Decimal,
        wacc: Decimal,
        terminal_growth_rate: Decimal,
        forecast_years: int,
    ) -> TerminalValueResult:
        """Calculate terminal value and its present value.

        Parameters
        ----------
        final_forecast_year_fcff:
            FCFF in the final explicit forecast year.
        wacc:
            Weighted average cost of capital.
        terminal_growth_rate:
            Perpetual terminal growth rate.
        forecast_years:
            Number of years from valuation date to terminal value.

        Returns
        -------
        TerminalValueResult
            Final FCFF, growth, next-year FCFF, terminal value, present value,
            calculation steps, and diagnostics.

        Raises
        ------
        InvalidAssumptionsError
            If inputs are malformed, forecast years are invalid, or terminal
            growth is not below WACC.
        """

        _validate_decimal_input(
            "final_forecast_year_fcff",
            final_forecast_year_fcff,
            period=forecast_years,
        )
        validate_positive_operating_base(
            "final_forecast_year_fcff",
            final_forecast_year_fcff,
            period=forecast_years,
        )
        if isinstance(forecast_years, bool) or not isinstance(forecast_years, int):
            raise InvalidAssumptionsError(
                "forecast_years",
                "Forecast years must be an integer.",
                value=forecast_years,
            )
        if forecast_years <= 0:
            raise InvalidAssumptionsError(
                "forecast_years",
                "Forecast years must be positive.",
                value=forecast_years,
            )
        validate_terminal_growth(wacc, terminal_growth_rate)
        next_year_fcff = final_forecast_year_fcff * (
            Decimal("1") + terminal_growth_rate
        )
        terminal_value = next_year_fcff / (wacc - terminal_growth_rate)
        present_value = terminal_value / ((Decimal("1") + wacc) ** forecast_years)
        diagnostic = Diagnostic(
            kind="calculation",
            message=(
                f"Validated terminal growth below WACC and discounted terminal "
                f"value over {forecast_years} years."
            ),
            ticker=self._ticker(),
            metric="terminal value",
            source_attempted="Gordon growth model",
        )
        self._record_diagnostic(diagnostic)
        return TerminalValueResult(
            final_forecast_year_fcff=final_forecast_year_fcff,
            terminal_growth_rate=terminal_growth_rate,
            next_year_fcff=next_year_fcff,
            terminal_value=terminal_value,
            present_value_terminal_value=present_value,
            calculation_steps=(
                "FCF_n+1 = FCF_n * (1 + g)",
                "Terminal Value = FCF_n+1 / (WACC - g)",
                "Present Value of Terminal Value = Terminal Value / (1 + WACC)^n",
            ),
            diagnostics=[diagnostic],
        )

    def forecast_fcff_discount_table(
        self,
        forecast_fcffs: Sequence[Decimal],
        wacc: Decimal,
    ) -> pd.DataFrame:
        """Build an ordered table of discounted forecast FCFF values.

        Parameters
        ----------
        forecast_fcffs:
            Forecast FCFF values ordered from year one onward.
        wacc:
            Weighted average cost of capital.

        Returns
        -------
        pandas.DataFrame
            Deterministic rows with year, FCFF, discount factor, and present
            value.
        """

        rows: list[dict[str, Any]] = []
        for year, fcff in enumerate(forecast_fcffs, start=1):
            _validate_decimal_input("forecast_fcff", fcff, period=year)
            factor = discount_factor(wacc, year)
            rows.append(
                {
                    "year": year,
                    "fcff": fcff,
                    "discount_factor": factor,
                    "present_value": fcff * factor,
                    "cash_flow_type": "forecast",
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "year",
                "fcff",
                "discount_factor",
                "present_value",
                "cash_flow_type",
            ],
        )

    def discount_to_today(
        self,
        forecast_fcffs: Sequence[Decimal],
        wacc: Decimal,
        terminal_value: Decimal,
    ) -> DiscountResult:
        """Discount forecast FCFF and terminal value to today.

        Parameters
        ----------
        forecast_fcffs:
            Forecast FCFF values ordered from year one onward.
        wacc:
            Weighted average cost of capital.
        terminal_value:
            Undiscounted terminal value at the final forecast year.

        Returns
        -------
        DiscountResult
            Discount table, forecast present value, terminal present value,
            enterprise value, and diagnostics.
        """

        if not forecast_fcffs:
            raise InvalidAssumptionsError(
                "forecast_fcffs",
                "At least one forecast FCFF value is required.",
            )
        _validate_decimal_input(
            "terminal_value", terminal_value, period=len(forecast_fcffs)
        )
        forecast_table = self.forecast_fcff_discount_table(forecast_fcffs, wacc)
        terminal_factor = discount_factor(wacc, len(forecast_fcffs))
        present_value_terminal = terminal_value * terminal_factor
        terminal_row = pd.DataFrame(
            [
                {
                    "year": len(forecast_fcffs),
                    "fcff": terminal_value,
                    "discount_factor": terminal_factor,
                    "present_value": present_value_terminal,
                    "cash_flow_type": "terminal",
                }
            ]
        )
        discount_table = pd.concat(
            [forecast_table, terminal_row],
            ignore_index=True,
        )
        present_value_forecasts = sum(
            forecast_table["present_value"],
            Decimal("0"),
        )
        enterprise_value = present_value_forecasts + present_value_terminal
        diagnostic = Diagnostic(
            kind="calculation",
            message=(
                f"Built discount table with {len(forecast_fcffs)} forecast rows "
                "and one terminal row."
            ),
            ticker=self._ticker(),
            metric="discounting",
            source_attempted="forecast FCFF and terminal value",
        )
        self._record_diagnostic(diagnostic)
        return DiscountResult(
            discount_table=discount_table,
            present_value_forecast_fcffs=present_value_forecasts,
            present_value_terminal_value=present_value_terminal,
            enterprise_value=enterprise_value,
            diagnostics=[diagnostic],
        )

    def debt_adjustment(self, wacc_result: WaccResult | None = None) -> Decimal:
        """Resolve the positive debt claim used by the equity bridge.

        Parameters
        ----------
        wacc_result:
            Optional WACC result containing an already-resolved market debt.

        Returns
        -------
        Decimal
            Positive debt deducted from enterprise value.
        """

        override = (
            self.assumptions.market_debt_override
            if self.assumptions is not None
            else None
        )
        if override is not None:
            debt = override
            kind = "override"
            message = "Used market-debt override for the equity bridge."
            source = "ValuationAssumptions.market_debt_override"
        elif wacc_result is not None:
            debt = wacc_result.market_value_of_debt
            kind = "source"
            message = "Reused market debt resolved by the WACC flow."
            source = "WaccResult.market_value_of_debt"
        else:
            debt = self._latest_book_debt()
            kind = "fallback"
            message = "Used latest book debt for the equity bridge."
            source = "Stock.debt_series"
        _validate_non_negative_decimal("debt", debt)
        if debt == 0:
            raise MetricUnavailableError(
                self._ticker(),
                "debt adjustment",
                source_attempted=source,
                suggested_override="Provide a positive market_debt_override.",
            )
        self._record_diagnostic(
            Diagnostic(
                kind=kind,
                message=message,
                ticker=self._ticker(),
                metric="debt adjustment",
                source_attempted=source,
            )
        )
        return debt

    def cash_adjustment(self, *, allow_zero_when_missing: bool = True) -> Decimal:
        """Resolve the positive cash adjustment used by the equity bridge.

        Parameters
        ----------
        allow_zero_when_missing:
            Whether unavailable cash may default to zero with a diagnostic.

        Returns
        -------
        Decimal
            Positive cash added to enterprise value.

        Raises
        ------
        MetricUnavailableError
            If cash is unavailable and zero fallback is disabled.
        """

        try:
            cash_series = self.stock.cash_series().sort_index()
            self._record_stock_mapping_diagnostics()
            if cash_series.empty:
                raise MetricUnavailableError(
                    self._ticker(),
                    "cash adjustment",
                    source_attempted="Stock.cash_series",
                )
            cash = cash_series.iloc[-1]
            _validate_non_negative_decimal("cash", cash)
        except MetricUnavailableError:
            if not allow_zero_when_missing:
                raise
            cash = Decimal("0")
            diagnostic = Diagnostic(
                kind="fallback",
                message="Cash was unavailable; used zero for the equity bridge.",
                ticker=self._ticker(),
                metric="cash adjustment",
                source_attempted="Stock.cash_series",
            )
        else:
            diagnostic = Diagnostic(
                kind="source",
                message="Resolved cash from the latest normalized balance-sheet value.",
                ticker=self._ticker(),
                metric="cash adjustment",
                source_attempted="Stock.cash_series",
            )
        self._record_diagnostic(diagnostic)
        return cash

    def non_operating_assets(
        self,
        explicit_value: Decimal | None = None,
    ) -> Decimal:
        """Resolve non-operating assets or default the adjustment to zero.

        Parameters
        ----------
        explicit_value:
            Optional user-supplied non-operating asset value.

        Returns
        -------
        Decimal
            Positive non-operating assets added to enterprise value.
        """

        if explicit_value is not None:
            _validate_non_negative_decimal("non_operating_assets", explicit_value)
            value = explicit_value
            diagnostic = Diagnostic(
                kind="override",
                message="Used explicit non-operating assets value.",
                ticker=self._ticker(),
                metric="non-operating assets",
                source_attempted="explicit_value",
            )
        else:
            series = self.stock.non_operating_assets_series()
            if series is None or series.empty:
                value = Decimal("0")
                diagnostic = Diagnostic(
                    kind="fallback",
                    message=(
                        "No identifiable non-operating assets source existed; "
                        "used zero."
                    ),
                    ticker=self._ticker(),
                    metric="non-operating assets",
                    source_attempted="Stock.non_operating_assets_series",
                )
            else:
                value = series.sort_index().iloc[-1]
                _validate_non_negative_decimal("non_operating_assets", value)
                diagnostic = Diagnostic(
                    kind="source",
                    message=(
                        "Resolved non-operating assets from the latest normalized "
                        "balance-sheet value."
                    ),
                    ticker=self._ticker(),
                    metric="non-operating assets",
                    source_attempted="Stock.non_operating_assets_series",
                )
        self._record_diagnostic(diagnostic)
        return value

    def minority_interest(self) -> Decimal:
        """Resolve minority interest or default the adjustment to zero.

        Returns
        -------
        Decimal
            Positive minority interest deducted from enterprise value.
        """

        series = self.stock.minority_interest_series()
        if series is None or series.empty:
            value = Decimal("0")
            diagnostic = Diagnostic(
                kind="fallback",
                message="Minority interest was unavailable; used zero.",
                ticker=self._ticker(),
                metric="minority interest",
                source_attempted="Stock.minority_interest_series",
            )
        else:
            value = series.sort_index().iloc[-1]
            _validate_non_negative_decimal("minority_interest", value)
            diagnostic = Diagnostic(
                kind="source",
                message=(
                    "Resolved minority interest from the latest normalized "
                    "balance-sheet value."
                ),
                ticker=self._ticker(),
                metric="minority interest",
                source_attempted="Stock.minority_interest_series",
            )
        self._record_diagnostic(diagnostic)
        return value

    def equity_bridge(
        self,
        enterprise_value: Decimal,
        *,
        wacc_result: WaccResult | None = None,
        non_operating_assets: Decimal | None = None,
        allow_zero_cash: bool = True,
    ) -> EquityBridgeResult:
        """Convert enterprise value to equity value.

        Parameters
        ----------
        enterprise_value:
            Discounted enterprise value.
        wacc_result:
            Optional WACC result containing resolved market debt.
        non_operating_assets:
            Optional explicit non-operating assets value.
        allow_zero_cash:
            Whether unavailable cash may default to zero.

        Returns
        -------
        EquityBridgeResult
            Every bridge component, equity value, formula, and diagnostics.
        """

        _validate_decimal_input("enterprise_value", enterprise_value, period=None)
        diagnostic_start = len(self.diagnostics)
        debt = self.debt_adjustment(wacc_result)
        cash = self.cash_adjustment(allow_zero_when_missing=allow_zero_cash)
        other_assets = self.non_operating_assets(non_operating_assets)
        minority_interest = self.minority_interest()
        equity_value = enterprise_value - debt + cash + other_assets - minority_interest
        diagnostic = Diagnostic(
            kind="calculation",
            message="Calculated equity value from explicit bridge adjustments.",
            ticker=self._ticker(),
            metric="equity value",
            source_attempted="enterprise value and equity bridge components",
        )
        self._record_diagnostic(diagnostic)
        return EquityBridgeResult(
            enterprise_value=enterprise_value,
            debt=debt,
            cash=cash,
            non_operating_assets=other_assets,
            minority_interest=minority_interest,
            equity_value=equity_value,
            diagnostics=self.diagnostics[diagnostic_start:],
            calculation_steps=(
                "Equity Value = Enterprise Value - Debt + Cash "
                "+ Non-Operating Assets - Minority Interest",
            ),
        )

    def shares_outstanding(self) -> Decimal:
        """Resolve and validate shares outstanding.

        Returns
        -------
        Decimal
            Positive shares outstanding.

        Raises
        ------
        InvalidAssumptionsError
            If an override is zero or negative.
        MetricUnavailableError
            If the normalized stock value is zero or negative.
        """

        override = (
            self.assumptions.shares_outstanding_override
            if self.assumptions is not None
            else None
        )
        if override is not None:
            if not override.is_finite() or override <= 0:
                raise InvalidAssumptionsError(
                    "shares_outstanding_override",
                    "Shares outstanding must be finite and greater than zero.",
                    value=override,
                )
            shares = override
            diagnostic = Diagnostic(
                kind="override",
                message="Used shares-outstanding override.",
                ticker=self._ticker(),
                metric="shares outstanding",
                source_attempted="ValuationAssumptions.shares_outstanding_override",
            )
        else:
            shares = self.stock.shares_outstanding()
            self._record_stock_mapping_diagnostics()
            if not isinstance(shares, Decimal) or not shares.is_finite() or shares <= 0:
                raise MetricUnavailableError(
                    self._ticker(),
                    "shares outstanding",
                    source_attempted="Stock.shares_outstanding",
                    suggested_override="Provide a positive shares_outstanding_override.",
                )
            diagnostic = Diagnostic(
                kind="source",
                message="Resolved shares outstanding from normalized stock data.",
                ticker=self._ticker(),
                metric="shares outstanding",
                source_attempted="Stock.shares_outstanding",
            )
        self._record_diagnostic(diagnostic)
        return shares

    def intrinsic_value_per_share(
        self,
        equity_value: Decimal,
        shares_outstanding: Decimal | None = None,
    ) -> Decimal:
        """Calculate intrinsic value per share.

        Parameters
        ----------
        equity_value:
            Equity value after bridge adjustments.
        shares_outstanding:
            Optional already-resolved positive share count.

        Returns
        -------
        Decimal
            Exact Decimal quotient of equity value and shares outstanding.
        """

        _validate_decimal_input("equity_value", equity_value, period=None)
        shares = (
            self.shares_outstanding()
            if shares_outstanding is None
            else shares_outstanding
        )
        if not isinstance(shares, Decimal) or not shares.is_finite() or shares <= 0:
            raise InvalidAssumptionsError(
                "shares_outstanding",
                "Shares outstanding must be finite and greater than zero.",
                value=shares,
            )
        return equity_value / shares

    def current_price(self) -> Decimal | None:
        """Resolve current price without blocking intrinsic value calculation.

        Returns
        -------
        Decimal | None
            Current market price, or ``None`` when unavailable.
        """

        try:
            price = self.stock.current_price()
            self._record_stock_mapping_diagnostics()
        except MetricUnavailableError:
            self._record_diagnostic(
                Diagnostic(
                    kind="fallback",
                    message="Current price was unavailable; omitted market comparison.",
                    ticker=self._ticker(),
                    metric="current price",
                    source_attempted="Stock.current_price",
                )
            )
            return None
        price = self._convert_market_amount(price, metric_name="current price")
        self._record_diagnostic(
            Diagnostic(
                kind="source",
                message="Resolved current price from normalized stock data.",
                ticker=self._ticker(),
                metric="current price",
                source_attempted="Stock.current_price",
            )
        )
        return price

    def upside_downside(
        self,
        intrinsic_value_per_share: Decimal,
        current_price: Decimal | None,
    ) -> Decimal | None:
        """Calculate upside or downside relative to current price.

        Parameters
        ----------
        intrinsic_value_per_share:
            Calculated intrinsic value per share.
        current_price:
            Current market price, or ``None`` when unavailable.

        Returns
        -------
        Decimal | None
            Relative upside or downside, or ``None`` without current price.
        """

        _validate_decimal_input(
            "intrinsic_value_per_share",
            intrinsic_value_per_share,
            period=None,
        )
        if current_price is None:
            self._record_diagnostic(
                Diagnostic(
                    kind="fallback",
                    message="Upside/downside omitted because current price is unavailable.",
                    ticker=self._ticker(),
                    metric="upside/downside",
                    source_attempted="current price",
                )
            )
            return None
        if not isinstance(current_price, Decimal) or not current_price.is_finite():
            raise InvalidAssumptionsError(
                "current_price",
                "Current price must be a finite Decimal.",
                value=current_price,
            )
        if current_price <= 0:
            raise InvalidAssumptionsError(
                "current_price",
                "Current price must be greater than zero.",
                value=current_price,
            )
        return intrinsic_value_per_share / current_price - Decimal("1")

    def value(self) -> ValuationResult:
        """Run the complete valuation workflow.

        Returns
        -------
        ValuationResult
            Complete valuation result with calculations and diagnostics.
        """

        assumptions = self._require_assumptions()
        fcff = self.calculate_fcff()
        growth = self.estimated_growth()
        cost_of_equity = self.cost_of_equity()
        cost_of_debt = self.cost_of_debt()
        wacc = self.wacc(
            cost_of_equity_result=cost_of_equity,
            cost_of_debt_result=cost_of_debt,
        )
        forecast_fcffs = [
            fcff.fcff * ((Decimal("1") + growth.estimated_growth) ** year)
            for year in range(1, assumptions.forecast_years + 1)
        ]
        forecast_table = pd.DataFrame(
            {
                "year": range(1, assumptions.forecast_years + 1),
                "fcff": forecast_fcffs,
            }
        )
        terminal_growth = self.terminal_growth_rate()
        terminal_value = self.terminal_value(
            forecast_fcffs[-1],
            wacc.wacc,
            terminal_growth.yield_value,
            assumptions.forecast_years,
        )
        discounting = self.discount_to_today(
            forecast_fcffs,
            wacc.wacc,
            terminal_value.terminal_value,
        )
        return self.assemble_valuation_result(
            forecast_table=forecast_table,
            fcff=fcff,
            growth=growth,
            cost_of_equity=cost_of_equity,
            cost_of_debt=cost_of_debt,
            wacc=wacc,
            terminal_value=terminal_value,
            discounting=discounting,
        )

    def assemble_valuation_result(
        self,
        *,
        forecast_table: pd.DataFrame,
        fcff: FcffResult,
        growth: EstimatedGrowthResult,
        cost_of_equity: CostOfEquityResult,
        cost_of_debt: CostOfDebtResult,
        wacc: WaccResult,
        terminal_value: TerminalValueResult,
        discounting: DiscountResult,
        non_operating_assets: Decimal | None = None,
        allow_zero_cash: bool = True,
    ) -> ValuationResult:
        """Assemble upstream valuation outputs with bridge and per-share values.

        Parameters
        ----------
        forecast_table:
            Explicit forecast-period data.
        fcff:
            FCFF result.
        growth:
            Estimated-growth result.
        cost_of_equity:
            Cost-of-equity result.
        cost_of_debt:
            Cost-of-debt result.
        wacc:
            WACC result containing resolved market debt.
        terminal_value:
            Terminal-value result.
        discounting:
            Discounting result containing enterprise value.
        non_operating_assets:
            Optional explicit non-operating assets value.
        allow_zero_cash:
            Whether unavailable cash may default to zero.

        Returns
        -------
        ValuationResult
            Complete valuation result including bridge and market comparison.
        """

        assumptions = self._require_assumptions()
        bridge = self.equity_bridge(
            discounting.enterprise_value,
            wacc_result=wacc,
            non_operating_assets=non_operating_assets,
            allow_zero_cash=allow_zero_cash,
        )
        shares = self.shares_outstanding()
        intrinsic_value = self.intrinsic_value_per_share(
            bridge.equity_value,
            shares,
        )
        current_price = self.current_price()
        upside_downside = self.upside_downside(intrinsic_value, current_price)
        self._record_stock_mapping_diagnostics()
        return ValuationResult(
            ticker=self._ticker(),
            valuation_date=assumptions.valuation_date,
            valuation_currency=(
                assumptions.valuation_currency_override
                or self.stock.valuation_currency()
            ),
            forecast_table=forecast_table,
            fcff=fcff,
            growth=growth,
            cost_of_equity=cost_of_equity,
            cost_of_debt=cost_of_debt,
            wacc=wacc,
            terminal_value=terminal_value,
            discounting=discounting,
            enterprise_value=discounting.enterprise_value,
            equity_value=bridge.equity_value,
            intrinsic_value_per_share=intrinsic_value,
            current_price=current_price,
            upside_downside_pct=upside_downside,
            diagnostics=list(self.diagnostics),
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
                source_rows=("Invested Capital",),
            )
            nopat = calculate_nopat(ebit.loc[period], inputs.tax_rate, period=period)
            values.append(nopat / capital)
        return pd.Series(values, index=common_periods, dtype=object)

    def _average_debt_details(self) -> tuple[Decimal, tuple[Any, Any]]:
        debt = self.stock.debt_series().sort_index()
        if len(debt) < 2:
            raise MetricUnavailableError(
                self._ticker(),
                "average debt",
                source_attempted="Stock.debt_series",
                suggested_override="Provide at least two adjacent annual debt observations.",
            )
        periods = (debt.index[-2], debt.index[-1])
        previous_debt = debt.iloc[-2]
        current_debt = debt.iloc[-1]
        if not isinstance(previous_debt, Decimal) or not isinstance(
            current_debt, Decimal
        ):
            raise MetricUnavailableError(
                self._ticker(),
                "average debt",
                source_attempted="Stock.debt_series",
            )
        average_debt = (previous_debt + current_debt) / Decimal("2")
        if not average_debt.is_finite() or average_debt <= 0:
            raise MetricUnavailableError(
                self._ticker(),
                "average debt",
                source_attempted=f"Stock.debt_series periods {periods[0]} and {periods[1]}",
                suggested_override="Provide positive adjacent annual debt values.",
            )
        return average_debt, periods

    def _latest_book_debt(self) -> Decimal:
        debt = self.stock.debt_series().sort_index()
        if debt.empty:
            raise MetricUnavailableError(
                self._ticker(),
                "market value of debt",
                source_attempted="Stock.debt_series",
                suggested_override="Provide market_debt_override.",
            )
        latest_debt = debt.iloc[-1]
        if not isinstance(latest_debt, Decimal) or not latest_debt.is_finite():
            raise MetricUnavailableError(
                self._ticker(),
                "market value of debt",
                source_attempted="Stock.debt_series",
                suggested_override="Provide market_debt_override.",
            )
        return latest_debt

    def _require_assumptions(self) -> ValuationAssumptions:
        if self.assumptions is None:
            raise InvalidAssumptionsError(
                "assumptions",
                "Cost-of-capital calculations require valuation assumptions.",
                suggested_override="Provide ValuationAssumptions with a valuation date.",
            )
        return self.assumptions

    def _ticker(self) -> str:
        return self.stock.normalized_ticker()

    def _record_diagnostic(self, diagnostic: Diagnostic) -> None:
        if diagnostic not in self.diagnostics:
            self.diagnostics.append(diagnostic)

    def _convert_market_amount(
        self,
        amount: Decimal,
        *,
        metric_name: str,
    ) -> Decimal:
        assumptions = self._require_assumptions()
        valuation_currency = (
            assumptions.valuation_currency_override or self.stock.valuation_currency()
        )
        trading_currency_method = getattr(self.stock, "trading_currency", None)
        trading_currency = (
            trading_currency_method()
            if callable(trading_currency_method)
            else valuation_currency
        )
        if trading_currency == valuation_currency:
            return amount
        if self.fx_rate_provider is None:
            raise ProviderUnavailableError(
                "FX rate provider",
                f"{trading_currency}/{valuation_currency} conversion",
                suggested_override=(
                    "Configure an FX provider or use a valuation currency matching "
                    "the trading currency."
                ),
            )
        converted = self.fx_rate_provider.convert(
            amount,
            trading_currency,
            valuation_currency,
            assumptions.valuation_date,
        )
        rate = self.fx_rate_provider.get_rate(
            trading_currency,
            valuation_currency,
            assumptions.valuation_date,
        )
        _validate_decimal_input(
            metric_name, converted, period=assumptions.valuation_date
        )
        _validate_decimal_input("fx_rate", rate, period=assumptions.valuation_date)
        provider_name = type(self.fx_rate_provider).__name__
        self._record_diagnostic(
            Diagnostic(
                kind="provider",
                message=(
                    f"Converted {metric_name} from {trading_currency} to "
                    f"{valuation_currency} at FX rate {rate} on "
                    f"{assumptions.valuation_date}."
                ),
                ticker=self._ticker(),
                metric=metric_name,
                provider=provider_name,
                source_attempted=f"{trading_currency}/{valuation_currency} FX rate",
                metadata={
                    "from_currency": trading_currency,
                    "to_currency": valuation_currency,
                    "rate": rate,
                    "valuation_date": assumptions.valuation_date,
                },
            )
        )
        return converted

    def _record_stock_mapping_diagnostics(self) -> None:
        metadata = getattr(self.stock, "mapping_metadata", ())
        for source in metadata[self._stock_metadata_count :]:
            selected_source = source.selected_source
            if selected_source is None:
                kind = "warning"
                message = f"No source resolved {source.metric_name}."
            elif source.used_override:
                kind = "override"
                message = f"Used override for {source.metric_name}."
            elif source.fallbacks_attempted:
                kind = "fallback"
                message = (
                    f"Resolved {source.metric_name} from {selected_source} after "
                    f"trying {', '.join(source.fallbacks_attempted)}."
                )
            else:
                kind = "source"
                message = f"Resolved {source.metric_name} from {selected_source}."
            self._record_diagnostic(
                Diagnostic(
                    kind=kind,
                    message=message,
                    ticker=self._ticker(),
                    metric=source.metric_name,
                    source_attempted=selected_source,
                    fallbacks_attempted=source.fallbacks_attempted,
                )
            )
        self._stock_metadata_count = len(metadata)


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


def _validate_non_negative_decimal(field_name: str, value: Decimal) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise InvalidAssumptionsError(
            field_name,
            "Equity bridge adjustments must be finite, non-negative Decimals.",
            value=value,
        )


__all__ = [
    "DamodaranValuationProcessor",
    "calculate_nopat",
    "discount_factor",
    "validate_terminal_growth",
    "validate_positive_operating_base",
]
