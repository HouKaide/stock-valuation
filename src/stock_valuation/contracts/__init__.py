"""Shared valuation data contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd

from stock_valuation.errors import InvalidAssumptionsError

RATE_OVERRIDE_FIELDS = (
    "tax_rate_override",
    "equity_risk_premium_override",
    "risk_free_rate_override",
    "terminal_growth_rate_override",
)


@dataclass(frozen=True)
class Diagnostic:
    """Structured diagnostic for sources, fallbacks, providers, overrides, and failures.

    Attributes
    ----------
    kind:
        Diagnostic category such as ``source``, ``fallback``, ``provider``, ``override``, or ``failure``.
    message:
        Human-readable diagnostic message.
    ticker:
        Optional ticker context.
    metric:
        Optional metric name.
    provider:
        Optional provider name.
    source_attempted:
        Optional source attempted for the metric or provider result.
    fallbacks_attempted:
        Fallback sources attempted before the diagnostic was emitted.
    suggested_override:
        Optional user-supplied override that can resolve the diagnostic.
    """

    kind: str
    message: str
    ticker: str | None = None
    metric: str | None = None
    provider: str | None = None
    source_attempted: str | None = None
    fallbacks_attempted: tuple[str, ...] = ()
    suggested_override: str | None = None


@dataclass(frozen=True)
class ValuationAssumptions:
    """User-controlled assumptions and overrides for a valuation run.

    Parameters
    ----------
    valuation_date:
        Date used for market, provider, and assumption resolution.
    forecast_years:
        Number of explicit forecast years. Defaults to 5.
    company_country_override:
        Optional company country override.
    valuation_currency_override:
        Optional valuation currency override.
    beta_override:
        Optional beta override.
    tax_rate_override:
        Optional tax-rate override in decimal representation.
    equity_risk_premium_override:
        Optional equity-risk-premium override in decimal representation.
    risk_free_rate_override:
        Optional risk-free-rate override in decimal representation.
    terminal_growth_rate_override:
        Optional terminal-growth-rate override in decimal representation.
    shares_outstanding_override:
        Optional shares outstanding override.
    market_debt_override:
        Optional market debt override.
    """

    valuation_date: date
    forecast_years: int = 5
    company_country_override: str | None = None
    valuation_currency_override: str | None = None
    beta_override: Decimal | None = None
    tax_rate_override: Decimal | None = None
    equity_risk_premium_override: Decimal | None = None
    risk_free_rate_override: Decimal | None = None
    terminal_growth_rate_override: Decimal | None = None
    shares_outstanding_override: Decimal | None = None
    market_debt_override: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate assumption boundaries."""

        if self.forecast_years <= 0:
            raise InvalidAssumptionsError(
                "forecast_years",
                "Forecast years must be positive.",
                suggested_override="Provide a positive forecast horizon.",
            )

        for field_name in (
            "beta_override",
            "tax_rate_override",
            "equity_risk_premium_override",
            "risk_free_rate_override",
            "terminal_growth_rate_override",
            "shares_outstanding_override",
            "market_debt_override",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_finite_decimal(field_name, value)

        for field_name in RATE_OVERRIDE_FIELDS:
            value = getattr(self, field_name)
            if value is not None and not Decimal("-1") < value < Decimal("1"):
                raise InvalidAssumptionsError(
                    field_name,
                    "Rate overrides must use decimal representation, where 0.10 means 10%.",
                    suggested_override="Use decimal rate notation.",
                )


@dataclass(frozen=True)
class FcffInputs:
    """Normalized inputs needed to calculate free cash flow to the firm.

    Attributes
    ----------
    period:
        Statement period or valuation date context.
    ebit:
        Earnings before interest and taxes.
    tax_rate:
        Corporate tax rate as a decimal fraction.
    depreciation_amortization:
        Depreciation and amortization add-back.
    capex:
        Capital expenditures as a positive outflow.
    change_in_non_cash_working_capital:
        Increase in non-cash working capital as a positive outflow.
    diagnostics:
        Diagnostics attached to the normalized inputs.
    """

    period: Any
    ebit: Decimal
    tax_rate: Decimal
    depreciation_amortization: Decimal
    capex: Decimal
    change_in_non_cash_working_capital: Decimal
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def nopat(self) -> Decimal:
        """Return EBIT after tax.

        Returns
        -------
        Decimal
            Net operating profit after tax.
        """

        return self.ebit * (Decimal("1") - self.tax_rate)


@dataclass(frozen=True)
class FcffResult:
    """FCFF calculation result with enough detail to reproduce the formula."""

    inputs: FcffInputs
    nopat: Decimal
    fcff: Decimal
    calculation_steps: tuple[str, ...]
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class GrowthRegressionResult:
    """Revenue-growth regression result."""

    slope: Decimal
    intercept: Decimal
    sample_size: int
    predicted_next_year_growth: Decimal
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class EstimatedGrowthResult:
    """Estimated-growth result from reinvestment rate and return on capital."""

    reinvestment_rate: Decimal
    return_on_capital: Decimal
    estimated_growth: Decimal
    source_method: str
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class CostOfEquityResult:
    """Cost-of-equity calculation result."""

    risk_free_rate: Decimal
    beta: Decimal
    equity_risk_premium: Decimal
    cost_of_equity: Decimal
    source_details: tuple[str, ...]
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class CostOfDebtResult:
    """Cost-of-debt calculation result."""

    interest_expense: Decimal
    average_debt: Decimal
    pretax_cost_of_debt: Decimal
    tax_rate: Decimal
    after_tax_cost_of_debt: Decimal
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class WaccResult:
    """Weighted average cost of capital calculation result."""

    market_value_of_equity: Decimal
    market_value_of_debt: Decimal
    equity_weight: Decimal
    debt_weight: Decimal
    cost_of_equity: Decimal
    pretax_cost_of_debt: Decimal
    tax_rate: Decimal
    wacc: Decimal
    calculation_steps: tuple[str, ...]
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class TerminalGrowthResult:
    """Terminal-growth source selection result."""

    selected_instrument: str
    yield_value: Decimal
    valuation_date: date
    provider: str
    fallbacks_attempted: tuple[str, ...]
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class TerminalValueResult:
    """Terminal-value calculation result."""

    final_forecast_year_fcff: Decimal
    terminal_growth_rate: Decimal
    next_year_fcff: Decimal
    terminal_value: Decimal
    present_value_terminal_value: Decimal
    calculation_steps: tuple[str, ...]
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class DiscountResult:
    """Discounted forecast cash-flow result."""

    discount_table: pd.DataFrame
    present_value_forecast_fcffs: Decimal
    present_value_terminal_value: Decimal
    enterprise_value: Decimal
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class EquityBridgeResult:
    """Equity bridge from enterprise value to equity value."""

    enterprise_value: Decimal
    debt: Decimal
    cash: Decimal
    non_operating_assets: Decimal
    minority_interest: Decimal
    equity_value: Decimal
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class ValuationResult:
    """Complete structured output for a valuation run."""

    ticker: str
    valuation_date: date
    valuation_currency: str
    forecast_table: pd.DataFrame
    fcff: FcffResult
    growth: EstimatedGrowthResult
    cost_of_equity: CostOfEquityResult
    cost_of_debt: CostOfDebtResult
    wacc: WaccResult
    terminal_value: TerminalValueResult
    discounting: DiscountResult
    enterprise_value: Decimal
    equity_value: Decimal
    intrinsic_value_per_share: Decimal
    current_price: Decimal | None
    upside_downside_pct: Decimal | None
    diagnostics: list[Diagnostic]


def to_json_safe(value: Any) -> Any:
    """Return a JSON-safe representation of a valuation contract value.

    Parameters
    ----------
    value:
        Dataclass, scalar, pandas object, mapping, or sequence to serialize.

    Returns
    -------
    Any
        JSON-safe value with ``Decimal`` values represented as strings.
    """

    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [_json_safe_mapping(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): to_json_safe(item) for key, item in value.to_dict().items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_json_safe(item) for item in value]
    return value


def _validate_finite_decimal(field_name: str, value: Decimal) -> None:
    if not value.is_finite():
        raise InvalidAssumptionsError(
            field_name,
            "Decimal values must be finite.",
            suggested_override="Provide a finite Decimal value.",
        )


def _json_safe_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): to_json_safe(item) for key, item in value.items()}


__all__ = [
    "CostOfDebtResult",
    "CostOfEquityResult",
    "Diagnostic",
    "DiscountResult",
    "EquityBridgeResult",
    "EstimatedGrowthResult",
    "FcffInputs",
    "FcffResult",
    "GrowthRegressionResult",
    "TerminalGrowthResult",
    "TerminalValueResult",
    "ValuationAssumptions",
    "ValuationResult",
    "WaccResult",
    "to_json_safe",
]
