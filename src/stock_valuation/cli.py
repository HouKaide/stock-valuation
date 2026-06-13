"""Command-line entrypoint for the stock valuation application."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from stock_valuation.contracts import (
    Diagnostic,
    ValuationAssumptions,
    ValuationResult,
    to_json_safe,
)
from stock_valuation.diagnostics import diagnostic_to_dict, error_to_dict
from stock_valuation.errors import (
    InvalidAssumptionsError,
    ProviderUnavailableError,
    StockValuationError,
)
from stock_valuation.processor import DamodaranValuationProcessor
from stock_valuation.providers import (
    EquityRiskPremiumProvider,
    FxRateProvider,
    MacroRateProvider,
    MarketDebtProvider,
    ProviderConfig,
    SovereignYieldProvider,
    TaxRateProvider,
)
from stock_valuation.stock import Stock
from stock_valuation.yfinance_client import YFinanceClient

ASSUMPTION_FIELDS = {field.name for field in fields(ValuationAssumptions)}
DECIMAL_FILE_FIELDS = {
    "beta_override",
    "tax_rate_override",
    "equity_risk_premium_override",
    "risk_free_rate_override",
    "terminal_growth_rate_override",
    "shares_outstanding_override",
    "market_debt_override",
}
CLI_TO_ASSUMPTION_FIELD = {
    "forecast_years": "forecast_years",
    "company_country": "company_country_override",
    "valuation_currency": "valuation_currency_override",
    "beta": "beta_override",
    "tax_rate": "tax_rate_override",
    "erp": "equity_risk_premium_override",
    "risk_free_rate": "risk_free_rate_override",
    "terminal_growth_rate": "terminal_growth_rate_override",
    "shares_outstanding": "shares_outstanding_override",
    "market_debt": "market_debt_override",
}
PROVIDER_KINDS = {
    "macro",
    "erp",
    "fx",
    "tax_rate",
    "market_debt",
    "sovereign_yield",
}


class ValuationProcessor(Protocol):
    """Protocol for an executable valuation workflow."""

    def value(self) -> ValuationResult:
        """Return a complete valuation result."""


ProviderFactory = Callable[[ProviderConfig], object]


@dataclass(frozen=True)
class ProviderBundle:
    """Configured providers passed to the valuation processor."""

    macro_provider: MacroRateProvider | None = None
    erp_provider: EquityRiskPremiumProvider | None = None
    tax_rate_provider: TaxRateProvider | None = None
    market_debt_provider: MarketDebtProvider | None = None
    sovereign_yield_provider: SovereignYieldProvider | None = None
    fx_rate_provider: FxRateProvider | None = None


@dataclass(frozen=True)
class LoadedConfiguration:
    """Assumption values and provider references loaded from JSON."""

    assumptions: Mapping[str, Any]
    providers: Mapping[str, ProviderConfig]


def decimal_value(value: str) -> Decimal:
    """Parse a finite Decimal command-line value.

    Parameters
    ----------
    value:
        Raw command-line value.

    Returns
    -------
    Decimal
        Parsed finite decimal.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a finite decimal.
    """

    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value!r}") from error
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"decimal value must be finite: {value!r}")
    return parsed


def positive_integer(value: str) -> int:
    """Parse a positive integer command-line value."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_empty_string(value: str) -> str:
    """Parse a non-empty string command-line value."""

    parsed = value.strip()
    if not parsed:
        raise argparse.ArgumentTypeError("value must not be empty")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the stock valuation argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the ``stock-valuation`` command.
    """

    parser = argparse.ArgumentParser(prog="stock-valuation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    value_parser = subparsers.add_parser("value", help="value a stock")
    value_parser.add_argument("ticker", type=non_empty_string)
    value_parser.add_argument("--assumptions-file", type=Path)
    value_parser.add_argument("--forecast-years", type=positive_integer)
    value_parser.add_argument("--company-country", type=non_empty_string)
    value_parser.add_argument("--valuation-currency", type=non_empty_string)
    value_parser.add_argument("--beta", type=decimal_value)
    value_parser.add_argument("--tax-rate", type=decimal_value)
    value_parser.add_argument("--erp", type=decimal_value)
    value_parser.add_argument("--risk-free-rate", type=decimal_value)
    value_parser.add_argument("--terminal-growth-rate", type=decimal_value)
    value_parser.add_argument("--shares-outstanding", type=decimal_value)
    value_parser.add_argument("--market-debt", type=decimal_value)
    value_parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def load_assumptions_file(path: Path) -> LoadedConfiguration:
    """Load strict JSON assumptions and provider references.

    Parameters
    ----------
    path:
        JSON file containing assumption fields and an optional ``providers``
        object.

    Returns
    -------
    LoadedConfiguration
        Normalized assumptions and secret-free provider references.

    Raises
    ------
    InvalidAssumptionsError
        If the file is missing, malformed, or contains unsupported fields.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InvalidAssumptionsError(
            "assumptions_file",
            f"Could not read assumptions file {path}.",
            suggested_override="Provide a readable JSON assumptions file.",
        ) from error
    except json.JSONDecodeError as error:
        raise InvalidAssumptionsError(
            "assumptions_file",
            f"Assumptions file {path} is not valid JSON.",
            suggested_override="Provide a valid JSON object.",
        ) from error
    if not isinstance(raw, dict):
        raise InvalidAssumptionsError(
            "assumptions_file",
            "The assumptions file root must be a JSON object.",
        )

    unknown = set(raw) - ASSUMPTION_FIELDS - {"providers"}
    if unknown:
        raise InvalidAssumptionsError(
            "assumptions_file",
            f"Unknown fields: {', '.join(sorted(unknown))}.",
        )
    assumptions = {
        key: _normalize_file_assumption(key, value)
        for key, value in raw.items()
        if key != "providers"
    }
    providers = _parse_provider_references(raw.get("providers", {}))
    return LoadedConfiguration(assumptions=assumptions, providers=providers)


def merge_assumptions(
    arguments: argparse.Namespace,
    file_values: Mapping[str, Any] | None = None,
    *,
    valuation_date: date | None = None,
) -> tuple[ValuationAssumptions, list[Diagnostic]]:
    """Merge defaults, file values, and explicit CLI options."""

    values: dict[str, Any] = {
        "valuation_date": valuation_date or date.today(),
        "forecast_years": 5,
    }
    diagnostics: list[Diagnostic] = []
    for field_name, value in (file_values or {}).items():
        values[field_name] = value
        diagnostics.append(_override_diagnostic(field_name, "assumptions file"))
    for argument_name, field_name in CLI_TO_ASSUMPTION_FIELD.items():
        value = getattr(arguments, argument_name, None)
        if value is not None:
            values[field_name] = value
            diagnostics.append(_override_diagnostic(field_name, "command line"))
    return ValuationAssumptions(**values), diagnostics


def configure_providers(
    file_references: Mapping[str, ProviderConfig] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    factories: Mapping[str, Mapping[str, ProviderFactory]] | None = None,
) -> ProviderBundle:
    """Configure provider objects from JSON references and environment values."""

    environment = environ if environ is not None else os.environ
    references = dict(file_references or {})
    for kind in PROVIDER_KINDS:
        prefix = f"STOCK_VALUATION_{kind.upper()}_PROVIDER"
        name = environment.get(prefix)
        if name:
            references[kind] = ProviderConfig(
                name=name,
                base_url=environment.get(f"{prefix}_BASE_URL"),
                api_key_env_var=environment.get(f"{prefix}_API_KEY_ENV_VAR"),
            )

    configured: dict[str, object] = {}
    for kind, reference in references.items():
        factory = (factories or {}).get(kind, {}).get(reference.name)
        if factory is None:
            raise ProviderUnavailableError(
                reference.name,
                f"{kind} provider configuration",
                suggested_override=(
                    f"Install or register provider '{reference.name}' for {kind}, "
                    "or supply the corresponding valuation override."
                ),
            )
        if reference.api_key_env_var and not environment.get(reference.api_key_env_var):
            raise ProviderUnavailableError(
                reference.name,
                f"{kind} provider configuration",
                suggested_override=(
                    f"Set environment variable {reference.api_key_env_var} "
                    "or supply the corresponding valuation override."
                ),
            )
        configured[kind] = factory(reference)

    return ProviderBundle(
        macro_provider=cast(MacroRateProvider | None, configured.get("macro")),
        erp_provider=cast(EquityRiskPremiumProvider | None, configured.get("erp")),
        tax_rate_provider=cast(TaxRateProvider | None, configured.get("tax_rate")),
        market_debt_provider=cast(
            MarketDebtProvider | None, configured.get("market_debt")
        ),
        sovereign_yield_provider=cast(
            SovereignYieldProvider | None, configured.get("sovereign_yield")
        ),
        fx_rate_provider=cast(FxRateProvider | None, configured.get("fx")),
    )


def build_valuation_workflow(
    ticker: str,
    assumptions: ValuationAssumptions,
    providers: ProviderBundle,
    diagnostics: Sequence[Diagnostic] = (),
) -> DamodaranValuationProcessor:
    """Construct the yfinance client, stock model, and valuation processor."""

    client = YFinanceClient(ticker)
    stock = Stock(
        ticker,
        yfinance_client=client,
        tax_rate_provider=providers.tax_rate_provider,
        tax_rate=assumptions.tax_rate_override,
    )
    processor = DamodaranValuationProcessor(
        stock=stock,
        assumptions=assumptions,
        macro_provider=providers.macro_provider,
        erp_provider=providers.erp_provider,
        tax_rate_provider=providers.tax_rate_provider,
        market_debt_provider=providers.market_debt_provider,
        sovereign_yield_provider=providers.sovereign_yield_provider,
        fx_rate_provider=providers.fx_rate_provider,
    )
    processor.diagnostics.extend(diagnostics)
    return processor


def render_human(result: ValuationResult) -> str:
    """Render a complete human-readable valuation report."""

    fcff_inputs = result.fcff.inputs
    lines = [
        "Company Summary",
        f"Ticker: {result.ticker}",
        f"Valuation date: {result.valuation_date.isoformat()}",
        f"Valuation currency: {result.valuation_currency}",
        "",
        "FCFF",
        f"EBIT: {fcff_inputs.ebit}",
        f"Tax rate: {fcff_inputs.tax_rate}",
        f"Depreciation and amortization: {fcff_inputs.depreciation_amortization}",
        f"Capital expenditure: {fcff_inputs.capex}",
        f"Working capital change: {fcff_inputs.change_in_non_cash_working_capital}",
        f"Calculated FCFF: {result.fcff.fcff}",
        "",
        "Growth",
        f"Method: {result.growth.source_method}",
        f"Estimated growth: {result.growth.estimated_growth}",
        "",
        "Cost of Equity",
        f"Risk-free rate: {result.cost_of_equity.risk_free_rate}",
        f"Beta: {result.cost_of_equity.beta}",
        f"Equity risk premium: {result.cost_of_equity.equity_risk_premium}",
        f"Cost of equity: {result.cost_of_equity.cost_of_equity}",
        "",
        "Cost of Debt",
        f"Pre-tax cost of debt: {result.cost_of_debt.pretax_cost_of_debt}",
        f"After-tax cost of debt: {result.cost_of_debt.after_tax_cost_of_debt}",
        "",
        "WACC",
        f"Equity weight: {result.wacc.equity_weight}",
        f"Debt weight: {result.wacc.debt_weight}",
        f"WACC: {result.wacc.wacc}",
        "",
        "Forecast FCFF",
        result.forecast_table.to_string(index=False),
        "",
        "Terminal Value",
        f"Terminal growth rate: {result.terminal_value.terminal_growth_rate}",
        f"Terminal value: {result.terminal_value.terminal_value}",
        "",
        "Discounting",
        result.discounting.discount_table.to_string(index=False),
        "",
        f"Enterprise value: {result.enterprise_value}",
        "Equity Bridge",
        f"Enterprise value: {result.enterprise_value}",
        f"Equity value: {result.equity_value}",
        f"Intrinsic value per share: {result.intrinsic_value_per_share}",
    ]
    if result.current_price is not None:
        lines.append(f"Current price: {result.current_price}")
    if result.upside_downside_pct is not None:
        lines.append(f"Upside/downside: {result.upside_downside_pct}")
    lines.extend(["", "Diagnostics"])
    lines.extend(
        f"[{diagnostic.kind}] {diagnostic.message}" for diagnostic in result.diagnostics
    )
    return "\n".join(lines)


def render_json(result: ValuationResult) -> str:
    """Render a valuation result as stable JSON."""

    return json.dumps(to_json_safe(result), indent=2, sort_keys=True)


def render_error(error: StockValuationError, *, json_output: bool) -> str:
    """Render a typed project error without tracebacks or secrets."""

    diagnostic = Diagnostic(
        kind="failure",
        message=error.safe_message,
        ticker=error.ticker,
        metric=error.metric,
        provider=error.provider,
        source_attempted=error.source_attempted,
        fallbacks_attempted=error.fallbacks_attempted,
        suggested_override=error.suggested_override,
        metadata={"error_type": type(error).__name__},
    )
    if json_output:
        return json.dumps(
            {
                "error": error_to_dict(error),
                "diagnostics": [diagnostic_to_dict(diagnostic)],
            },
            indent=2,
            sort_keys=True,
        )
    lines = [f"Error: {error.safe_message}"]
    if error.suggested_override:
        lines.append(f"Suggested action: {error.suggested_override}")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    workflow_builder: Callable[
        [str, ValuationAssumptions, ProviderBundle, Sequence[Diagnostic]],
        ValuationProcessor,
    ] = build_valuation_workflow,
    provider_factories: Mapping[str, Mapping[str, ProviderFactory]] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv:
        Optional arguments excluding the executable name.
    stdout:
        Optional success output stream.
    stderr:
        Optional error output stream.
    workflow_builder:
        Injectable valuation workflow constructor.
    provider_factories:
        Injectable provider factory registry.
    environ:
        Injectable environment mapping.

    Returns
    -------
    int
        Process exit code.
    """

    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    arguments = build_parser().parse_args(argv)
    loaded = LoadedConfiguration(assumptions={}, providers={})
    try:
        if arguments.assumptions_file is not None:
            loaded = load_assumptions_file(arguments.assumptions_file)
        assumptions, diagnostics = merge_assumptions(
            arguments,
            loaded.assumptions,
        )
        providers = configure_providers(
            loaded.providers,
            environ=environ,
            factories=provider_factories,
        )
        processor = workflow_builder(
            arguments.ticker,
            assumptions,
            providers,
            diagnostics,
        )
        result = processor.value()
    except StockValuationError as error:
        print(
            render_error(error, json_output=arguments.json_output),
            file=error_output,
        )
        return 1

    renderer = render_json if arguments.json_output else render_human
    print(renderer(result), file=output)
    return 0


def _normalize_file_assumption(field_name: str, value: Any) -> Any:
    if field_name == "valuation_date":
        if not isinstance(value, str):
            raise InvalidAssumptionsError(
                field_name,
                "valuation_date must be an ISO date string.",
            )
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise InvalidAssumptionsError(
                field_name,
                "valuation_date must use YYYY-MM-DD format.",
            ) from error
    if field_name in DECIMAL_FILE_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise InvalidAssumptionsError(
                field_name,
                "Decimal assumptions must be JSON strings or numbers.",
            )
        try:
            parsed = Decimal(str(value))
        except InvalidOperation as error:
            raise InvalidAssumptionsError(
                field_name,
                "Decimal assumption is invalid.",
            ) from error
        if not parsed.is_finite():
            raise InvalidAssumptionsError(
                field_name,
                "Decimal assumption must be finite.",
            )
        return parsed
    return value


def _parse_provider_references(value: Any) -> dict[str, ProviderConfig]:
    if not isinstance(value, dict):
        raise InvalidAssumptionsError(
            "providers",
            "Provider configuration must be a JSON object.",
        )
    unknown = set(value) - PROVIDER_KINDS
    if unknown:
        raise InvalidAssumptionsError(
            "providers",
            f"Unknown provider kinds: {', '.join(sorted(unknown))}.",
        )
    providers: dict[str, ProviderConfig] = {}
    for kind, raw_reference in value.items():
        if not isinstance(raw_reference, dict):
            raise InvalidAssumptionsError(
                "providers",
                f"Provider reference for {kind} must be a JSON object.",
            )
        unknown_fields = set(raw_reference) - {
            "name",
            "base_url",
            "api_key_env_var",
        }
        name = raw_reference.get("name")
        base_url = raw_reference.get("base_url")
        api_key_env_var = raw_reference.get("api_key_env_var")
        if (
            unknown_fields
            or not isinstance(name, str)
            or not name.strip()
            or (base_url is not None and not isinstance(base_url, str))
            or (api_key_env_var is not None and not isinstance(api_key_env_var, str))
        ):
            raise InvalidAssumptionsError(
                "providers",
                f"Invalid provider reference for {kind}.",
            )
        providers[kind] = ProviderConfig(
            name=name.strip(),
            base_url=base_url,
            api_key_env_var=api_key_env_var,
        )
    return providers


def _override_diagnostic(field_name: str, source: str) -> Diagnostic:
    return Diagnostic(
        kind="override",
        message=f"Applied {field_name} from {source}.",
        metric=field_name,
        source_attempted=source,
    )


if __name__ == "__main__":
    raise SystemExit(main())
