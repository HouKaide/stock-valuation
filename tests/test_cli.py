"""Tests for the stock valuation command-line contract."""

from __future__ import annotations

import io
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stock_valuation.cli import (
    ProviderBundle,
    build_parser,
    configure_providers,
    load_assumptions_file,
    main,
    merge_assumptions,
    render_human,
)
from stock_valuation.contracts import (
    CostOfDebtResult,
    CostOfEquityResult,
    Diagnostic,
    DiscountResult,
    EstimatedGrowthResult,
    FcffInputs,
    FcffResult,
    TerminalValueResult,
    ValuationResult,
    WaccResult,
)
from stock_valuation.errors import (
    InvalidAssumptionsError,
    MetricUnavailableError,
    ProviderUnavailableError,
    UnsupportedCurrencyError,
)
from stock_valuation.providers import ProviderConfig


class FakeProcessor:
    """Fake executable valuation processor."""

    def __init__(
        self,
        result: ValuationResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or valuation_result()
        self.error = error

    def value(self) -> ValuationResult:
        """Return a canned result or raise a canned error."""

        if self.error is not None:
            raise self.error
        return self.result


def valuation_result() -> ValuationResult:
    """Build a compact complete valuation result."""

    diagnostic = Diagnostic(kind="calculation", message="Used deterministic data.")
    inputs = FcffInputs(
        period="2025",
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
        source_details=("override",),
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
    wacc = WaccResult(
        market_value_of_equity=Decimal("800"),
        market_value_of_debt=Decimal("200"),
        equity_weight=Decimal("0.8"),
        debt_weight=Decimal("0.2"),
        cost_of_equity=Decimal("0.095"),
        pretax_cost_of_debt=Decimal("0.05"),
        tax_rate=Decimal("0.20"),
        wacc=Decimal("0.084"),
        calculation_steps=("WACC",),
        diagnostics=[diagnostic],
    )
    terminal = TerminalValueResult(
        final_forecast_year_fcff=Decimal("90"),
        terminal_growth_rate=Decimal("0.03"),
        next_year_fcff=Decimal("92.7"),
        terminal_value=Decimal("1716.67"),
        present_value_terminal_value=Decimal("1200"),
        calculation_steps=("Terminal",),
        diagnostics=[diagnostic],
    )
    discount_table = pd.DataFrame(
        [
            {
                "year": 1,
                "fcff": Decimal("73.5"),
                "discount_factor": Decimal("0.92"),
                "present_value": Decimal("67.62"),
                "cash_flow_type": "forecast",
            }
        ]
    )
    discounting = DiscountResult(
        discount_table=discount_table,
        present_value_forecast_fcffs=Decimal("300"),
        present_value_terminal_value=Decimal("1200"),
        enterprise_value=Decimal("1500"),
        diagnostics=[diagnostic],
    )
    return ValuationResult(
        ticker="AAPL",
        valuation_date=date(2026, 6, 13),
        valuation_currency="USD",
        forecast_table=pd.DataFrame([{"year": 1, "fcff": Decimal("73.5")}]),
        fcff=fcff,
        growth=growth,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        wacc=wacc,
        terminal_value=terminal,
        discounting=discounting,
        enterprise_value=Decimal("1500"),
        equity_value=Decimal("1340"),
        intrinsic_value_per_share=Decimal("89.33"),
        current_price=Decimal("80"),
        upside_downside_pct=Decimal("0.116625"),
        diagnostics=[diagnostic],
    )


def test_parser_supports_all_value_options() -> None:
    """All documented options should parse into normalized values."""

    arguments = build_parser().parse_args(
        [
            "value",
            " aapl ",
            "--forecast-years",
            "7",
            "--company-country",
            "United States",
            "--valuation-currency",
            "USD",
            "--beta",
            "1.2",
            "--tax-rate",
            "0.21",
            "--erp",
            "0.05",
            "--risk-free-rate",
            "0.04",
            "--terminal-growth-rate",
            "0.025",
            "--shares-outstanding",
            "100",
            "--market-debt",
            "50",
            "--json",
        ]
    )

    assert arguments.ticker == "aapl"
    assert arguments.forecast_years == 7
    assert arguments.beta == Decimal("1.2")
    assert arguments.tax_rate == Decimal("0.21")
    assert arguments.erp == Decimal("0.05")
    assert arguments.risk_free_rate == Decimal("0.04")
    assert arguments.terminal_growth_rate == Decimal("0.025")
    assert arguments.shares_outstanding == Decimal("100")
    assert arguments.market_debt == Decimal("50")
    assert arguments.json_output is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["value"],
        ["value", "AAPL", "--unknown"],
        ["value", "AAPL", "--beta", "not-a-number"],
        ["value", "AAPL", "--forecast-years", "0"],
        ["value", "   "],
    ],
)
def test_parser_rejects_invalid_input(arguments: list[str]) -> None:
    """Invalid syntax and values should fail before valuation starts."""

    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_load_assumptions_file_normalizes_values(tmp_path: Path) -> None:
    """Strict JSON files should load assumptions and provider references."""

    path = tmp_path / "assumptions.json"
    path.write_text(
        json.dumps(
            {
                "valuation_date": "2026-06-13",
                "forecast_years": 6,
                "tax_rate_override": "0.21",
                "providers": {
                    "macro": {
                        "name": "example",
                        "base_url": "https://example.test",
                        "api_key_env_var": "EXAMPLE_API_KEY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_assumptions_file(path)

    assert loaded.assumptions["valuation_date"] == date(2026, 6, 13)
    assert loaded.assumptions["tax_rate_override"] == Decimal("0.21")
    assert loaded.providers["macro"] == ProviderConfig(
        name="example",
        base_url="https://example.test",
        api_key_env_var="EXAMPLE_API_KEY",
    )


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("missing.json", None),
        ("malformed.json", "{"),
        ("unknown.json", '{"unknown": 1}'),
        ("providers.json", '{"providers": {"unknown": {"name": "x"}}}'),
    ],
)
def test_load_assumptions_file_rejects_invalid_files(
    tmp_path: Path,
    filename: str,
    content: str | None,
) -> None:
    """Missing, malformed, and unsupported file content should be typed errors."""

    path = tmp_path / filename
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(InvalidAssumptionsError):
        load_assumptions_file(path)


def test_cli_flags_override_file_values_only_when_supplied() -> None:
    """Explicit CLI options should take precedence over file values."""

    parser = build_parser()
    file_values = {
        "valuation_date": date(2026, 1, 2),
        "forecast_years": 8,
        "beta_override": Decimal("0.9"),
    }

    file_only, file_diagnostics = merge_assumptions(
        parser.parse_args(["value", "AAPL"]),
        file_values,
    )
    overridden, cli_diagnostics = merge_assumptions(
        parser.parse_args(["value", "AAPL", "--forecast-years", "3", "--beta", "1.2"]),
        file_values,
    )

    assert file_only.forecast_years == 8
    assert file_only.beta_override == Decimal("0.9")
    assert overridden.forecast_years == 3
    assert overridden.beta_override == Decimal("1.2")
    assert len(file_diagnostics) == 3
    assert len(cli_diagnostics) == 5


def test_configure_providers_uses_references_without_exposing_secret() -> None:
    """Provider factories should receive references after secret validation."""

    received: list[ProviderConfig] = []

    def factory(config: ProviderConfig) -> object:
        received.append(config)
        return object()

    bundle = configure_providers(
        {
            "macro": ProviderConfig(
                name="example",
                api_key_env_var="EXAMPLE_API_KEY",
            )
        },
        environ={"EXAMPLE_API_KEY": "super-secret"},
        factories={"macro": {"example": factory}},
    )

    assert bundle.macro_provider is not None
    assert received == [
        ProviderConfig(name="example", api_key_env_var="EXAMPLE_API_KEY")
    ]
    assert "super-secret" not in repr(bundle)


def test_configure_providers_reports_missing_setup_without_secret() -> None:
    """Missing provider configuration should produce actionable safe guidance."""

    with pytest.raises(ProviderUnavailableError) as captured:
        configure_providers(
            {
                "macro": ProviderConfig(
                    name="example",
                    api_key_env_var="EXAMPLE_API_KEY",
                )
            },
            environ={"EXAMPLE_API_KEY": "super-secret"},
        )

    assert "example" in str(captured.value)
    assert "super-secret" not in str(captured.value)


def test_human_renderer_contains_major_calculation_sections() -> None:
    """Human output should expose the complete calculation outline."""

    output = render_human(valuation_result())

    for section in (
        "Company Summary",
        "FCFF",
        "Growth",
        "Cost of Equity",
        "Cost of Debt",
        "WACC",
        "Forecast FCFF",
        "Terminal Value",
        "Discounting",
        "Equity Bridge",
        "Intrinsic value per share",
        "Diagnostics",
    ):
        assert section in output


def test_main_runs_fake_workflow_and_renders_json() -> None:
    """The entry point should pass normalized inputs to an injected workflow."""

    stdout = io.StringIO()
    captured: dict[str, Any] = {}

    def builder(
        ticker: str,
        assumptions: Any,
        providers: ProviderBundle,
        diagnostics: Any,
    ) -> FakeProcessor:
        captured.update(
            ticker=ticker,
            assumptions=assumptions,
            providers=providers,
            diagnostics=diagnostics,
        )
        return FakeProcessor()

    exit_code = main(
        [
            "value",
            "AAPL",
            "--forecast-years",
            "4",
            "--tax-rate",
            "0.21",
            "--json",
        ],
        stdout=stdout,
        workflow_builder=builder,
        environ={},
    )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert captured["ticker"] == "AAPL"
    assert captured["assumptions"].forecast_years == 4
    assert captured["assumptions"].tax_rate_override == Decimal("0.21")
    assert payload["ticker"] == "AAPL"
    assert payload["intrinsic_value_per_share"] == "89.33"
    assert payload["diagnostics"]


@pytest.mark.parametrize(
    "error",
    [
        MetricUnavailableError("AAPL", "EBIT"),
        ProviderUnavailableError(
            "macro",
            "risk-free rate",
            suggested_override="Set MACRO_API_KEY.",
        ),
        InvalidAssumptionsError("tax_rate", "Invalid rate."),
        UnsupportedCurrencyError("XYZ", "valuation currency"),
    ],
)
def test_main_renders_typed_errors_in_human_and_json_modes(
    error: Exception,
) -> None:
    """Representative typed failures should be concise in both modes."""

    def builder(*args: Any) -> FakeProcessor:
        return FakeProcessor(error=error)

    human_stderr = io.StringIO()
    json_stderr = io.StringIO()

    human_exit = main(
        ["value", "AAPL"],
        stderr=human_stderr,
        workflow_builder=builder,
        environ={},
    )
    json_exit = main(
        ["value", "AAPL", "--json"],
        stderr=json_stderr,
        workflow_builder=builder,
        environ={},
    )

    assert human_exit == 1
    assert "Error:" in human_stderr.getvalue()
    assert "Traceback" not in human_stderr.getvalue()
    assert json_exit == 1
    payload = json.loads(json_stderr.getvalue())
    assert payload["error"]["type"] == type(error).__name__
    assert payload["diagnostics"][0]["category"] == "failure"
