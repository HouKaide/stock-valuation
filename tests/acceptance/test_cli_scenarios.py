"""Acceptance scenarios for human and JSON CLI output."""

from __future__ import annotations

import io
import json

from stock_valuation.cli import ProviderBundle, main
from stock_valuation.contracts import Diagnostic, ValuationAssumptions, ValuationResult

from .conftest import CompanyDataset, build_processor


class ResultProcessor:
    """Return one deterministic valuation result."""

    def __init__(self, result: ValuationResult) -> None:
        """Initialize the processor.

        Parameters
        ----------
        result:
            Result returned by ``value``.
        """

        self.result = result

    def value(self) -> ValuationResult:
        """Return the configured valuation result."""

        return self.result


def test_cli_human_and_json_modes_have_output_parity(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Both CLI modes should expose the same core valuation result."""

    result = build_processor(usd_company_dataset).value()

    def workflow_builder(
        ticker: str,
        assumptions: ValuationAssumptions,
        providers: ProviderBundle,
        diagnostics: list[Diagnostic],
    ) -> ResultProcessor:
        """Return the shared deterministic result."""

        return ResultProcessor(result)

    human_output = io.StringIO()
    json_output = io.StringIO()

    human_exit = main(
        ["value", result.ticker],
        stdout=human_output,
        workflow_builder=workflow_builder,
        environ={},
    )
    json_exit = main(
        ["value", result.ticker, "--json"],
        stdout=json_output,
        workflow_builder=workflow_builder,
        environ={},
    )
    payload = json.loads(json_output.getvalue())
    human = human_output.getvalue()

    assert human_exit == 0
    assert json_exit == 0
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
        "Diagnostics",
    ):
        assert section in human
    assert result.ticker in human
    assert result.valuation_date.isoformat() in human
    assert result.valuation_currency in human
    assert str(result.intrinsic_value_per_share) in human
    assert payload["ticker"] == result.ticker
    assert payload["valuation_date"] == result.valuation_date.isoformat()
    assert payload["valuation_currency"] == result.valuation_currency
    assert payload["intrinsic_value_per_share"] == str(result.intrinsic_value_per_share)
    assert isinstance(payload["enterprise_value"], str)
    assert payload["diagnostics"]
