"""Integrated acceptance scenarios for valuation and fallback behavior."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pandas as pd
import pytest

from stock_valuation.contracts import ValuationAssumptions, ValuationResult
from stock_valuation.errors import (
    InvalidAssumptionsError,
    MetricUnavailableError,
    ProviderUnavailableError,
    StatementUnavailableError,
    UnsupportedCurrencyError,
)
from stock_valuation.stock import Stock

from .conftest import (
    VALUATION_DATE,
    CompanyDataset,
    FakeFxRateProvider,
    FakeEquityRiskPremiumProvider,
    FakeMacroRateProvider,
    FakeMarketDebtProvider,
    FakeSovereignYieldProvider,
    FakeTaxRateProvider,
    FakeYFinanceClient,
    build_processor,
    sovereign_candidate,
)


def test_deterministic_fixture_set_is_complete(
    usd_company_dataset: CompanyDataset,
    eur_company_dataset: CompanyDataset,
    currency_mismatch_dataset: CompanyDataset,
) -> None:
    """All reusable datasets should expose complete annual statements."""

    for dataset in (
        usd_company_dataset,
        eur_company_dataset,
        currency_mismatch_dataset,
    ):
        assert len(dataset.income_statement.columns) == 3
        assert len(dataset.balance_sheet.columns) == 3
        assert len(dataset.cashflow.columns) == 3


def test_successful_usd_valuation_produces_complete_result(
    usd_company_dataset: CompanyDataset,
) -> None:
    """USD scenario should complete with provider and terminal diagnostics."""

    result = build_processor(usd_company_dataset).value()

    assert isinstance(result, ValuationResult)
    assert result.valuation_currency == "USD"
    assert result.cost_of_equity.risk_free_rate == Decimal("0.04")
    assert result.terminal_value.terminal_growth_rate == Decimal("0.025")
    assert result.fcff.fcff > 0
    assert result.wacc.wacc > result.terminal_value.terminal_growth_rate
    assert result.enterprise_value > 0
    assert result.equity_value > 0
    assert result.intrinsic_value_per_share > 0
    assert any(
        diagnostic.source_attempted == "US10Y" for diagnostic in result.diagnostics
    )


def test_eur_valuation_selects_german_bund(
    eur_company_dataset: CompanyDataset,
) -> None:
    """EUR scenario should select the German ten-year Bund deterministically."""

    provider = FakeSovereignYieldProvider(
        (
            sovereign_candidate("FR10Y", "EUR", "France", "0.031", confidence="0.98"),
            sovereign_candidate("DE10Y", "EUR", "Germany", "0.024"),
        )
    )
    result = build_processor(
        eur_company_dataset,
        sovereign_provider=provider,
    ).value()

    assert result.valuation_currency == "EUR"
    assert result.terminal_value.terminal_growth_rate == Decimal("0.024")
    assert result.terminal_value.terminal_growth_rate < result.wacc.wacc
    assert any(
        diagnostic.source_attempted == "DE10Y"
        and str(VALUATION_DATE) in diagnostic.message
        for diagnostic in result.diagnostics
    )
    assert any(
        "FR10Y" in diagnostic.fallbacks_attempted for diagnostic in result.diagnostics
    )


def test_currency_mismatch_converts_market_outputs(
    currency_mismatch_dataset: CompanyDataset,
) -> None:
    """GBP market data should be converted while operating values remain USD."""

    fx_provider = FakeFxRateProvider(rate=Decimal("1.25"))
    result = build_processor(
        currency_mismatch_dataset,
        fx_provider=fx_provider,
    ).value()

    assert result.valuation_currency == "USD"
    assert result.fcff.fcff == Decimal("103.00")
    assert result.wacc.market_value_of_equity == Decimal("1000.00")
    assert result.current_price == Decimal("80.00")
    assert result.enterprise_value > 0
    assert result.upside_downside_pct == (
        result.intrinsic_value_per_share / Decimal("80.00")
    ) - Decimal("1")
    fx_diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.provider == "FakeFxRateProvider"
    ]
    assert {diagnostic.metric for diagnostic in fx_diagnostics} == {
        "market value of equity",
        "current price",
    }
    assert all(
        diagnostic.metadata["rate"] == Decimal("1.25") for diagnostic in fx_diagnostics
    )
    assert all(
        diagnostic.metadata["valuation_date"] == VALUATION_DATE
        for diagnostic in fx_diagnostics
    )


def test_ebit_fallback_continues_and_missing_proxies_fail(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Operating income should replace EBIT, while no proxies should fail."""

    fallback_income = usd_company_dataset.income_statement.rename(
        index={"EBIT": "Operating Income"}
    )
    fallback_result = build_processor(
        replace(usd_company_dataset, income_statement=fallback_income)
    ).value()

    assert fallback_result.fcff.inputs.ebit == Decimal("150")
    assert any(
        diagnostic.metric == "EBIT"
        and diagnostic.source_attempted == "income_statement.Operating Income"
        and diagnostic.kind == "fallback"
        for diagnostic in fallback_result.diagnostics
    )

    missing_income = fallback_income.drop(index="Operating Income")
    with pytest.raises(MetricUnavailableError):
        build_processor(
            replace(usd_company_dataset, income_statement=missing_income)
        ).value()


def test_depreciation_fallback_continues_and_missing_sources_fail(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Reconciled depreciation should replace a missing cash-flow D&A row."""

    cashflow = usd_company_dataset.cashflow.drop(index="Depreciation And Amortization")
    income = usd_company_dataset.income_statement.copy()
    income.loc["Reconciled Depreciation"] = [10, 11, 12]
    result = build_processor(
        replace(
            usd_company_dataset,
            cashflow=cashflow,
            income_statement=income,
        )
    ).value()

    assert result.fcff.inputs.depreciation_amortization == Decimal("12")
    assert any(
        diagnostic.metric == "depreciation and amortization"
        and diagnostic.source_attempted == "income_statement.Reconciled Depreciation"
        for diagnostic in result.diagnostics
    )

    with pytest.raises(MetricUnavailableError):
        build_processor(replace(usd_company_dataset, cashflow=cashflow)).value()


def test_working_capital_derivation_continues_and_missing_inputs_fail(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Balance-sheet working capital should replace a missing cash-flow row."""

    cashflow = usd_company_dataset.cashflow.drop(index="Change In Working Capital")
    result = build_processor(replace(usd_company_dataset, cashflow=cashflow)).value()

    assert result.fcff.inputs.change_in_non_cash_working_capital == Decimal("7")
    assert any(
        diagnostic.metric == "change in non-cash working capital"
        and diagnostic.source_attempted
        == "derived: non-cash current assets - non-debt current liabilities"
        for diagnostic in result.diagnostics
    )

    incomplete_balance = usd_company_dataset.balance_sheet.drop(index="Current Assets")
    with pytest.raises(MetricUnavailableError):
        build_processor(
            replace(
                usd_company_dataset,
                cashflow=cashflow,
                balance_sheet=incomplete_balance,
            )
        ).value()


def test_market_debt_none_falls_back_to_book_debt(
    usd_company_dataset: CompanyDataset,
) -> None:
    """A provider without market debt should use latest book debt."""

    result = build_processor(
        usd_company_dataset,
        debt_provider=FakeMarketDebtProvider(debt=None),
    ).value()

    assert result.wacc.market_value_of_debt == Decimal("120")
    assert any(
        diagnostic.metric == "market value of debt"
        and diagnostic.kind == "fallback"
        and "returned no value" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_shares_history_fallback_and_missing_history_failure(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Latest shares history should replace missing metadata shares."""

    shares = pd.Series(
        [Decimal("9"), Decimal("11")],
        index=pd.to_datetime(["2025-01-01", "2026-01-01"]),
    )
    dataset = replace(
        usd_company_dataset,
        fast_info={
            key: value
            for key, value in usd_company_dataset.fast_info.items()
            if key != "shares"
        },
        shares=shares,
    )
    result = build_processor(dataset).value()

    assert result.intrinsic_value_per_share == result.equity_value / Decimal("11")
    assert any(
        diagnostic.metric == "shares outstanding"
        and diagnostic.source_attempted == "shares_history"
        for diagnostic in result.diagnostics
    )

    with pytest.raises(MetricUnavailableError):
        build_processor(replace(dataset, shares=None)).value()


def test_risk_free_rate_falls_back_to_us_treasury(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Unavailable primary government yield should use the US ten-year yield."""

    provider = FakeMacroRateProvider(fail_primary=True)
    result = build_processor(
        usd_company_dataset,
        macro_provider=provider,
    ).value()

    assert result.cost_of_equity.risk_free_rate == Decimal("0.045")
    assert provider.calls == [
        "primary:USD:United States",
        "fallback:US10Y",
    ]
    assert any(
        diagnostic.metric == "risk-free rate" and diagnostic.kind == "fallback"
        for diagnostic in result.diagnostics
    )


@pytest.mark.parametrize(
    "provider_kind",
    ["tax", "erp", "macro", "market_debt", "fx"],
)
def test_provider_failure_stops_without_leaking_secret(
    usd_company_dataset: CompanyDataset,
    currency_mismatch_dataset: CompanyDataset,
    provider_kind: str,
) -> None:
    """Unrecoverable provider failures should be typed and secret-safe."""

    secret = "provider-secret-value"

    if provider_kind == "tax":
        processor = build_processor(
            usd_company_dataset,
            tax_provider=FakeTaxRateProvider(fail=True),
        )
    elif provider_kind == "erp":
        processor = build_processor(
            usd_company_dataset,
            erp_provider=FakeEquityRiskPremiumProvider(fail=True),
        )
    elif provider_kind == "macro":
        processor = build_processor(
            usd_company_dataset,
            macro_provider=FakeMacroRateProvider(
                fail_primary=True,
                fail_fallback=True,
            ),
        )
    elif provider_kind == "market_debt":
        processor = build_processor(
            usd_company_dataset,
            debt_provider=FakeMarketDebtProvider(fail=True),
        )
    else:
        processor = build_processor(
            currency_mismatch_dataset,
            fx_provider=FakeFxRateProvider(
                provider_name="secret-safe FX",
                fail=True,
            ),
        )

    with pytest.raises(ProviderUnavailableError) as captured:
        try:
            processor.value()
        except ProviderUnavailableError as error:
            raise error from RuntimeError(secret)

    assert captured.value.provider_name
    assert captured.value.input_name
    assert secret not in str(captured.value)


def test_sovereign_selection_reports_rejections_and_ambiguity(
    eur_company_dataset: CompanyDataset,
) -> None:
    """Sovereign ranking should select deterministically or fail when tied."""

    selected = sovereign_candidate("DE10Y", "EUR", "Germany", "0.024")
    rejected = sovereign_candidate(
        "DE9Y",
        "EUR",
        "Germany",
        "0.023",
        maturity_years="9",
        confidence="0.99",
    )
    result = build_processor(
        eur_company_dataset,
        sovereign_provider=FakeSovereignYieldProvider((rejected, selected)),
    ).terminal_growth_rate()

    assert result.selected_instrument == "DE10Y"
    assert result.fallbacks_attempted == ("DE9Y",)
    assert "DE9Y" in result.diagnostics[-1].fallbacks_attempted

    tied = replace(selected, symbol="DE10Y-ALT")
    with pytest.raises(ProviderUnavailableError):
        build_processor(
            eur_company_dataset,
            sovereign_provider=FakeSovereignYieldProvider((selected, tied)),
        ).terminal_growth_rate()


def test_missing_sovereign_yield_is_actionable(
    usd_company_dataset: CompanyDataset,
) -> None:
    """No sovereign candidates should suggest a terminal-growth override."""

    with pytest.raises(ProviderUnavailableError) as captured:
        build_processor(
            usd_company_dataset,
            sovereign_provider=FakeSovereignYieldProvider(()),
        ).terminal_growth_rate()

    assert "terminal_growth_rate_override" in captured.value.suggested_override


def test_invalid_terminal_growth_includes_wacc_and_growth(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Terminal growth at or above WACC should retain validation context."""

    assumptions = ValuationAssumptions(
        valuation_date=VALUATION_DATE,
        terminal_growth_rate_override=Decimal("0.20"),
    )
    with pytest.raises(InvalidAssumptionsError) as captured:
        build_processor(
            usd_company_dataset,
            assumptions=assumptions,
        ).value()

    assert captured.value.wacc is not None
    assert captured.value.terminal_growth_rate == Decimal("0.20")
    assert "terminal_growth_rate_override" in captured.value.suggested_override


@pytest.mark.parametrize("ebit", [Decimal("0"), Decimal("-1")])
def test_invalid_nopat_stops_valuation(
    usd_company_dataset: CompanyDataset,
    ebit: Decimal,
) -> None:
    """Non-positive NOPAT should stop the integrated workflow."""

    income = usd_company_dataset.income_statement.astype(object)
    income.loc["EBIT", income.columns[-1]] = ebit

    with pytest.raises(InvalidAssumptionsError) as captured:
        build_processor(replace(usd_company_dataset, income_statement=income)).value()

    assert captured.value.metric == "nopat"
    assert captured.value.period == income.columns[-1]
    assert captured.value.metadata["source_rows"] == ("EBIT", "tax_rate")


@pytest.mark.parametrize("fcff_capex", [Decimal("127"), Decimal("128")])
def test_invalid_fcff_stops_valuation(
    usd_company_dataset: CompanyDataset,
    fcff_capex: Decimal,
) -> None:
    """Zero or negative FCFF should stop the integrated workflow."""

    cashflow = usd_company_dataset.cashflow.astype(object)
    cashflow.loc["Capital Expenditure", cashflow.columns[-1]] = fcff_capex

    with pytest.raises(InvalidAssumptionsError) as captured:
        build_processor(replace(usd_company_dataset, cashflow=cashflow)).value()

    assert captured.value.metric == "fcff"
    assert captured.value.period == cashflow.columns[-1]
    assert "Capital Expenditure" in captured.value.metadata["source_rows"]


@pytest.mark.parametrize("capital", [Decimal("0"), Decimal("-1")])
def test_invalid_invested_capital_stops_valuation(
    usd_company_dataset: CompanyDataset,
    capital: Decimal,
) -> None:
    """Non-positive prior invested capital should stop growth estimation."""

    balance = usd_company_dataset.balance_sheet.astype(object)
    balance.loc["Invested Capital", balance.columns[-2]] = capital

    with pytest.raises(InvalidAssumptionsError) as captured:
        build_processor(replace(usd_company_dataset, balance_sheet=balance)).value()

    assert captured.value.metric == "invested_capital"
    assert captured.value.period == balance.columns[-1]
    assert captured.value.metadata["source_rows"] == ("Invested Capital",)


def test_unsupported_currency_raises_typed_error(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Unsupported valuation currency should include override guidance."""

    info = dict(usd_company_dataset.info)
    info["financialCurrency"] = "NOT_A_CURRENCY"

    with pytest.raises(UnsupportedCurrencyError) as captured:
        build_processor(replace(usd_company_dataset, info=info)).value()

    assert "valuation_currency_override" in captured.value.suggested_override


def test_missing_statement_preserves_yfinance_context(
    usd_company_dataset: CompanyDataset,
) -> None:
    """Missing required statement should retain typed source context."""

    class MissingStatementClient(FakeYFinanceClient):
        """Raise a typed missing-statement error."""

        def get_income_statement(self, freq: str = "yearly") -> pd.DataFrame:
            """Raise the deterministic statement failure."""

            raise StatementUnavailableError(
                self.normalized_ticker(),
                "income statement",
                source_attempted="yfinance.Ticker.get_income_stmt",
                fallbacks_attempted=("quarterly income statement",),
            )

    stock = Stock(
        usd_company_dataset.ticker,
        yfinance_client=MissingStatementClient(usd_company_dataset),
        tax_rate=Decimal("0.20"),
    )

    with pytest.raises(StatementUnavailableError) as captured:
        stock.revenue_series()

    assert captured.value.ticker == usd_company_dataset.ticker
    assert captured.value.statement_name == "income statement"
    assert captured.value.source_attempted == "yfinance.Ticker.get_income_stmt"
    assert captured.value.fallbacks_attempted == ("quarterly income statement",)
