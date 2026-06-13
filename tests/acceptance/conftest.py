"""Deterministic fixtures for valuation acceptance scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from stock_valuation.contracts import Diagnostic, ValuationAssumptions
from stock_valuation.errors import ProviderUnavailableError
from stock_valuation.processor import DamodaranValuationProcessor
from stock_valuation.providers import (
    SovereignYieldCandidate,
    SovereignYieldResult,
)
from stock_valuation.stock import Stock

VALUATION_DATE = date(2026, 6, 13)


@dataclass(frozen=True)
class CompanyDataset:
    """Raw company data used by the deterministic yfinance fake.

    Attributes
    ----------
    ticker:
        Yahoo Finance ticker symbol.
    info:
        Raw ticker metadata.
    fast_info:
        Raw fast market metadata.
    income_statement:
        Annual income statement.
    balance_sheet:
        Annual balance sheet.
    cashflow:
        Annual cash-flow statement.
    shares:
        Optional shares history.
    history:
        Historical price data.
    """

    ticker: str
    info: dict[str, Any]
    fast_info: dict[str, Any]
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cashflow: pd.DataFrame
    shares: pd.Series | None = None
    history: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame({"Close": [Decimal("80")]})
    )


class FakeYFinanceClient:
    """Expose deterministic yfinance client surfaces without network calls."""

    def __init__(self, dataset: CompanyDataset) -> None:
        """Initialize the client.

        Parameters
        ----------
        dataset:
            Raw deterministic company data.
        """

        self.dataset = dataset

    def normalized_ticker(self) -> str:
        """Return the normalized ticker.

        Returns
        -------
        str
            Uppercase ticker symbol.
        """

        return self.dataset.ticker.strip().upper()

    def get_info(self) -> dict[str, Any]:
        """Return raw ticker metadata."""

        return self.dataset.info

    def get_fast_info(self) -> dict[str, Any]:
        """Return raw fast market metadata."""

        return self.dataset.fast_info

    def get_income_statement(self, freq: str = "yearly") -> pd.DataFrame:
        """Return the annual income statement."""

        assert freq == "yearly"
        return self.dataset.income_statement

    def get_balance_sheet(self, freq: str = "yearly") -> pd.DataFrame:
        """Return the annual balance sheet."""

        assert freq == "yearly"
        return self.dataset.balance_sheet

    def get_cashflow(self, freq: str = "yearly") -> pd.DataFrame:
        """Return the annual cash-flow statement."""

        assert freq == "yearly"
        return self.dataset.cashflow

    def get_shares_full(self) -> pd.Series | None:
        """Return optional shares history."""

        return self.dataset.shares

    def get_history(
        self,
        period: str = "5d",
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Return deterministic price history."""

        return self.dataset.history


@dataclass
class FakeTaxRateProvider:
    """Return a deterministic corporate tax rate."""

    rate: Decimal = Decimal("0.20")
    fail: bool = False

    def get_corporate_tax_rate(
        self,
        country: str,
        valuation_date: date,
    ) -> Decimal:
        """Return the configured corporate tax rate."""

        if self.fail:
            raise ProviderUnavailableError("fake tax", "corporate tax rate")
        return self.rate


@dataclass
class FakeEquityRiskPremiumProvider:
    """Return a deterministic equity risk premium."""

    premium: Decimal = Decimal("0.05")
    fail: bool = False

    def get_equity_risk_premium(
        self,
        country: str,
        valuation_date: date,
    ) -> Decimal:
        """Return the configured equity risk premium."""

        if self.fail:
            raise ProviderUnavailableError("fake ERP", "equity risk premium")
        return self.premium


@dataclass
class FakeMacroRateProvider:
    """Return deterministic primary and fallback government yields."""

    primary_rate: Decimal = Decimal("0.04")
    fallback_rate: Decimal = Decimal("0.045")
    fail_primary: bool = False
    fail_fallback: bool = False
    calls: list[str] = field(default_factory=list)

    def get_long_term_government_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> Decimal:
        """Return the currency-matched long-term yield."""

        self.calls.append(f"primary:{currency}:{country}")
        if self.fail_primary:
            raise ProviderUnavailableError(
                "fake macro",
                "currency-matched government yield",
            )
        return self.primary_rate

    def get_us_10y_treasury_yield(self, valuation_date: date) -> Decimal:
        """Return the US 10-year Treasury fallback."""

        self.calls.append("fallback:US10Y")
        if self.fail_fallback:
            raise ProviderUnavailableError("fake macro", "US 10-year Treasury")
        return self.fallback_rate


@dataclass
class FakeMarketDebtProvider:
    """Return deterministic market debt or no value."""

    debt: Decimal | None = Decimal("120")
    fail: bool = False

    def get_market_value_of_debt(
        self,
        ticker: str,
        valuation_date: date,
    ) -> Decimal | None:
        """Return configured market debt."""

        if self.fail:
            raise ProviderUnavailableError("fake debt", "market debt")
        return self.debt


@dataclass
class FakeFxRateProvider:
    """Convert market values with an exact deterministic FX rate."""

    rate: Decimal = Decimal("1.25")
    provider_name: str = "fake FX"
    fail: bool = False
    calls: list[tuple[str, str, date]] = field(default_factory=list)

    def convert(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        valuation_date: date,
    ) -> Decimal:
        """Convert an amount using the configured rate."""

        self.calls.append((from_currency, to_currency, valuation_date))
        if self.fail:
            raise ProviderUnavailableError(
                self.provider_name,
                f"{from_currency}/{to_currency} conversion",
            )
        return amount * self.rate

    def get_rate(
        self,
        from_currency: str,
        to_currency: str,
        valuation_date: date,
    ) -> Decimal:
        """Return the configured FX rate."""

        if self.fail:
            raise ProviderUnavailableError(
                self.provider_name,
                f"{from_currency}/{to_currency} rate",
            )
        return self.rate


@dataclass
class FakeSovereignYieldProvider:
    """Select a sovereign-yield candidate deterministically."""

    candidates: tuple[SovereignYieldCandidate, ...]
    provider_name: str = "fake sovereign"

    def find_10y_sovereign_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> SovereignYieldResult:
        """Select by currency, country, maturity, type, and confidence."""

        ranked = sorted(
            self.candidates,
            key=lambda candidate: (
                candidate.currency == currency,
                candidate.country == country,
                candidate.maturity_years == Decimal("10"),
                candidate.instrument_type == "government-bond",
                candidate.confidence,
            ),
            reverse=True,
        )
        if not ranked:
            raise ProviderUnavailableError(
                self.provider_name,
                "10-year sovereign yield",
                suggested_override="Provide terminal_growth_rate_override.",
            )
        selected = ranked[0]
        selected_rank = (
            selected.currency == currency,
            selected.country == country,
            selected.maturity_years == Decimal("10"),
            selected.instrument_type == "government-bond",
            selected.confidence,
        )
        if len(ranked) > 1:
            runner_up = ranked[1]
            runner_up_rank = (
                runner_up.currency == currency,
                runner_up.country == country,
                runner_up.maturity_years == Decimal("10"),
                runner_up.instrument_type == "government-bond",
                runner_up.confidence,
            )
            if runner_up_rank == selected_rank:
                raise ProviderUnavailableError(
                    self.provider_name,
                    "deterministic sovereign-yield selection",
                    suggested_override="Provide terminal_growth_rate_override.",
                )
        rejected = tuple(
            replace(candidate, rejection_reason="Lower deterministic rank.")
            for candidate in ranked[1:]
        )
        diagnostic = Diagnostic(
            kind="provider",
            message=(
                f"Selected {selected.symbol} for {currency}/{country} on "
                f"{valuation_date}."
            ),
            provider=self.provider_name,
            source_attempted=selected.symbol,
            fallbacks_attempted=tuple(item.symbol for item in rejected),
        )
        return SovereignYieldResult(
            selected=selected,
            provider=self.provider_name,
            valuation_date=valuation_date,
            rejected=rejected,
            diagnostics=[diagnostic],
        )


def sovereign_candidate(
    symbol: str,
    currency: str,
    country: str,
    yield_value: str,
    *,
    confidence: str = "0.95",
    maturity_years: str = "10",
    instrument_type: str = "government-bond",
) -> SovereignYieldCandidate:
    """Build a deterministic sovereign-yield candidate.

    Returns
    -------
    SovereignYieldCandidate
        Candidate with exact Decimal ranking fields.
    """

    return SovereignYieldCandidate(
        symbol=symbol,
        currency=currency,
        country=country,
        maturity_years=Decimal(maturity_years),
        instrument_type=instrument_type,
        confidence=Decimal(confidence),
        yield_value=Decimal(yield_value),
    )


def make_company_dataset(
    *,
    ticker: str = "TEST",
    country: str = "United States",
    valuation_currency: str = "USD",
    trading_currency: str | None = None,
    market_cap: Decimal = Decimal("1000"),
    current_price: Decimal = Decimal("80"),
    shares_in_metadata: Decimal | None = Decimal("10"),
) -> CompanyDataset:
    """Build a complete deterministic company dataset.

    Returns
    -------
    CompanyDataset
        Dataset with three aligned annual statement periods.
    """

    periods = pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])
    trading_currency = trading_currency or valuation_currency
    info: dict[str, Any] = {
        "longName": f"{ticker} Corporation",
        "country": country,
        "financialCurrency": valuation_currency,
        "currency": trading_currency,
        "beta": Decimal("1.1"),
    }
    fast_info: dict[str, Any] = {
        "currency": trading_currency,
        "market_cap": market_cap,
        "last_price": current_price,
    }
    if shares_in_metadata is not None:
        fast_info["shares"] = shares_in_metadata
    income_statement = pd.DataFrame(
        {
            periods[0]: [400, 120, -4],
            periods[1]: [450, 140, -5],
            periods[2]: [500, 150, -6],
        },
        index=["Total Revenue", "EBIT", "Interest Expense"],
    )
    balance_sheet = pd.DataFrame(
        {
            periods[0]: [100, 30, 500, 570, 180, 90, 10],
            periods[1]: [110, 35, 550, 625, 200, 100, 10],
            periods[2]: [120, 40, 600, 680, 220, 110, 12],
        },
        index=[
            "Total Debt",
            "Cash And Cash Equivalents",
            "Stockholders Equity",
            "Invested Capital",
            "Current Assets",
            "Current Liabilities",
            "Current Debt",
        ],
    )
    cashflow = pd.DataFrame(
        {
            periods[0]: [10, -20, -3],
            periods[1]: [11, -22, -4],
            periods[2]: [12, -24, -5],
        },
        index=[
            "Depreciation And Amortization",
            "Capital Expenditure",
            "Change In Working Capital",
        ],
    )
    return CompanyDataset(
        ticker=ticker,
        info=info,
        fast_info=fast_info,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cashflow=cashflow,
    )


def build_processor(
    dataset: CompanyDataset,
    *,
    assumptions: ValuationAssumptions | None = None,
    macro_provider: FakeMacroRateProvider | None = None,
    erp_provider: FakeEquityRiskPremiumProvider | None = None,
    tax_provider: FakeTaxRateProvider | None = None,
    debt_provider: FakeMarketDebtProvider | None = None,
    sovereign_provider: FakeSovereignYieldProvider | None = None,
    fx_provider: FakeFxRateProvider | None = None,
) -> DamodaranValuationProcessor:
    """Build the integrated deterministic valuation workflow.

    Returns
    -------
    DamodaranValuationProcessor
        Processor wired to real stock normalization and fake providers.
    """

    tax_provider = tax_provider or FakeTaxRateProvider()
    stock = Stock(
        dataset.ticker,
        yfinance_client=FakeYFinanceClient(dataset),
        tax_rate_provider=tax_provider,
        tax_rate=tax_provider.rate,
    )
    return DamodaranValuationProcessor(
        stock=stock,
        assumptions=assumptions or ValuationAssumptions(valuation_date=VALUATION_DATE),
        macro_provider=macro_provider or FakeMacroRateProvider(),
        erp_provider=erp_provider or FakeEquityRiskPremiumProvider(),
        tax_rate_provider=tax_provider,
        market_debt_provider=debt_provider or FakeMarketDebtProvider(),
        sovereign_yield_provider=sovereign_provider
        or FakeSovereignYieldProvider(
            (
                sovereign_candidate(
                    "US10Y",
                    "USD",
                    "United States",
                    "0.025",
                ),
            )
        ),
        fx_rate_provider=fx_provider,
    )


@pytest.fixture
def usd_company_dataset() -> CompanyDataset:
    """Return the default USD company dataset.

    Returns
    -------
    CompanyDataset
        Complete USD fixture.
    """

    return make_company_dataset()


@pytest.fixture
def eur_company_dataset() -> CompanyDataset:
    """Return a EUR company dataset.

    Returns
    -------
    CompanyDataset
        Complete German EUR fixture.
    """

    return make_company_dataset(
        ticker="SAP",
        country="Germany",
        valuation_currency="EUR",
    )


@pytest.fixture
def currency_mismatch_dataset() -> CompanyDataset:
    """Return USD statements with GBP trading data.

    Returns
    -------
    CompanyDataset
        Complete currency-mismatch fixture.
    """

    return make_company_dataset(
        ticker="MISMATCH",
        valuation_currency="USD",
        trading_currency="GBP",
        market_cap=Decimal("800"),
        current_price=Decimal("64"),
    )
