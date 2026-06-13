"""Valuation-ready stock model built on the Yahoo Finance client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from stock_valuation.contracts import FcffInputs
from stock_valuation.errors import MetricUnavailableError
from stock_valuation.mapping import (
    SourceMetadata,
    align_series,
    map_beta,
    map_capex_series,
    map_cash_series,
    map_company_name,
    map_current_price,
    map_debt_series,
    map_depreciation_series,
    map_ebit_series,
    map_headquarters_country,
    map_interest_expense_series,
    map_invested_capital_series,
    map_market_cap,
    map_minority_interest_series,
    map_non_operating_assets_series,
    map_revenue_series,
    map_shares_outstanding,
    map_trading_currency,
    map_valuation_currency,
    map_working_capital_change_series,
)
from stock_valuation.providers import TaxRateProvider
from stock_valuation.yfinance_client import YFinanceClient

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Stock:
    """Normalize yfinance company data into valuation-ready metrics.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol.
    yfinance_client:
        Optional ticker-bound client. When omitted, one is created from ``ticker``.
    tax_rate_provider:
        Optional provider for country-level corporate tax rates.
    tax_rate:
        Optional explicit tax-rate override.
    """

    ticker: str
    yfinance_client: YFinanceClient | None = None
    tax_rate_provider: TaxRateProvider | None = None
    tax_rate: Decimal | None = None
    diagnostics: list[str] = field(default_factory=list, init=False)
    mapping_metadata: list[SourceMetadata] = field(default_factory=list, init=False)
    _info: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _fast_info: Mapping[str, Any] | None = field(default=None, init=False, repr=False)
    _annual_income_statement: pd.DataFrame | None = field(
        default=None, init=False, repr=False
    )
    _annual_balance_sheet: pd.DataFrame | None = field(
        default=None, init=False, repr=False
    )
    _annual_cashflow: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create a ticker-bound client when one is not injected."""

        if self.yfinance_client is None:
            self.yfinance_client = YFinanceClient(self.ticker)

    def company_name(self) -> str:
        """Return the best available company display name.

        Returns
        -------
        str
            yfinance long name, short name, or normalized ticker fallback.
        """

        name = map_company_name(
            self.raw_info(), self.normalized_ticker(), metadata=self.mapping_metadata
        )
        if self.mapping_metadata[-1].selected_source == "ticker":
            self.diagnostics.append("Company name unavailable; using ticker symbol.")
        return name

    def headquarters_country(self) -> str:
        """Return the company headquarters country from yfinance metadata.

        Returns
        -------
        str
            Company headquarters country.

        Raises
        ------
        MetricUnavailableError
            If yfinance metadata does not expose a country.
        """

        return map_headquarters_country(
            self.raw_info(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def valuation_currency(self) -> str:
        """Return the currency used by the company's financial statements.

        Returns
        -------
        str
            Normalized valuation currency code.
        """

        return map_valuation_currency(
            self.raw_info(),
            self.raw_fast_info(),
            metadata=self.mapping_metadata,
        )

    def trading_currency(self) -> str:
        """Return the currency used for market prices.

        Returns
        -------
        str
            Normalized trading currency code.
        """

        trading_currency = map_trading_currency(
            self.raw_fast_info(),
            self.raw_info(),
            metadata=self.mapping_metadata,
        )
        valuation_currency = self.valuation_currency()
        if trading_currency != valuation_currency:
            self.diagnostics.append(
                f"Trading currency {trading_currency} differs from valuation currency {valuation_currency}."
            )
        return trading_currency

    def current_price(self) -> Decimal:
        """Return the latest stock price in the trading currency.

        Returns
        -------
        Decimal
            Latest available stock price.
        """

        try:
            return map_current_price(
                self.raw_fast_info(),
                self.raw_info(),
                None,
                self.normalized_ticker(),
                metadata=self.mapping_metadata,
            )
        except MetricUnavailableError:
            history = self.yfinance_client.get_history(period="5d", interval="1d")
            price = map_current_price(
                self.raw_fast_info(),
                self.raw_info(),
                history,
                self.normalized_ticker(),
                metadata=self.mapping_metadata,
            )
            self.diagnostics.append(
                "Current price resolved from latest historical close."
            )
            return price

    def market_cap(self) -> Decimal | None:
        """Return market capitalization when available.

        Returns
        -------
        Decimal | None
            Market capitalization, or ``None`` when unavailable.
        """

        market_cap = map_market_cap(
            self.raw_fast_info(),
            self.raw_info(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )
        if market_cap is None:
            self.diagnostics.append("Market capitalization unavailable.")
        return market_cap

    def beta(self) -> Decimal | None:
        """Return the company's equity beta when available.

        Returns
        -------
        Decimal | None
            Equity beta, or ``None`` when unavailable.
        """

        beta = map_beta(
            self.raw_info(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )
        if beta is None:
            self.diagnostics.append("Beta unavailable.")
        return beta

    def shares_outstanding(self) -> Decimal:
        """Return the latest shares outstanding count.

        Returns
        -------
        Decimal
            Latest shares outstanding count.
        """

        try:
            return map_shares_outstanding(
                self.raw_fast_info(),
                self.raw_info(),
                None,
                self.normalized_ticker(),
                metadata=self.mapping_metadata,
            )
        except MetricUnavailableError:
            return map_shares_outstanding(
                self.raw_fast_info(),
                self.raw_info(),
                self.yfinance_client.get_shares_full(),
                self.normalized_ticker(),
                metadata=self.mapping_metadata,
            )

    def annual_income_statement(self) -> "pd.DataFrame":
        """Return the cached annual income statement.

        Returns
        -------
        pandas.DataFrame
            Raw annual income statement from ``YFinanceClient``.
        """

        if self._annual_income_statement is None:
            self._annual_income_statement = self.yfinance_client.get_income_statement(
                freq="yearly"
            )
        return self._annual_income_statement

    def annual_balance_sheet(self) -> "pd.DataFrame":
        """Return the cached annual balance sheet.

        Returns
        -------
        pandas.DataFrame
            Raw annual balance sheet from ``YFinanceClient``.
        """

        if self._annual_balance_sheet is None:
            self._annual_balance_sheet = self.yfinance_client.get_balance_sheet(
                freq="yearly"
            )
        return self._annual_balance_sheet

    def annual_cashflow(self) -> "pd.DataFrame":
        """Return the cached annual cash flow statement.

        Returns
        -------
        pandas.DataFrame
            Raw annual cash flow statement from ``YFinanceClient``.
        """

        if self._annual_cashflow is None:
            self._annual_cashflow = self.yfinance_client.get_cashflow(freq="yearly")
        return self._annual_cashflow

    def revenue_series(self) -> "pd.Series":
        """Return annual revenue as a chronological series.

        Returns
        -------
        pandas.Series
            Annual revenue values ordered from oldest to newest.
        """

        return map_revenue_series(
            self.annual_income_statement(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def ebit_series(self) -> "pd.Series":
        """Return annual EBIT as a chronological series.

        Returns
        -------
        pandas.Series
            Annual EBIT values ordered from oldest to newest.

        Raises
        ------
        MetricUnavailableError
            If no EBIT row or derivation proxy exists.
        """

        return map_ebit_series(
            self.annual_income_statement(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def interest_expense_series(self) -> "pd.Series":
        """Return annual interest expense as positive values.

        Returns
        -------
        pandas.Series
            Annual interest expense values ordered from oldest to newest.
        """

        return map_interest_expense_series(
            self.annual_income_statement(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def depreciation_series(self) -> "pd.Series":
        """Return annual depreciation and amortization as positive values.

        Returns
        -------
        pandas.Series
            Annual depreciation and amortization ordered from oldest to newest.
        """

        return map_depreciation_series(
            self.annual_cashflow(),
            self.normalized_ticker(),
            income_statement=self.annual_income_statement(),
            metadata=self.mapping_metadata,
        )

    def capex_series(self) -> "pd.Series":
        """Return annual capital expenditures as positive outflows.

        Returns
        -------
        pandas.Series
            Annual capital expenditures ordered from oldest to newest.
        """

        return map_capex_series(
            self.annual_cashflow(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def change_in_non_cash_working_capital_series(self) -> "pd.Series":
        """Return annual non-cash working-capital increases as positive outflows.

        Returns
        -------
        pandas.Series
            Annual changes in non-cash working capital ordered from oldest to newest.
        """

        return map_working_capital_change_series(
            self.annual_cashflow(),
            self.annual_balance_sheet(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def debt_series(self) -> "pd.Series":
        """Return annual total debt as positive values.

        Returns
        -------
        pandas.Series
            Annual total debt ordered from oldest to newest.
        """

        return map_debt_series(
            self.annual_balance_sheet(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def cash_series(self) -> "pd.Series":
        """Return annual cash and equivalents as positive values.

        Returns
        -------
        pandas.Series
            Annual cash and equivalent balances ordered from oldest to newest.
        """

        return map_cash_series(
            self.annual_balance_sheet(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def non_operating_assets_series(self) -> "pd.Series | None":
        """Return identifiable annual non-operating financial assets.

        Returns
        -------
        pandas.Series | None
            Positive annual non-operating assets, or ``None`` when unavailable.
        """

        return map_non_operating_assets_series(
            self.annual_balance_sheet(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def minority_interest_series(self) -> "pd.Series | None":
        """Return annual minority interest as a positive claim.

        Returns
        -------
        pandas.Series | None
            Positive annual minority interest, or ``None`` when unavailable.
        """

        return map_minority_interest_series(
            self.annual_balance_sheet(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def latest_fcff_inputs(self) -> FcffInputs:
        """Return the latest common-period FCFF input bundle.

        Returns
        -------
        FcffInputs
            Latest aligned normalized inputs needed by FCFF calculation.

        Raises
        ------
        MetricUnavailableError
            If annual input series do not share a common period.
        """

        ebit, depreciation = align_series(
            self.ebit_series(),
            self.depreciation_series(),
            self.normalized_ticker(),
            "FCFF inputs",
        )
        capex, working_capital = align_series(
            self.capex_series(),
            self.change_in_non_cash_working_capital_series(),
            self.normalized_ticker(),
            "FCFF inputs",
        )
        common_periods = (
            ebit.index.intersection(depreciation.index)
            .intersection(capex.index)
            .intersection(working_capital.index)
        )
        if common_periods.empty:
            raise self._metric_error("FCFF inputs", "common annual statement periods")
        period = common_periods[-1]
        self._record_diagnostic_once(
            f"FCFF inputs resolved from annual statement period {period}."
        )
        return FcffInputs(
            period=period,
            ebit=ebit.loc[period],
            tax_rate=self._tax_rate_for_valuation(),
            depreciation_amortization=depreciation.loc[period],
            capex=capex.loc[period],
            change_in_non_cash_working_capital=working_capital.loc[period],
        )

    def invested_capital_series(self) -> "pd.Series":
        """Return annual invested capital.

        Returns
        -------
        pandas.Series
            Annual invested capital ordered from oldest to newest.
        """

        return map_invested_capital_series(
            self.annual_balance_sheet(),
            self.normalized_ticker(),
            metadata=self.mapping_metadata,
        )

    def return_on_capital_series(self) -> "pd.Series":
        """Return annual return on capital using prior-period invested capital.

        Returns
        -------
        pandas.Series
            Annual return on capital ordered from oldest to newest.
        """

        nopat = self.ebit_series() * (Decimal("1") - self._tax_rate_for_valuation())
        invested_capital = self.invested_capital_series().shift(1)
        nopat, invested_capital = align_series(
            nopat,
            invested_capital.dropna(),
            self.normalized_ticker(),
            "return on capital",
        )
        valid = invested_capital[invested_capital != Decimal("0")]
        nopat = nopat.loc[valid.index]
        if valid.empty:
            raise self._metric_error(
                "return on capital", "prior-period invested capital"
            )
        return nopat / valid

    def reinvestment_rate_series(self) -> "pd.Series":
        """Return annual reinvestment rate.

        Returns
        -------
        pandas.Series
            Annual reinvestment rate ordered from oldest to newest.
        """

        capex, depreciation = align_series(
            self.capex_series(),
            self.depreciation_series(),
            self.normalized_ticker(),
            "reinvestment rate",
        )
        capex, working_capital = align_series(
            capex,
            self.change_in_non_cash_working_capital_series(),
            self.normalized_ticker(),
            "reinvestment rate",
        )
        nopat = self.ebit_series() * (Decimal("1") - self._tax_rate_for_valuation())
        capex, nopat = align_series(
            capex, nopat, self.normalized_ticker(), "reinvestment rate"
        )
        reinvestment = (
            capex - depreciation.loc[capex.index] + working_capital.loc[capex.index]
        )
        valid = nopat[nopat != Decimal("0")]
        reinvestment = reinvestment.loc[valid.index]
        if valid.empty:
            raise self._metric_error("reinvestment rate", "NOPAT")
        return reinvestment / valid

    def raw_info(self) -> dict[str, Any]:
        """Return cached yfinance info metadata.

        Returns
        -------
        dict[str, Any]
            Raw yfinance info metadata.
        """

        if self._info is None:
            self._info = self.yfinance_client.get_info()
        return self._info

    def raw_fast_info(self) -> Mapping[str, Any]:
        """Return cached yfinance fast-info metadata.

        Returns
        -------
        Mapping[str, Any]
            Raw yfinance fast-info metadata.
        """

        if self._fast_info is None:
            self._fast_info = self.yfinance_client.get_fast_info()
        return self._fast_info

    def normalized_ticker(self) -> str:
        """Return the normalized ticker symbol.

        Returns
        -------
        str
            Normalized ticker symbol from the client.
        """

        return self.yfinance_client.normalized_ticker()

    def _tax_rate_for_valuation(self) -> Decimal:
        if self.tax_rate is not None:
            self._record_diagnostic_once("Tax rate resolved from explicit override.")
            return self.tax_rate
        if self.tax_rate_provider is None:
            self._record_diagnostic_once(
                "Tax rate provider unavailable; using 0 for stock-level normalized rates."
            )
            return Decimal("0")
        valuation_date = date.today()
        country = self.headquarters_country()
        self._record_diagnostic_once(
            f"Tax rate resolved from provider for {country} on {valuation_date}."
        )
        return self.tax_rate_provider.get_corporate_tax_rate(country, valuation_date)

    def _metric_error(
        self, metric_name: str, source_attempted: str
    ) -> MetricUnavailableError:
        return MetricUnavailableError(
            self.normalized_ticker(),
            metric_name,
            source_attempted=source_attempted,
        )

    def _record_diagnostic_once(self, message: str) -> None:
        if message not in self.diagnostics:
            self.diagnostics.append(message)
