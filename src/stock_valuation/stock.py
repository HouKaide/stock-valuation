"""Valuation-ready stock model built on the Yahoo Finance client."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Protocol

from stock_valuation.contracts import FcffInputs
from stock_valuation.errors import MetricUnavailableError
from stock_valuation.yfinance_client import YFinanceClient

if TYPE_CHECKING:
    import pandas as pd


class TaxRateProvider(Protocol):
    """Provider for country-level corporate tax rates.

    Methods
    -------
    get_corporate_tax_rate(country, valuation_date)
        Return the marginal corporate tax rate for a country and date.
    """

    def get_corporate_tax_rate(self, country: str, valuation_date: date) -> Decimal:
        """Return the marginal corporate tax rate for a country and date.

        Parameters
        ----------
        country:
            Company headquarters country.
        valuation_date:
            Date used to resolve the applicable tax rate.

        Returns
        -------
        Decimal
            Corporate tax rate as a decimal fraction.
        """


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
    _info: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _fast_info: Mapping[str, Any] | None = field(default=None, init=False, repr=False)
    _annual_income_statement: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _annual_balance_sheet: pd.DataFrame | None = field(default=None, init=False, repr=False)
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

        info = self.raw_info()
        name = _first_present(info, ("longName", "shortName"))
        if name is None:
            self.diagnostics.append("Company name unavailable; using ticker symbol.")
            return self.normalized_ticker()
        return str(name)

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

        country = _first_present(self.raw_info(), ("country",))
        if country is None or not str(country).strip():
            raise self._metric_error("headquarters country", "yfinance.Ticker.get_info.country")
        return str(country).strip()

    def valuation_currency(self) -> str:
        """Return the currency used by the company's financial statements.

        Returns
        -------
        str
            Normalized valuation currency code.
        """

        currency = _first_present(self.raw_info(), ("financialCurrency",))
        if currency is None:
            currency = _first_present(self.raw_fast_info(), ("currency",))
        return self._normalize_currency(currency, "valuation currency")

    def trading_currency(self) -> str:
        """Return the currency used for market prices.

        Returns
        -------
        str
            Normalized trading currency code.
        """

        currency = _first_present(self.raw_fast_info(), ("currency",))
        if currency is None:
            currency = _first_present(self.raw_info(), ("currency",))

        trading_currency = self._normalize_currency(currency, "trading currency")
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

        price = self._first_decimal(
            self.raw_fast_info(),
            ("last_price", "lastPrice", "lastPriceRaw", "regularMarketPrice"),
        )
        if price is not None:
            return price

        price = self._first_decimal(self.raw_info(), ("currentPrice", "regularMarketPrice"))
        if price is not None:
            return price

        history = self.yfinance_client.get_history(period="5d", interval="1d")
        close = self._latest_column_value(history, "Close")
        if close is not None:
            self.diagnostics.append("Current price resolved from latest historical close.")
            return close
        raise self._metric_error("current price", "fast_info.last_price")

    def market_cap(self) -> Decimal | None:
        """Return market capitalization when available.

        Returns
        -------
        Decimal | None
            Market capitalization, or ``None`` when unavailable.
        """

        market_cap = self._first_decimal(self.raw_fast_info(), ("market_cap", "marketCap"))
        if market_cap is not None:
            return market_cap
        market_cap = self._first_decimal(self.raw_info(), ("marketCap",))
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

        beta = self._first_decimal(self.raw_info(), ("beta",))
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

        shares = self._first_decimal(self.raw_fast_info(), ("shares", "sharesOutstanding"))
        if shares is not None:
            return shares

        shares = self._first_decimal(self.raw_info(), ("sharesOutstanding", "impliedSharesOutstanding"))
        if shares is not None:
            return shares

        shares_history = self.yfinance_client.get_shares_full()
        if shares_history is not None:
            shares_history = shares_history.dropna()
            if not shares_history.empty:
                return _to_decimal(shares_history.iloc[-1])
        raise self._metric_error("shares outstanding", "yfinance.Ticker.get_shares_full")

    def annual_income_statement(self) -> "pd.DataFrame":
        """Return the cached annual income statement.

        Returns
        -------
        pandas.DataFrame
            Raw annual income statement from ``YFinanceClient``.
        """

        if self._annual_income_statement is None:
            self._annual_income_statement = self.yfinance_client.get_income_statement(freq="yearly")
        return self._annual_income_statement

    def annual_balance_sheet(self) -> "pd.DataFrame":
        """Return the cached annual balance sheet.

        Returns
        -------
        pandas.DataFrame
            Raw annual balance sheet from ``YFinanceClient``.
        """

        if self._annual_balance_sheet is None:
            self._annual_balance_sheet = self.yfinance_client.get_balance_sheet(freq="yearly")
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

        return self._statement_series(self.annual_income_statement(), ("Total Revenue", "Operating Revenue"), "revenue")

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

        try:
            return self._statement_series(self.annual_income_statement(), ("EBIT", "Operating Income"), "EBIT")
        except MetricUnavailableError:
            revenue = self.revenue_series()
            operating_expenses = self._statement_series(
                self.annual_income_statement(),
                ("Operating Expense", "Total Operating Expenses"),
                "operating expenses",
            )
            revenue, operating_expenses = _align_pair(
                revenue, operating_expenses, "EBIT derivation", self.normalized_ticker()
            )
            return revenue - operating_expenses

    def interest_expense_series(self) -> "pd.Series":
        """Return annual interest expense as positive values.

        Returns
        -------
        pandas.Series
            Annual interest expense values ordered from oldest to newest.
        """

        return self._statement_series(
            self.annual_income_statement(),
            ("Interest Expense", "Interest Expense Non Operating"),
            "interest expense",
            absolute=True,
        )

    def depreciation_series(self) -> "pd.Series":
        """Return annual depreciation and amortization as positive values.

        Returns
        -------
        pandas.Series
            Annual depreciation and amortization ordered from oldest to newest.
        """

        return self._statement_series(
            self.annual_cashflow(),
            (
                "Depreciation And Amortization",
                "Depreciation Amortization Depletion",
                "Depreciation",
                "Reconciled Depreciation",
            ),
            "depreciation and amortization",
            absolute=True,
        )

    def capex_series(self) -> "pd.Series":
        """Return annual capital expenditures as positive outflows.

        Returns
        -------
        pandas.Series
            Annual capital expenditures ordered from oldest to newest.
        """

        return self._statement_series(
            self.annual_cashflow(),
            ("Capital Expenditure", "Capital Expenditure Reported", "Purchase Of PPE", "Net PPE Purchase And Sale"),
            "capital expenditure",
            absolute=True,
        )

    def change_in_non_cash_working_capital_series(self) -> "pd.Series":
        """Return annual non-cash working-capital increases as positive outflows.

        Returns
        -------
        pandas.Series
            Annual changes in non-cash working capital ordered from oldest to newest.
        """

        try:
            working_capital = self._statement_series(
                self.annual_cashflow(),
                ("Change In Working Capital",),
                "change in non-cash working capital",
            )
            return -working_capital
        except MetricUnavailableError:
            current_assets = self._statement_series(
                self.annual_balance_sheet(),
                ("Current Assets", "Total Current Assets"),
                "current assets",
            )
            cash = self.cash_series()
            current_liabilities = self._statement_series(
                self.annual_balance_sheet(),
                ("Current Liabilities", "Total Current Liabilities"),
                "current liabilities",
            )
            current_debt = self._optional_statement_series(
                self.annual_balance_sheet(),
                ("Current Debt", "Current Debt And Capital Lease Obligation"),
                current_assets.index,
            )
            assets, liabilities = _align_pair(
                current_assets - cash, current_liabilities - current_debt, "working capital", self.normalized_ticker()
            )
            non_cash_working_capital = assets - liabilities
            return non_cash_working_capital.diff().dropna()

    def debt_series(self) -> "pd.Series":
        """Return annual total debt as positive values.

        Returns
        -------
        pandas.Series
            Annual total debt ordered from oldest to newest.
        """

        try:
            return self._statement_series(self.annual_balance_sheet(), ("Total Debt",), "total debt", absolute=True)
        except MetricUnavailableError:
            long_term_debt = self._optional_statement_series(
                self.annual_balance_sheet(),
                ("Long Term Debt", "Long Term Debt And Capital Lease Obligation"),
                self.annual_balance_sheet().columns,
            )
            current_debt = self._optional_statement_series(
                self.annual_balance_sheet(),
                ("Current Debt", "Current Debt And Capital Lease Obligation"),
                self.annual_balance_sheet().columns,
            )
            debt = _chronological_series(long_term_debt + current_debt)
            if debt.empty:
                raise self._metric_error("total debt", "yfinance balance sheet debt rows")
            return debt.abs()

    def cash_series(self) -> "pd.Series":
        """Return annual cash and equivalents as positive values.

        Returns
        -------
        pandas.Series
            Annual cash and equivalent balances ordered from oldest to newest.
        """

        return self._statement_series(
            self.annual_balance_sheet(),
            ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
            "cash and cash equivalents",
            absolute=True,
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

        ebit, depreciation = _align_pair(
            self.ebit_series(), self.depreciation_series(), "FCFF inputs", self.normalized_ticker()
        )
        capex, working_capital = _align_pair(
            self.capex_series(),
            self.change_in_non_cash_working_capital_series(),
            "FCFF inputs",
            self.normalized_ticker(),
        )
        common_periods = (
            ebit.index.intersection(depreciation.index).intersection(capex.index).intersection(working_capital.index)
        )
        if common_periods.empty:
            raise self._metric_error("FCFF inputs", "common annual statement periods")
        period = common_periods[-1]
        self._record_diagnostic_once(f"FCFF inputs resolved from annual statement period {period}.")
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

        try:
            return self._statement_series(
                self.annual_balance_sheet(),
                ("Invested Capital",),
                "invested capital",
                absolute=True,
            )
        except MetricUnavailableError:
            debt = self.debt_series()
            equity = self._statement_series(
                self.annual_balance_sheet(),
                ("Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"),
                "book equity",
            )
            cash = self.cash_series()
            debt, equity = _align_pair(debt, equity, "invested capital", self.normalized_ticker())
            debt, cash = _align_pair(debt, cash, "invested capital", self.normalized_ticker())
            return debt + equity - cash

    def return_on_capital_series(self) -> "pd.Series":
        """Return annual return on capital using prior-period invested capital.

        Returns
        -------
        pandas.Series
            Annual return on capital ordered from oldest to newest.
        """

        nopat = self.ebit_series() * (Decimal("1") - self._tax_rate_for_valuation())
        invested_capital = self.invested_capital_series().shift(1)
        nopat, invested_capital = _align_pair(
            nopat, invested_capital.dropna(), "return on capital", self.normalized_ticker()
        )
        valid = invested_capital[invested_capital != Decimal("0")]
        nopat = nopat.loc[valid.index]
        if valid.empty:
            raise self._metric_error("return on capital", "prior-period invested capital")
        return nopat / valid

    def reinvestment_rate_series(self) -> "pd.Series":
        """Return annual reinvestment rate.

        Returns
        -------
        pandas.Series
            Annual reinvestment rate ordered from oldest to newest.
        """

        capex, depreciation = _align_pair(
            self.capex_series(), self.depreciation_series(), "reinvestment rate", self.normalized_ticker()
        )
        capex, working_capital = _align_pair(
            capex, self.change_in_non_cash_working_capital_series(), "reinvestment rate", self.normalized_ticker()
        )
        nopat = self.ebit_series() * (Decimal("1") - self._tax_rate_for_valuation())
        capex, nopat = _align_pair(capex, nopat, "reinvestment rate", self.normalized_ticker())
        reinvestment = capex - depreciation.loc[capex.index] + working_capital.loc[capex.index]
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

    def _statement_series(
        self,
        statement: "pd.DataFrame",
        row_names: Sequence[str],
        metric_name: str,
        *,
        absolute: bool = False,
    ) -> "pd.Series":
        row = _find_statement_row(statement, row_names)
        if row is None:
            raise self._metric_error(metric_name, f"yfinance statement rows {tuple(row_names)}")
        series = _chronological_series(row)
        if series.empty:
            raise self._metric_error(metric_name, f"yfinance statement rows {tuple(row_names)}")
        return series.abs() if absolute else series

    def _optional_statement_series(
        self,
        statement: "pd.DataFrame",
        row_names: Sequence[str],
        fallback_index: Iterable[Any],
    ) -> "pd.Series":
        import pandas as pd

        row = _find_statement_row(statement, row_names)
        if row is None:
            fallback_index = list(fallback_index)
            return pd.Series([Decimal("0")] * len(fallback_index), index=fallback_index, dtype="object")
        return _chronological_series(row).abs()

    def _tax_rate_for_valuation(self) -> Decimal:
        if self.tax_rate is not None:
            self._record_diagnostic_once("Tax rate resolved from explicit override.")
            return self.tax_rate
        if self.tax_rate_provider is None:
            self._record_diagnostic_once("Tax rate provider unavailable; using 0 for stock-level normalized rates.")
            return Decimal("0")
        valuation_date = date.today()
        country = self.headquarters_country()
        self._record_diagnostic_once(f"Tax rate resolved from provider for {country} on {valuation_date}.")
        return self.tax_rate_provider.get_corporate_tax_rate(country, valuation_date)

    def _first_decimal(self, source: Mapping[str, Any], keys: Sequence[str]) -> Decimal | None:
        value = _first_present(source, keys)
        if value is None:
            return None
        return _to_decimal(value)

    def _latest_column_value(self, frame: "pd.DataFrame", column: str) -> Decimal | None:
        if column not in frame:
            return None
        values = frame[column].dropna()
        if values.empty:
            return None
        return _to_decimal(values.iloc[-1])

    def _normalize_currency(self, value: Any, metric_name: str) -> str:
        if value is None or not str(value).strip():
            raise self._metric_error(metric_name, "yfinance currency metadata")
        currency = str(value).strip().upper()
        return {
            "GBP": "GBP",
            "GBX": "GBP",
            "GBPENCE": "GBP",
            "GBPENNY": "GBP",
            "GBPEN": "GBP",
        }.get(currency, currency)

    def _metric_error(self, metric_name: str, source_attempted: str) -> MetricUnavailableError:
        return MetricUnavailableError(
            self.normalized_ticker(),
            metric_name,
            source_attempted=source_attempted,
        )

    def _record_diagnostic_once(self, message: str) -> None:
        if message not in self.diagnostics:
            self.diagnostics.append(message)


def _first_present(source: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        try:
            value = source[key]
        except (KeyError, TypeError):
            value = getattr(source, key, None)
        if value is not None:
            return value
    return None


def _to_decimal(value: Any) -> Decimal:
    try:
        if hasattr(value, "item"):
            value = value.item()
        if value is None:
            raise ValueError
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise MetricUnavailableError("unknown", "numeric value", source_attempted="decimal conversion") from error


def _find_statement_row(statement: "pd.DataFrame", row_names: Sequence[str]) -> "pd.Series | None":
    normalized_targets = {_normalize_label(name) for name in row_names}
    for index_value, row in statement.iterrows():
        if _normalize_label(str(index_value)) in normalized_targets:
            return row
    return None


def _chronological_series(series: "pd.Series") -> "pd.Series":
    converted = series.dropna().map(_to_decimal)
    return converted.sort_index()


def _align_pair(
    left: "pd.Series",
    right: "pd.Series",
    metric_name: str,
    ticker: str,
) -> tuple["pd.Series", "pd.Series"]:
    common_index = left.index.intersection(right.index)
    if common_index.empty:
        raise MetricUnavailableError(ticker, metric_name, source_attempted="common annual statement periods")
    return left.loc[common_index], right.loc[common_index]


def _normalize_label(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
