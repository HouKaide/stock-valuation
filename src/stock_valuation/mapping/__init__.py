"""Deterministic mapping of raw yfinance values into valuation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

from stock_valuation.errors import MetricUnavailableError, UnsupportedCurrencyError

if TYPE_CHECKING:
    from pandas import DataFrame, Series

T = TypeVar("T")

SUPPORTED_CURRENCIES = frozenset(
    """
    AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BOV BRL BSD BTN BWP BYN BZD
    CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP
    GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HRK HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS
    KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK
    MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD
    SCR SDG SEK SGD SHP SLE SLL SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX
    USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XAG XAU XBA XBB XBC XBD XCD XDR XOF XPD XPF XPT
    XSU XTS XUA XXX YER ZAR ZMW ZWL
    """.split()
)


@dataclass(frozen=True)
class SourceMetadata:
    """Describe how a normalized metric value was selected.

    Attributes
    ----------
    metric_name:
        Human-readable normalized metric name.
    selected_source:
        Source selected for the returned value, or ``None`` when unavailable.
    fallbacks_attempted:
        Sources checked before selecting or rejecting the value.
    used_override:
        Whether the selected value came from an explicit override.
    """

    metric_name: str
    selected_source: str | None
    fallbacks_attempted: tuple[str, ...] = ()
    used_override: bool = False


def select_mapping_value(
    source: Mapping[str, Any],
    keys: Sequence[str],
    metric_name: str,
    *,
    source_name: str,
    metadata: list[SourceMetadata] | None = None,
) -> Any | None:
    """Select the first present, non-empty value from mapping keys.

    Parameters
    ----------
    source:
        Mapping to inspect.
    keys:
        Keys in fallback order.
    metric_name:
        Metric being selected.
    source_name:
        Prefix used in source metadata.
    metadata:
        Optional metadata collection to append to.

    Returns
    -------
    Any | None
        Selected value, or ``None`` when every key is empty.
    """

    attempted: list[str] = []
    for key in keys:
        source_path = f"{source_name}.{key}"
        attempted.append(source_path)
        value = _mapping_get(source, key)
        if _is_present(value):
            _record(metadata, metric_name, source_path, attempted[:-1])
            return value
    return None


def select_statement_row(
    statement: "DataFrame",
    row_names: Sequence[str],
    metric_name: str,
    *,
    statement_name: str,
    metadata: list[SourceMetadata] | None = None,
) -> "Series | None":
    """Select the first present, non-empty statement row.

    Parameters
    ----------
    statement:
        Raw yfinance statement.
    row_names:
        Row labels in fallback order.
    metric_name:
        Metric being selected.
    statement_name:
        Statement name used in source metadata.
    metadata:
        Optional metadata collection to append to.

    Returns
    -------
    pandas.Series | None
        Selected row, or ``None`` when no candidate row has values.
    """

    normalized_rows = {
        _normalize_label(str(index)): row for index, row in statement.iterrows()
    }
    attempted: list[str] = []
    for row_name in row_names:
        source_path = f"{statement_name}.{row_name}"
        attempted.append(source_path)
        row = normalized_rows.get(_normalize_label(row_name))
        if row is not None and not row.dropna().empty:
            _record(metadata, metric_name, source_path, attempted[:-1])
            return row
    return None


def require_metric(
    value: T | None,
    ticker: str,
    metric_name: str,
    sources_attempted: Sequence[str],
) -> T:
    """Return a required metric or raise a typed error.

    Parameters
    ----------
    value:
        Candidate metric value.
    ticker:
        Ticker associated with the metric.
    metric_name:
        Human-readable metric name.
    sources_attempted:
        Sources checked in fallback order.

    Returns
    -------
    T
        Non-null metric value.

    Raises
    ------
    MetricUnavailableError
        If the metric value is unavailable.
    """

    if value is None:
        source = sources_attempted[0] if sources_attempted else None
        raise MetricUnavailableError(
            ticker,
            metric_name,
            source_attempted=source,
            fallbacks_attempted=sources_attempted[1:],
        )
    return value


def optional_metric(
    value: T | None,
    metric_name: str,
    sources_attempted: Sequence[str],
    *,
    metadata: list[SourceMetadata] | None = None,
) -> T | None:
    """Return an optional metric and record unavailability metadata.

    Parameters
    ----------
    value:
        Candidate metric value.
    metric_name:
        Human-readable metric name.
    sources_attempted:
        Sources checked in fallback order.
    metadata:
        Optional metadata collection to append to.

    Returns
    -------
    T | None
        Original metric value.
    """

    if value is None:
        _record(metadata, metric_name, None, sources_attempted)
    return value


def to_decimal(
    value: Any, ticker: str, metric_name: str, *, absolute: bool = False
) -> Decimal:
    """Convert a finite numeric scalar to ``Decimal``.

    Parameters
    ----------
    value:
        Numeric scalar to convert.
    ticker:
        Ticker associated with the metric.
    metric_name:
        Human-readable metric name.
    absolute:
        Whether to normalize the result to a positive value.

    Returns
    -------
    Decimal
        Exact decimal representation of the scalar.

    Raises
    ------
    MetricUnavailableError
        If the value is null, non-numeric, NaN, or infinite.
    """

    try:
        if hasattr(value, "item"):
            value = value.item()
        if value is None or isinstance(value, bool):
            raise ValueError
        decimal_value = Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError
    except (InvalidOperation, TypeError, ValueError) as error:
        raise MetricUnavailableError(
            ticker,
            metric_name,
            source_attempted="numeric scalar conversion",
        ) from error
    return abs(decimal_value) if absolute else decimal_value


def normalize_currency(value: Any, metric_name: str) -> str:
    """Normalize a supported currency or Yahoo unit currency.

    Parameters
    ----------
    value:
        Raw currency value.
    metric_name:
        Metric requiring the currency.

    Returns
    -------
    str
        Uppercase ISO currency code.

    Raises
    ------
    UnsupportedCurrencyError
        If the value is empty or unsupported.
    """

    raw_currency = "" if value is None else str(value).strip()
    if raw_currency == "GBp" or raw_currency.upper() in {
        "GBX",
        "GBPENCE",
        "GBPENNY",
        "GBPEN",
    }:
        currency = "GBP"
    else:
        currency = raw_currency.upper()
    if currency not in SUPPORTED_CURRENCIES:
        raise UnsupportedCurrencyError(value, metric_name)
    return currency


def map_company_name(
    info: Mapping[str, Any],
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> str:
    """Map a company display name from yfinance metadata.

    Parameters
    ----------
    info:
        Raw yfinance info mapping.
    ticker:
        Normalized ticker fallback.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    str
        Company long name, short name, or ticker.
    """

    value = select_mapping_value(
        info,
        ("longName", "shortName"),
        "company name",
        source_name="info",
        metadata=metadata,
    )
    if value is not None:
        return str(value).strip()
    _record(metadata, "company name", "ticker", ("info.longName", "info.shortName"))
    return ticker


def map_headquarters_country(
    info: Mapping[str, Any],
    ticker: str,
    *,
    override: str | None = None,
    metadata: list[SourceMetadata] | None = None,
) -> str:
    """Map headquarters country with an explicit override fallback.

    Parameters
    ----------
    info:
        Raw yfinance info mapping.
    ticker:
        Ticker associated with the company.
    override:
        Optional country override.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    str
        Headquarters country.
    """

    value = select_mapping_value(
        info,
        ("country",),
        "headquarters country",
        source_name="info",
        metadata=metadata,
    )
    if value is not None:
        return str(value).strip()
    if _is_present(override):
        _record(
            metadata,
            "headquarters country",
            "override",
            ("info.country",),
            used_override=True,
        )
        return str(override).strip()
    return require_metric(
        None, ticker, "headquarters country", ("info.country", "override")
    )


def map_valuation_currency(
    info: Mapping[str, Any],
    fast_info: Mapping[str, Any],
    *,
    override: str | None = None,
    metadata: list[SourceMetadata] | None = None,
) -> str:
    """Map and normalize the financial-statement currency.

    Parameters
    ----------
    info:
        Raw yfinance info mapping.
    fast_info:
        Raw yfinance fast-info mapping.
    override:
        Optional valuation-currency override.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    str
        Normalized valuation currency.
    """

    value = select_mapping_value(
        info,
        ("financialCurrency",),
        "valuation currency",
        source_name="info",
        metadata=metadata,
    )
    if value is not None:
        return normalize_currency(value, "valuation currency")
    value = select_mapping_value(
        fast_info,
        ("currency",),
        "valuation currency",
        source_name="fast_info",
        metadata=metadata,
    )
    if value is not None:
        return normalize_currency(value, "valuation currency")
    if _is_present(override):
        _record(
            metadata,
            "valuation currency",
            "override",
            ("info.financialCurrency", "fast_info.currency"),
            used_override=True,
        )
        return normalize_currency(override, "valuation currency")
    raise UnsupportedCurrencyError(None, "valuation currency")


def map_trading_currency(
    fast_info: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    metadata: list[SourceMetadata] | None = None,
) -> str:
    """Map and normalize the market-price currency.

    Parameters
    ----------
    fast_info:
        Raw yfinance fast-info mapping.
    info:
        Raw yfinance info mapping.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    str
        Normalized trading currency.
    """

    value = select_mapping_value(
        fast_info,
        ("currency",),
        "trading currency",
        source_name="fast_info",
        metadata=metadata,
    )
    if value is not None:
        return normalize_currency(value, "trading currency")
    value = select_mapping_value(
        info,
        ("currency",),
        "trading currency",
        source_name="info",
        metadata=metadata,
    )
    if value is None:
        raise UnsupportedCurrencyError(None, "trading currency")
    return normalize_currency(value, "trading currency")


def map_current_price(
    fast_info: Mapping[str, Any],
    info: Mapping[str, Any],
    history: "DataFrame | None",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> Decimal:
    """Map the latest price from fast info, info, or history.

    Parameters
    ----------
    fast_info:
        Raw yfinance fast-info mapping.
    info:
        Raw yfinance info mapping.
    history:
        Optional price-history table.
    ticker:
        Ticker associated with the price.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    Decimal
        Latest available price.
    """

    value = select_mapping_value(
        fast_info,
        ("last_price", "lastPrice", "lastPriceRaw", "regularMarketPrice"),
        "current price",
        source_name="fast_info",
        metadata=metadata,
    )
    if value is not None:
        return to_decimal(value, ticker, "current price")
    value = select_mapping_value(
        info,
        ("currentPrice", "regularMarketPrice"),
        "current price",
        source_name="info",
        metadata=metadata,
    )
    if value is not None:
        return to_decimal(value, ticker, "current price")
    if history is not None and "Close" in history:
        closes = history["Close"].dropna()
        if not closes.empty:
            _record(
                metadata,
                "current price",
                "history.Close",
                ("fast_info.last_price", "info.currentPrice"),
            )
            return to_decimal(closes.iloc[-1], ticker, "current price")
    return require_metric(
        None,
        ticker,
        "current price",
        ("fast_info.last_price", "info.currentPrice", "history.Close"),
    )


def map_market_cap(
    fast_info: Mapping[str, Any],
    info: Mapping[str, Any],
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> Decimal | None:
    """Map optional market capitalization.

    Parameters
    ----------
    fast_info:
        Raw yfinance fast-info mapping.
    info:
        Raw yfinance info mapping.
    ticker:
        Ticker associated with the market capitalization.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    Decimal | None
        Market capitalization when available.
    """

    value = select_mapping_value(
        fast_info,
        ("market_cap", "marketCap"),
        "market capitalization",
        source_name="fast_info",
        metadata=metadata,
    )
    if value is None:
        value = select_mapping_value(
            info,
            ("marketCap",),
            "market capitalization",
            source_name="info",
            metadata=metadata,
        )
    if value is None:
        return optional_metric(
            None,
            "market capitalization",
            ("fast_info.market_cap", "info.marketCap"),
            metadata=metadata,
        )
    return to_decimal(value, ticker, "market capitalization")


def map_beta(
    info: Mapping[str, Any],
    ticker: str,
    *,
    override: Decimal | None = None,
    required: bool = False,
    metadata: list[SourceMetadata] | None = None,
) -> Decimal | None:
    """Map equity beta with an explicit override fallback.

    Parameters
    ----------
    info:
        Raw yfinance info mapping.
    ticker:
        Ticker associated with beta.
    override:
        Optional explicit beta override.
    required:
        Whether missing beta should raise an error.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    Decimal | None
        Equity beta when available.
    """

    value = select_mapping_value(
        info, ("beta",), "beta", source_name="info", metadata=metadata
    )
    if value is not None:
        return to_decimal(value, ticker, "beta")
    if override is not None:
        _record(metadata, "beta", "override", ("info.beta",), used_override=True)
        return to_decimal(override, ticker, "beta")
    if required:
        return require_metric(None, ticker, "beta", ("info.beta", "override"))
    return optional_metric(None, "beta", ("info.beta", "override"), metadata=metadata)


def map_shares_outstanding(
    fast_info: Mapping[str, Any],
    info: Mapping[str, Any],
    shares_history: "Series | None",
    ticker: str,
    *,
    override: Decimal | None = None,
    metadata: list[SourceMetadata] | None = None,
) -> Decimal:
    """Map shares outstanding through all documented fallbacks.

    Parameters
    ----------
    fast_info:
        Raw yfinance fast-info mapping.
    info:
        Raw yfinance info mapping.
    shares_history:
        Optional shares history returned by yfinance.
    ticker:
        Ticker associated with the share count.
    override:
        Optional explicit shares override.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    Decimal
        Latest shares outstanding.
    """

    value = select_mapping_value(
        fast_info,
        ("shares", "sharesOutstanding"),
        "shares outstanding",
        source_name="fast_info",
        metadata=metadata,
    )
    if value is None:
        value = select_mapping_value(
            info,
            ("sharesOutstanding", "impliedSharesOutstanding"),
            "shares outstanding",
            source_name="info",
            metadata=metadata,
        )
    if value is not None:
        return to_decimal(value, ticker, "shares outstanding")
    if shares_history is not None:
        shares = shares_history.dropna()
        if not shares.empty:
            _record(
                metadata,
                "shares outstanding",
                "shares_history",
                ("fast_info.shares", "info.sharesOutstanding"),
            )
            return to_decimal(shares.iloc[-1], ticker, "shares outstanding")
    if override is not None:
        _record(
            metadata,
            "shares outstanding",
            "override",
            ("fast_info.shares", "info.sharesOutstanding", "shares_history"),
            used_override=True,
        )
        return to_decimal(override, ticker, "shares outstanding")
    return require_metric(
        None,
        ticker,
        "shares outstanding",
        ("fast_info.shares", "info.sharesOutstanding", "shares_history", "override"),
    )


def map_statement_series(
    statement: "DataFrame",
    row_names: Sequence[str],
    ticker: str,
    metric_name: str,
    *,
    statement_name: str,
    absolute: bool = False,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map a required statement row into a chronological Decimal series.

    Parameters
    ----------
    statement:
        Raw yfinance statement.
    row_names:
        Candidate rows in fallback order.
    ticker:
        Ticker associated with the statement.
    metric_name:
        Human-readable metric name.
    statement_name:
        Statement name used in source metadata.
    absolute:
        Whether values should be normalized as positive.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological Decimal metric series.
    """

    row = select_statement_row(
        statement,
        row_names,
        metric_name,
        statement_name=statement_name,
        metadata=metadata,
    )
    row = require_metric(
        row,
        ticker,
        metric_name,
        tuple(f"{statement_name}.{name}" for name in row_names),
    )
    return chronological_series(row, ticker, metric_name, absolute=absolute)


def map_revenue_series(
    income_statement: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map annual revenue.

    Parameters
    ----------
    income_statement:
        Raw annual income statement.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological annual revenue.
    """

    return map_statement_series(
        income_statement,
        ("Total Revenue", "Operating Revenue"),
        ticker,
        "revenue",
        statement_name="income_statement",
        metadata=metadata,
    )


def map_ebit_series(
    income_statement: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map annual EBIT, including the documented derivation fallback.

    Parameters
    ----------
    income_statement:
        Raw annual income statement.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological annual EBIT.
    """

    row = select_statement_row(
        income_statement,
        ("EBIT", "Operating Income"),
        "EBIT",
        statement_name="income_statement",
        metadata=metadata,
    )
    if row is not None:
        return chronological_series(row, ticker, "EBIT")
    revenue = map_revenue_series(income_statement, ticker, metadata=metadata)
    expenses = map_statement_series(
        income_statement,
        ("Operating Expense", "Total Operating Expenses"),
        ticker,
        "operating expenses",
        statement_name="income_statement",
        metadata=metadata,
    )
    revenue, expenses = align_series(revenue, expenses, ticker, "EBIT derivation")
    _record(
        metadata,
        "EBIT",
        "derived: revenue - operating expenses",
        ("income_statement.EBIT", "income_statement.Operating Income"),
    )
    return revenue - expenses


def map_interest_expense_series(
    income_statement: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map positive annual interest expense.

    Parameters
    ----------
    income_statement:
        Raw annual income statement.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological positive interest expense.
    """

    return map_statement_series(
        income_statement,
        ("Interest Expense", "Interest Expense Non Operating"),
        ticker,
        "interest expense",
        statement_name="income_statement",
        absolute=True,
        metadata=metadata,
    )


def map_depreciation_series(
    cashflow: "DataFrame",
    ticker: str,
    *,
    income_statement: "DataFrame | None" = None,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map positive annual depreciation and amortization.

    Parameters
    ----------
    cashflow:
        Raw annual cash-flow statement.
    ticker:
        Ticker associated with the statement.
    income_statement:
        Optional income statement for reconciliation fallback.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological positive depreciation and amortization.
    """

    row_names = (
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
        "Depreciation",
    )
    row = select_statement_row(
        cashflow,
        row_names,
        "depreciation and amortization",
        statement_name="cashflow",
        metadata=metadata,
    )
    if row is None and income_statement is not None:
        row = select_statement_row(
            income_statement,
            ("Reconciled Depreciation",),
            "depreciation and amortization",
            statement_name="income_statement",
            metadata=metadata,
        )
    row = require_metric(
        row,
        ticker,
        "depreciation and amortization",
        tuple(f"cashflow.{name}" for name in row_names)
        + ("income_statement.Reconciled Depreciation",),
    )
    return chronological_series(
        row, ticker, "depreciation and amortization", absolute=True
    )


def map_capex_series(
    cashflow: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map positive annual capital expenditures.

    Parameters
    ----------
    cashflow:
        Raw annual cash-flow statement.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological positive capital expenditures.
    """

    return map_statement_series(
        cashflow,
        (
            "Capital Expenditure",
            "Capital Expenditure Reported",
            "Purchase Of PPE",
            "Net PPE Purchase And Sale",
        ),
        ticker,
        "capital expenditure",
        statement_name="cashflow",
        absolute=True,
        metadata=metadata,
    )


def map_working_capital_change_series(
    cashflow: "DataFrame",
    balance_sheet: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map working-capital increases as positive reinvestment outflows.

    Parameters
    ----------
    cashflow:
        Raw annual cash-flow statement.
    balance_sheet:
        Raw annual balance sheet for derivation fallback.
    ticker:
        Ticker associated with the statements.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological working-capital increases.
    """

    row = select_statement_row(
        cashflow,
        ("Change In Working Capital",),
        "change in non-cash working capital",
        statement_name="cashflow",
        metadata=metadata,
    )
    if row is not None:
        return -chronological_series(row, ticker, "change in non-cash working capital")

    current_assets = map_statement_series(
        balance_sheet,
        ("Current Assets", "Total Current Assets"),
        ticker,
        "current assets",
        statement_name="balance_sheet",
        metadata=metadata,
    )
    cash = map_cash_series(balance_sheet, ticker, metadata=metadata)
    current_liabilities = map_statement_series(
        balance_sheet,
        ("Current Liabilities", "Total Current Liabilities"),
        ticker,
        "current liabilities",
        statement_name="balance_sheet",
        metadata=metadata,
    )
    current_debt = _optional_statement_series(
        balance_sheet,
        ("Current Debt", "Current Debt And Capital Lease Obligation"),
        current_assets.index,
        ticker,
        "current debt",
    )
    assets, cash = align_series(current_assets, cash, ticker, "working capital assets")
    liabilities, current_debt = align_series(
        current_liabilities,
        current_debt,
        ticker,
        "working capital liabilities",
    )
    assets, liabilities = align_series(
        assets - cash, liabilities - current_debt, ticker, "working capital"
    )
    _record(
        metadata,
        "change in non-cash working capital",
        "derived: non-cash current assets - non-debt current liabilities",
        ("cashflow.Change In Working Capital",),
    )
    return (assets - liabilities).diff().dropna()


def map_debt_series(
    balance_sheet: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map positive annual total debt.

    Parameters
    ----------
    balance_sheet:
        Raw annual balance sheet.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological positive total debt.
    """

    row = select_statement_row(
        balance_sheet,
        ("Total Debt",),
        "total debt",
        statement_name="balance_sheet",
        metadata=metadata,
    )
    if row is not None:
        return chronological_series(row, ticker, "total debt", absolute=True)
    long_term = _optional_statement_series(
        balance_sheet,
        ("Long Term Debt", "Long Term Debt And Capital Lease Obligation"),
        balance_sheet.columns,
        ticker,
        "long-term debt",
    )
    current = _optional_statement_series(
        balance_sheet,
        ("Current Debt", "Current Debt And Capital Lease Obligation"),
        balance_sheet.columns,
        ticker,
        "current debt",
    )
    debt = chronological_series(
        long_term + current, ticker, "total debt", absolute=True
    )
    if debt.empty or all(value == Decimal("0") for value in debt):
        return require_metric(
            None, ticker, "total debt", ("balance_sheet.Total Debt", "debt components")
        )
    _record(
        metadata,
        "total debt",
        "derived: long-term debt + current debt",
        ("balance_sheet.Total Debt",),
    )
    return debt


def map_cash_series(
    balance_sheet: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map positive annual cash and cash equivalents.

    Parameters
    ----------
    balance_sheet:
        Raw annual balance sheet.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological positive cash balances.
    """

    return map_statement_series(
        balance_sheet,
        (
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
        ),
        ticker,
        "cash and cash equivalents",
        statement_name="balance_sheet",
        absolute=True,
        metadata=metadata,
    )


def map_non_operating_assets_series(
    balance_sheet: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series | None":
    """Map identifiable non-operating financial assets when present.

    Parameters
    ----------
    balance_sheet:
        Raw annual balance sheet.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series | None
        Chronological positive non-operating assets, or ``None`` when no
        identifiable row exists.
    """

    row = select_statement_row(
        balance_sheet,
        (
            "Investments And Other Financial Assets",
            "Other Investments",
            "Investment In Financial Assets",
        ),
        "non-operating assets",
        statement_name="balance_sheet",
        metadata=metadata,
    )
    if row is None:
        return None
    return chronological_series(row, ticker, "non-operating assets", absolute=True)


def map_minority_interest_series(
    balance_sheet: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series | None":
    """Map minority interest as a positive claim when present.

    Parameters
    ----------
    balance_sheet:
        Raw annual balance sheet.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series | None
        Chronological positive minority interest, or ``None`` when unavailable.
    """

    row = select_statement_row(
        balance_sheet,
        (
            "Minority Interest",
            "Non Controlling Interest In Consolidated Entity",
        ),
        "minority interest",
        statement_name="balance_sheet",
        metadata=metadata,
    )
    if row is None:
        return None
    return chronological_series(row, ticker, "minority interest", absolute=True)


def map_book_equity_series(
    balance_sheet: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map annual book equity.

    Parameters
    ----------
    balance_sheet:
        Raw annual balance sheet.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological book equity.
    """

    return map_statement_series(
        balance_sheet,
        (
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ),
        ticker,
        "book equity",
        statement_name="balance_sheet",
        metadata=metadata,
    )


def map_invested_capital_series(
    balance_sheet: "DataFrame",
    ticker: str,
    *,
    metadata: list[SourceMetadata] | None = None,
) -> "Series":
    """Map annual invested capital with a debt-plus-equity-minus-cash fallback.

    Parameters
    ----------
    balance_sheet:
        Raw annual balance sheet.
    ticker:
        Ticker associated with the statement.
    metadata:
        Optional source metadata collection.

    Returns
    -------
    pandas.Series
        Chronological invested capital.
    """

    row = select_statement_row(
        balance_sheet,
        ("Invested Capital",),
        "invested capital",
        statement_name="balance_sheet",
        metadata=metadata,
    )
    if row is not None:
        return chronological_series(row, ticker, "invested capital", absolute=True)
    debt = map_debt_series(balance_sheet, ticker, metadata=metadata)
    equity = map_book_equity_series(balance_sheet, ticker, metadata=metadata)
    cash = map_cash_series(balance_sheet, ticker, metadata=metadata)
    debt, equity = align_series(debt, equity, ticker, "invested capital")
    debt, cash = align_series(debt, cash, ticker, "invested capital")
    _record(
        metadata,
        "invested capital",
        "derived: debt + book equity - cash",
        ("balance_sheet.Invested Capital",),
    )
    return debt + equity - cash


def chronological_series(
    series: "Series",
    ticker: str,
    metric_name: str,
    *,
    absolute: bool = False,
) -> "Series":
    """Convert a raw annual series to chronological Decimal values.

    Parameters
    ----------
    series:
        Raw annual series.
    ticker:
        Ticker associated with the series.
    metric_name:
        Human-readable metric name.
    absolute:
        Whether values should be normalized as positive.

    Returns
    -------
    pandas.Series
        Chronological Decimal series preserving period labels.
    """

    converted = series.dropna().map(
        lambda value: to_decimal(value, ticker, metric_name, absolute=absolute)
    )
    return converted.sort_index()


def align_series(
    left: "Series",
    right: "Series",
    ticker: str,
    metric_name: str,
) -> tuple["Series", "Series"]:
    """Align related annual series to common chronological periods.

    Parameters
    ----------
    left:
        First chronological annual series.
    right:
        Second chronological annual series.
    ticker:
        Ticker associated with the series.
    metric_name:
        Human-readable alignment context.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        Series restricted to common chronological periods.
    """

    common_index = left.index.intersection(right.index).sort_values()
    if common_index.empty:
        raise MetricUnavailableError(
            ticker,
            metric_name,
            source_attempted="common annual statement periods",
        )
    return left.loc[common_index], right.loc[common_index]


def _optional_statement_series(
    statement: "DataFrame",
    row_names: Sequence[str],
    fallback_index: Iterable[Any],
    ticker: str,
    metric_name: str,
) -> "Series":
    row = select_statement_row(
        statement,
        row_names,
        metric_name,
        statement_name="balance_sheet",
    )
    if row is None:
        index = list(fallback_index)
        return pd.Series([Decimal("0")] * len(index), index=index, dtype="object")
    return chronological_series(row, ticker, metric_name, absolute=True)


def _mapping_get(source: Mapping[str, Any], key: str) -> Any | None:
    try:
        return source[key]
    except (KeyError, TypeError):
        return getattr(source, key, None)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _record(
    metadata: list[SourceMetadata] | None,
    metric_name: str,
    selected_source: str | None,
    fallbacks_attempted: Iterable[str],
    *,
    used_override: bool = False,
) -> None:
    if metadata is not None:
        metadata.append(
            SourceMetadata(
                metric_name=metric_name,
                selected_source=selected_source,
                fallbacks_attempted=tuple(fallbacks_attempted),
                used_override=used_override,
            )
        )


def _normalize_label(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


__all__ = [
    "SourceMetadata",
    "align_series",
    "chronological_series",
    "map_beta",
    "map_book_equity_series",
    "map_capex_series",
    "map_cash_series",
    "map_company_name",
    "map_current_price",
    "map_debt_series",
    "map_depreciation_series",
    "map_ebit_series",
    "map_headquarters_country",
    "map_interest_expense_series",
    "map_invested_capital_series",
    "map_market_cap",
    "map_minority_interest_series",
    "map_non_operating_assets_series",
    "map_revenue_series",
    "map_shares_outstanding",
    "map_statement_series",
    "map_trading_currency",
    "map_valuation_currency",
    "map_working_capital_change_series",
    "normalize_currency",
    "optional_metric",
    "require_metric",
    "select_mapping_value",
    "select_statement_row",
    "to_decimal",
]
