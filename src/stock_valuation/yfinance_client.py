"""Ticker-bound access to raw yfinance data."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

from stock_valuation.errors import (
    MetricUnavailableError,
    StatementUnavailableError,
    StockValuationError,
    TickerNotFoundError,
)

LOOKUP_METHODS: Mapping[str | None, str] = {
    None: "get_all",
    "stock": "get_stock",
    "equity": "get_stock",
    "mutualfund": "get_mutualfund",
    "etf": "get_etf",
    "index": "get_index",
    "future": "get_future",
    "currency": "get_currency",
    "cryptocurrency": "get_cryptocurrency",
}


@dataclass
class YFinanceClient:
    """Client that isolates raw yfinance access for one ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol to resolve through yfinance. Input is normalized before
        yfinance objects are constructed.
    """

    ticker: str
    _ticker: Any | None = field(default=None, init=False, repr=False)

    def normalized_ticker(self) -> str:
        """Return the ticker normalized for yfinance calls.

        Returns
        -------
        str
            The stripped, uppercased ticker symbol.

        Raises
        ------
        TickerNotFoundError
            If the normalized ticker is empty.
        """
        normalized = self.ticker.strip().upper()
        if not normalized:
            raise TickerNotFoundError(
                "Ticker is required.",
                ticker=self.ticker,
                metric_name="ticker",
                source="YFinanceClient.normalized_ticker",
                suggested_override="ticker",
            )
        return normalized

    def get_ticker(self) -> yf.Ticker:
        """Return the cached yfinance ticker object.

        Returns
        -------
        yfinance.Ticker
            The cached ticker instance for this client.

        Raises
        ------
        TickerNotFoundError
            If yfinance cannot construct the ticker object.
        """
        if self._ticker is None:
            normalized = self.normalized_ticker()
            try:
                self._ticker = yf.Ticker(normalized)
            except Exception as exc:
                raise TickerNotFoundError(
                    "Ticker could not be resolved.",
                    ticker=normalized,
                    metric_name="ticker",
                    source="yfinance.Ticker",
                    suggested_override="ticker",
                ) from exc
        return self._ticker

    def get_info(self) -> dict[str, Any]:
        """Return raw yfinance company information.

        Returns
        -------
        dict[str, Any]
            Raw data returned by ``yfinance.Ticker.get_info``.
        """
        source = "yfinance.Ticker.get_info"
        try:
            info = self.get_ticker().get_info()
        except MetricUnavailableError:
            raise
        except Exception as exc:
            raise self._metric_error("info", source) from exc
        return self._require_mapping(info, self._metric_error("info", source))

    def get_fast_info(self) -> Mapping[str, Any]:
        """Return raw yfinance fast market information.

        Returns
        -------
        Mapping[str, Any]
            Raw data returned by ``yfinance.Ticker.get_fast_info``.
        """
        source = "yfinance.Ticker.get_fast_info"
        try:
            fast_info = self.get_ticker().get_fast_info()
        except MetricUnavailableError:
            raise
        except Exception as exc:
            raise self._metric_error("fast_info", source) from exc
        return self._require_mapping(fast_info, self._metric_error("fast_info", source))

    def get_income_statement(self, freq: str = "yearly") -> pd.DataFrame:
        """Return the raw yfinance income statement.

        Parameters
        ----------
        freq:
            yfinance statement frequency such as ``"yearly"``, ``"quarterly"``,
            or ``"trailing"``.
        """
        source = "yfinance.Ticker.get_income_stmt"
        try:
            statement = self.get_ticker().get_income_stmt(pretty=True, freq=freq)
        except Exception as exc:
            raise self._statement_error("income_statement", source) from exc
        return self._require_dataframe(statement, self._statement_error("income_statement", source))

    def get_balance_sheet(self, freq: str = "yearly") -> pd.DataFrame:
        """Return the raw yfinance balance sheet.

        Parameters
        ----------
        freq:
            yfinance statement frequency such as ``"yearly"`` or ``"quarterly"``.
        """
        source = "yfinance.Ticker.get_balance_sheet"
        try:
            statement = self.get_ticker().get_balance_sheet(pretty=True, freq=freq)
        except Exception as exc:
            raise self._statement_error("balance_sheet", source) from exc
        return self._require_dataframe(statement, self._statement_error("balance_sheet", source))

    def get_cashflow(self, freq: str = "yearly") -> pd.DataFrame:
        """Return the raw yfinance cash-flow statement.

        Parameters
        ----------
        freq:
            yfinance statement frequency such as ``"yearly"`` or ``"quarterly"``.
        """
        source = "yfinance.Ticker.get_cashflow"
        try:
            statement = self.get_ticker().get_cashflow(pretty=True, freq=freq)
        except Exception as exc:
            raise self._statement_error("cashflow", source) from exc
        return self._require_dataframe(statement, self._statement_error("cashflow", source))

    def get_shares_full(self, start: date | None = None, end: date | None = None) -> pd.Series | None:
        """Return raw yfinance full shares history.

        yfinance returns ``None`` when shares history is unavailable, so this
        method preserves ``None`` instead of converting it to an error.

        Parameters
        ----------
        start:
            Optional start date passed through to yfinance.
        end:
            Optional end date passed through to yfinance.
        """
        source = "yfinance.Ticker.get_shares_full"
        try:
            shares = self.get_ticker().get_shares_full(start=start, end=end)
        except Exception as exc:
            raise self._metric_error("shares_full", source) from exc
        if shares is None:
            return None
        return self._require_series(shares, self._metric_error("shares_full", source))

    def get_history(self, period: str = "10y", interval: str = "1mo", auto_adjust: bool = True) -> pd.DataFrame:
        """Return raw yfinance price history.

        Parameters
        ----------
        period:
            yfinance history period.
        interval:
            yfinance history interval.
        auto_adjust:
            Whether yfinance should auto-adjust price data.
        """
        source = "yfinance.Ticker.history"
        try:
            history = self.get_ticker().history(period=period, interval=interval, auto_adjust=auto_adjust)
        except Exception as exc:
            raise self._metric_error("history", source) from exc
        return self._require_dataframe(history, self._metric_error("history", source))

    def lookup_instrument(self, query: str, instrument_type: str | None = None) -> pd.DataFrame:
        """Return raw yfinance lookup candidates.

        Parameters
        ----------
        query:
            Search query to pass to yfinance lookup.
        instrument_type:
            Optional instrument type used to select the yfinance lookup method.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise self._lookup_error("query", "yfinance.Lookup")

        normalized_type = instrument_type.strip().lower() if instrument_type is not None else None
        method_name = LOOKUP_METHODS.get(normalized_type)
        if method_name is None:
            raise self._lookup_error("instrument_type", "yfinance.Lookup")

        source = f"yfinance.Lookup.{method_name}"
        try:
            lookup = yf.Lookup(normalized_query)
            candidates = getattr(lookup, method_name)()
        except Exception as exc:
            raise self._lookup_error("lookup_instrument", source) from exc
        return self._require_dataframe(candidates, self._lookup_error("lookup_instrument", source))

    def _metric_error(self, metric_name: str, source: str) -> MetricUnavailableError:
        return MetricUnavailableError(
            f"{metric_name} is unavailable.",
            ticker=self._safe_ticker_context(),
            metric_name=metric_name,
            source=source,
        )

    def _statement_error(self, statement_name: str, source: str) -> StatementUnavailableError:
        return StatementUnavailableError(
            f"{statement_name} is unavailable.",
            ticker=self._safe_ticker_context(),
            statement_name=statement_name,
            source=source,
        )

    @staticmethod
    def _lookup_error(metric_name: str, source: str) -> MetricUnavailableError:
        return MetricUnavailableError(f"{metric_name} is unavailable.", metric_name=metric_name, source=source)

    def _safe_ticker_context(self) -> str:
        try:
            return self.normalized_ticker()
        except TickerNotFoundError:
            return self.ticker

    @staticmethod
    def _require_dataframe(data: pd.DataFrame, error: StockValuationError) -> pd.DataFrame:
        if data.empty:
            raise error
        return data

    @staticmethod
    def _require_series(data: pd.Series, error: StockValuationError) -> pd.Series:
        if data.empty:
            raise error
        return data

    @staticmethod
    def _require_mapping(data: Mapping[str, Any], error: StockValuationError) -> Mapping[str, Any]:
        if len(data) == 0:
            raise error
        return data
