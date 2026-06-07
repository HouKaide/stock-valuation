"""Ticker-bound Yahoo Finance client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import yfinance as yf

from stock_valuation.errors import (
    MarketDataUnavailableError,
    MetricUnavailableError,
    StatementUnavailableError,
    TickerNotFoundError,
)

if TYPE_CHECKING:
    import pandas as pd


LOOKUP_METHODS: dict[str, str] = {
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
    """Ticker-bound adapter for raw yfinance data access.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol. Input is trimmed and uppercased when used.
    """

    ticker: str
    _ticker: Any | None = field(default=None, init=False, repr=False)

    def normalized_ticker(self) -> str:
        """Return the normalized ticker symbol.

        Returns
        -------
        str
            Uppercase ticker symbol with surrounding whitespace removed.

        Raises
        ------
        TickerNotFoundError
            If the ticker is empty after normalization.
        """

        normalized = self.ticker.strip().upper()
        if not normalized:
            raise TickerNotFoundError(self.ticker, source_attempted="ticker input")
        return normalized

    def get_ticker(self) -> Any:
        """Return the cached yfinance ticker for this client.

        Returns
        -------
        Any
            Cached `yfinance.Ticker` instance.

        Raises
        ------
        TickerNotFoundError
            If ticker construction fails.
        """

        normalized = self.normalized_ticker()
        if self._ticker is None:
            try:
                self._ticker = yf.Ticker(normalized)
            except Exception as error:
                raise TickerNotFoundError(normalized, source_attempted="yfinance.Ticker") from error
        return self._ticker

    def get_info(self) -> dict[str, Any]:
        """Return raw ticker metadata.

        Returns
        -------
        dict[str, Any]
            Raw metadata returned by yfinance.

        Raises
        ------
        TickerNotFoundError
            If metadata is empty or retrieval fails.
        """

        ticker_symbol = self.normalized_ticker()
        try:
            info = self.get_ticker().get_info()
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise TickerNotFoundError(ticker_symbol, source_attempted="yfinance.Ticker.get_info") from error

        if not info:
            raise TickerNotFoundError(ticker_symbol, source_attempted="yfinance.Ticker.get_info")
        return info

    def get_fast_info(self) -> Mapping[str, Any]:
        """Return raw fast market metadata.

        Returns
        -------
        Mapping[str, Any]
            Raw fast-info mapping returned by yfinance.

        Raises
        ------
        MarketDataUnavailableError
            If fast-info retrieval fails or returns no fields.
        """

        ticker_symbol = self.normalized_ticker()
        try:
            fast_info = self.get_ticker().get_fast_info()
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise MarketDataUnavailableError(
                ticker_symbol,
                "fast info",
                source_attempted="yfinance.Ticker.get_fast_info",
            ) from error

        if not fast_info:
            raise MarketDataUnavailableError(
                ticker_symbol,
                "fast info",
                source_attempted="yfinance.Ticker.get_fast_info",
            )
        return fast_info

    def get_income_statement(self, freq: str = "yearly") -> "pd.DataFrame":
        """Return raw income statement data.

        Parameters
        ----------
        freq:
            yfinance statement frequency.

        Returns
        -------
        pandas.DataFrame
            Raw yfinance income statement.
        """

        return self._get_statement(
            "income statement",
            "yfinance.Ticker.get_income_stmt",
            lambda ticker: ticker.get_income_stmt(pretty=True, freq=freq),
        )

    def get_balance_sheet(self, freq: str = "yearly") -> "pd.DataFrame":
        """Return raw balance sheet data.

        Parameters
        ----------
        freq:
            yfinance statement frequency.

        Returns
        -------
        pandas.DataFrame
            Raw yfinance balance sheet.
        """

        return self._get_statement(
            "balance sheet",
            "yfinance.Ticker.get_balance_sheet",
            lambda ticker: ticker.get_balance_sheet(pretty=True, freq=freq),
        )

    def get_cashflow(self, freq: str = "yearly") -> "pd.DataFrame":
        """Return raw cash flow statement data.

        Parameters
        ----------
        freq:
            yfinance statement frequency.

        Returns
        -------
        pandas.DataFrame
            Raw yfinance cash flow statement.
        """

        return self._get_statement(
            "cash flow statement",
            "yfinance.Ticker.get_cashflow",
            lambda ticker: ticker.get_cashflow(pretty=True, freq=freq),
        )

    def get_shares_full(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> "pd.Series | None":
        """Return raw shares outstanding history.

        Parameters
        ----------
        start:
            Optional inclusive start date passed to yfinance.
        end:
            Optional inclusive end date passed to yfinance.

        Returns
        -------
        pandas.Series | None
            Raw yfinance shares history, or `None` when yfinance has no history.
        """

        ticker_symbol = self.normalized_ticker()
        try:
            shares = self.get_ticker().get_shares_full(start=start, end=end)
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise MarketDataUnavailableError(
                ticker_symbol,
                "shares history",
                source_attempted="yfinance.Ticker.get_shares_full",
            ) from error

        if getattr(shares, "empty", False):
            raise MarketDataUnavailableError(
                ticker_symbol,
                "shares history",
                source_attempted="yfinance.Ticker.get_shares_full",
            )
        return shares

    def get_history(
        self,
        period: str = "10y",
        interval: str = "1mo",
        auto_adjust: bool = True,
    ) -> "pd.DataFrame":
        """Return raw historical price data.

        Parameters
        ----------
        period:
            History period accepted by yfinance.
        interval:
            History interval accepted by yfinance.
        auto_adjust:
            Whether yfinance should auto-adjust prices.

        Returns
        -------
        pandas.DataFrame
            Raw yfinance price history.
        """

        ticker_symbol = self.normalized_ticker()
        try:
            history = self.get_ticker().history(period=period, interval=interval, auto_adjust=auto_adjust)
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise MarketDataUnavailableError(
                ticker_symbol,
                "price history",
                source_attempted="yfinance.Ticker.history",
            ) from error

        if getattr(history, "empty", False):
            raise MarketDataUnavailableError(
                ticker_symbol,
                "price history",
                source_attempted="yfinance.Ticker.history",
            )
        return history

    def lookup_instrument(self, query: str, instrument_type: str | None = None) -> "pd.DataFrame":
        """Return raw yfinance lookup candidates for a query.

        Parameters
        ----------
        query:
            Search query passed to `yfinance.Lookup`.
        instrument_type:
            Optional yfinance lookup type. Supported values are `stock`,
            `equity`, `mutualfund`, `etf`, `index`, `future`, `currency`,
            and `cryptocurrency`.

        Returns
        -------
        pandas.DataFrame
            Raw lookup candidates returned by yfinance.
        """

        normalized_query = query.strip()
        if not normalized_query:
            raise MetricUnavailableError(
                self.normalized_ticker(),
                "instrument lookup",
                source_attempted="lookup query",
                suggested_override="Provide a non-empty lookup query.",
            )

        normalized_type = None
        method_name = "get_all"
        if instrument_type is not None:
            normalized_type = instrument_type.strip().lower()
            method_name = LOOKUP_METHODS.get(normalized_type, "")
            if not method_name:
                raise MetricUnavailableError(
                    self.normalized_ticker(),
                    "instrument lookup",
                    source_attempted=f"yfinance.Lookup.{normalized_type}",
                    fallbacks_attempted=tuple(LOOKUP_METHODS),
                    suggested_override="Use a supported lookup instrument type.",
                )

        source_attempted = f"yfinance.Lookup.{method_name}"
        try:
            lookup = yf.Lookup(normalized_query)
            candidates = getattr(lookup, method_name)()
        except Exception as error:
            raise MetricUnavailableError(
                self.normalized_ticker(),
                "instrument lookup",
                source_attempted=source_attempted,
            ) from error

        if getattr(candidates, "empty", False):
            raise MetricUnavailableError(
                self.normalized_ticker(),
                "instrument lookup",
                source_attempted=source_attempted,
                fallbacks_attempted=() if normalized_type is None else ("all",),
            )
        return candidates

    def _get_statement(
        self,
        statement_name: str,
        source_attempted: str,
        fetch_statement: Callable[[Any], "pd.DataFrame"],
    ) -> "pd.DataFrame":
        """Fetch a raw statement and convert empty responses into typed errors."""

        ticker_symbol = self.normalized_ticker()
        try:
            statement = fetch_statement(self.get_ticker())
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise StatementUnavailableError(
                ticker_symbol,
                statement_name,
                source_attempted=source_attempted,
            ) from error

        if getattr(statement, "empty", False):
            raise StatementUnavailableError(
                ticker_symbol,
                statement_name,
                source_attempted=source_attempted,
            )
        return statement
