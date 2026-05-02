"""Yahoo Finance client that owns all direct `yfinance` access."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import TYPE_CHECKING, Any

from stock_valuation.clients.ticker_factory import TickerFactory, default_ticker_factory
from stock_valuation.clients.ticker_protocol import TickerProtocol
from stock_valuation.errors import (
    MarketDataUnavailableError,
    StatementUnavailableError,
    TickerNotFoundError,
)

if TYPE_CHECKING:
    import pandas as pd


class YFinanceClient:
    """Yahoo Finance access layer for raw market and statement retrieval."""

    def __init__(self, ticker_factory: TickerFactory | None = None) -> None:
        """Initialize the client with an optional ticker factory.

        Args:
            ticker_factory: Optional factory used to create ticker objects.
                The default factory builds `yfinance.Ticker` instances.
        """

        self._ticker_factory = ticker_factory or default_ticker_factory
        self._tickers: dict[str, TickerProtocol] = {}

    def get_ticker(self, symbol: str) -> TickerProtocol:
        """Return a cached Yahoo ticker instance for the given symbol.

        Args:
            symbol: User-provided ticker symbol.

        Returns:
            The cached or newly created ticker instance.

        Raises:
            TickerNotFoundError: If the symbol is empty or ticker creation fails.
        """

        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol not in self._tickers:
            try:
                self._tickers[normalized_symbol] = self._ticker_factory(normalized_symbol)
            except Exception as error:
                raise TickerNotFoundError(normalized_symbol) from error

        return self._tickers[normalized_symbol]

    def get_info(self, symbol: str) -> dict[str, Any]:
        """Return raw ticker metadata from Yahoo Finance.

        Args:
            symbol: Ticker symbol to query.

        Returns:
            A raw metadata dictionary from Yahoo Finance.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            MarketDataUnavailableError: If metadata retrieval fails or returns no data.
        """

        ticker_symbol = self._normalize_symbol(symbol)

        try:
            info = self.get_ticker(ticker_symbol).info
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise MarketDataUnavailableError(ticker_symbol, "ticker info") from error

        if not info:
            raise TickerNotFoundError(ticker_symbol)

        return info

    def get_fast_info(self, symbol: str) -> dict[str, Any]:
        """Return lightweight market data fields for the ticker.

        Args:
            symbol: Ticker symbol to query.

        Returns:
            A raw dictionary of fast market data fields.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            MarketDataUnavailableError: If fast info retrieval fails or is empty.
        """

        ticker_symbol = self._normalize_symbol(symbol)

        try:
            fast_info = dict(self.get_ticker(ticker_symbol).fast_info)
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise MarketDataUnavailableError(ticker_symbol, "fast info") from error

        if not fast_info:
            raise MarketDataUnavailableError(ticker_symbol, "fast info")

        return fast_info

    def get_history(
        self,
        symbol: str,
        *,
        period: str = "10y",
        interval: str = "1mo",
        auto_adjust: bool = True,
    ) -> "pd.DataFrame":
        """Return raw historical price data for the ticker.

        Args:
            symbol: Ticker symbol to query.
            period: History period accepted by `yfinance`.
            interval: History interval accepted by `yfinance`.
            auto_adjust: Whether to return adjusted prices.

        Returns:
            A raw historical price `DataFrame`.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            MarketDataUnavailableError: If history retrieval fails or is empty.
        """

        ticker_symbol = self._normalize_symbol(symbol)

        try:
            history = self.get_ticker(ticker_symbol).history(
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
            )
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise MarketDataUnavailableError(ticker_symbol, "price history") from error

        if getattr(history, "empty", False):
            raise MarketDataUnavailableError(ticker_symbol, "price history")

        return history

    def get_income_statement(self, symbol: str, *, freq: str = "yearly") -> "pd.DataFrame":
        """Return raw income statement data for the ticker.

        Args:
            symbol: Ticker symbol to query.
            freq: Statement frequency accepted by `yfinance`.

        Returns:
            A raw income statement `DataFrame`.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            StatementUnavailableError: If the statement retrieval fails or is empty.
        """

        return self._get_statement(
            symbol,
            statement_name="income statement",
            fetch_statement=lambda ticker: ticker.get_income_stmt(pretty=True, freq=freq),
        )

    def get_balance_sheet(self, symbol: str, *, freq: str = "yearly") -> "pd.DataFrame":
        """Return raw balance sheet data for the ticker.

        Args:
            symbol: Ticker symbol to query.
            freq: Statement frequency accepted by `yfinance`.

        Returns:
            A raw balance sheet `DataFrame`.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            StatementUnavailableError: If the statement retrieval fails or is empty.
        """

        return self._get_statement(
            symbol,
            statement_name="balance sheet",
            fetch_statement=lambda ticker: ticker.get_balance_sheet(pretty=True, freq=freq),
        )

    def get_cashflow(self, symbol: str, *, freq: str = "yearly") -> "pd.DataFrame":
        """Return raw cash flow statement data for the ticker.

        Args:
            symbol: Ticker symbol to query.
            freq: Statement frequency accepted by `yfinance`.

        Returns:
            A raw cash flow `DataFrame`.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            StatementUnavailableError: If the statement retrieval fails or is empty.
        """

        return self._get_statement(
            symbol,
            statement_name="cash flow statement",
            fetch_statement=lambda ticker: ticker.get_cashflow(pretty=True, freq=freq),
        )

    def get_shares_full(
        self,
        symbol: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> "pd.Series | None":
        """Return raw shares outstanding history for the ticker.

        Args:
            symbol: Ticker symbol to query.
            start: Optional inclusive start date.
            end: Optional inclusive end date.

        Returns:
            A raw `Series` of shares outstanding history, or `None` if Yahoo
            exposes no shares history for the ticker.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            MarketDataUnavailableError: If Yahoo retrieval fails.
        """

        ticker_symbol = self._normalize_symbol(symbol)

        try:
            return self.get_ticker(ticker_symbol).get_shares_full(start=start, end=end)
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise MarketDataUnavailableError(ticker_symbol, "shares history") from error

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize user input into the cache key used for ticker lookup.

        Args:
            symbol: User-provided ticker symbol.

        Returns:
            The normalized uppercase ticker symbol.

        Raises:
            TickerNotFoundError: If the symbol is empty after normalization.
        """

        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise TickerNotFoundError(symbol)
        return normalized_symbol

    def _get_statement(
        self,
        symbol: str,
        *,
        statement_name: str,
        fetch_statement: Callable[[TickerProtocol], "pd.DataFrame"],
    ) -> "pd.DataFrame":
        """Fetch a statement and map empty or failed responses to shared errors.

        Args:
            symbol: Ticker symbol to query.
            statement_name: Human-readable name for the statement.
            fetch_statement: Callable that fetches the statement from a ticker.

        Returns:
            A raw financial statement `DataFrame`.

        Raises:
            TickerNotFoundError: If the ticker cannot be resolved.
            StatementUnavailableError: If retrieval fails or returns no rows.
        """

        ticker_symbol = self._normalize_symbol(symbol)

        try:
            statement = fetch_statement(self.get_ticker(ticker_symbol))
        except TickerNotFoundError:
            raise
        except Exception as error:
            raise StatementUnavailableError(ticker_symbol, statement_name) from error

        if getattr(statement, "empty", False):
            raise StatementUnavailableError(ticker_symbol, statement_name)

        return statement
