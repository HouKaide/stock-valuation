"""Exception raised when market data cannot be retrieved."""

from __future__ import annotations

from collections.abc import Sequence

from stock_valuation.errors.stock_valuation_error import StockValuationError


class MarketDataUnavailableError(StockValuationError):
    """Raised when required market data cannot be retrieved.

    Parameters
    ----------
    ticker:
        Ticker symbol associated with the missing data.
    data_name:
        Market data surface that was unavailable.
    source_attempted:
        Source attempted for the data.
    fallbacks_attempted:
        Fallback sources attempted before failing.
    suggested_override:
        Optional user action that can resolve the failure.
    """

    def __init__(
        self,
        ticker: str,
        data_name: str,
        *,
        source_attempted: str | None = None,
        fallbacks_attempted: Sequence[str] | None = None,
        suggested_override: str | None = None,
    ) -> None:
        self.symbol = ticker
        self.data_name = data_name
        super().__init__(
            f"{data_name} is unavailable for ticker '{ticker}'.",
            ticker=ticker,
            metric=data_name,
            source_attempted=source_attempted,
            fallbacks_attempted=fallbacks_attempted or (),
            suggested_override=suggested_override,
        )
