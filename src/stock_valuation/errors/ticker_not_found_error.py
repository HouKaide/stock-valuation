"""Exception raised when a ticker symbol cannot be resolved."""

from stock_valuation.errors.stock_valuation_error import StockValuationError


class TickerNotFoundError(StockValuationError):
    """Raised when a requested ticker symbol cannot be resolved.

    Parameters
    ----------
    ticker:
        Ticker symbol that could not be resolved.
    source_attempted:
        Source attempted while resolving the ticker.
    suggested_override:
        Optional user action that can resolve the failure.
    """

    def __init__(
        self,
        ticker: str,
        *,
        source_attempted: str | None = None,
        suggested_override: str | None = "Provide a valid Yahoo Finance ticker.",
    ) -> None:
        self.ticker = ticker
        self.symbol = ticker
        self.source_attempted = source_attempted
        self.fallbacks_attempted: tuple[str, ...] = ()
        self.suggested_override = suggested_override
        super().__init__(f"Ticker '{ticker}' could not be found.")
