"""Exception raised when a ticker symbol cannot be resolved."""

from stock_valuation.errors.stock_valuation_error import StockValuationError


class TickerNotFoundError(StockValuationError):
    """Raised when a requested ticker symbol cannot be resolved."""

    def __init__(self, symbol: str) -> None:
        """Initialize the error with the missing ticker symbol.

        Args:
            symbol: The ticker symbol that could not be resolved.
        """

        self.symbol = symbol
        super().__init__(f"Ticker '{symbol}' could not be found.")
