"""Exception raised when market data cannot be retrieved."""

from stock_valuation.errors.stock_valuation_error import StockValuationError


class MarketDataUnavailableError(StockValuationError):
    """Raised when required market data cannot be retrieved."""

    def __init__(self, symbol: str, data_name: str) -> None:
        """Initialize the error with the missing market data details.

        Args:
            symbol: The ticker symbol associated with the request.
            data_name: The market data surface that was unavailable.
        """

        self.symbol = symbol
        self.data_name = data_name
        super().__init__(f"{data_name} is unavailable for ticker '{symbol}'.")
