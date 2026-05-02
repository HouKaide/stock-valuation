"""Exception raised when a financial statement is unavailable."""

from stock_valuation.errors.stock_valuation_error import StockValuationError


class StatementUnavailableError(StockValuationError):
    """Raised when a required financial statement is unavailable."""

    def __init__(self, symbol: str, statement_name: str) -> None:
        """Initialize the error with the missing statement details.

        Args:
            symbol: The ticker symbol associated with the request.
            statement_name: The financial statement that was unavailable.
        """

        self.symbol = symbol
        self.statement_name = statement_name
        super().__init__(f"{statement_name} is unavailable for ticker '{symbol}'.")
