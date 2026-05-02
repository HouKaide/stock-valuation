"""Exception raised when valuation assumptions are invalid."""

from stock_valuation.errors.stock_valuation_error import StockValuationError


class InvalidAssumptionsError(StockValuationError):
    """Raised when valuation assumptions fail validation."""
