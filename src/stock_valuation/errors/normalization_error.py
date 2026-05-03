"""Exception raised when raw data cannot be normalized safely."""

from stock_valuation.errors.stock_valuation_error import StockValuationError


class NormalizationError(StockValuationError):
    """Raised when raw upstream data cannot be normalized safely."""
