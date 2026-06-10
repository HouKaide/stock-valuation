"""Exception raised when a currency cannot be normalized."""

from __future__ import annotations

from stock_valuation.errors.stock_valuation_error import StockValuationError


class UnsupportedCurrencyError(StockValuationError):
    """Raised when a currency value is missing or unsupported.

    Parameters
    ----------
    currency:
        Raw currency value that could not be normalized.
    metric_name:
        Metric requiring the currency.
    """

    def __init__(self, currency: object, metric_name: str) -> None:
        self.currency = currency
        self.metric_name = metric_name
        super().__init__(f"Unsupported currency {currency!r} for {metric_name}.")
