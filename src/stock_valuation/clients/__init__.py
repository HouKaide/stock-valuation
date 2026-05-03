"""External data retrieval adapters.

This package owns transport-specific integrations and translates upstream
failures into shared application errors without performing valuation logic.
"""

from stock_valuation.clients.yfinance_client import YFinanceClient

__all__ = [
    "YFinanceClient",
]
