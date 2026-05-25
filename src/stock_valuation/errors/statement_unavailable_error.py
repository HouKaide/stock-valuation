"""Exception raised when a financial statement is unavailable."""

from __future__ import annotations

from collections.abc import Sequence

from stock_valuation.errors.stock_valuation_error import StockValuationError


class StatementUnavailableError(StockValuationError):
    """Raised when a required financial statement is unavailable.

    Parameters
    ----------
    ticker:
        Ticker symbol associated with the missing statement.
    statement_name:
        Financial statement that was unavailable.
    source_attempted:
        Source attempted for the statement.
    fallbacks_attempted:
        Fallback sources attempted before failing.
    suggested_override:
        Optional user action that can resolve the failure.
    """

    def __init__(
        self,
        ticker: str,
        statement_name: str,
        *,
        source_attempted: str | None = None,
        fallbacks_attempted: Sequence[str] | None = None,
        suggested_override: str | None = None,
    ) -> None:
        self.ticker = ticker
        self.symbol = ticker
        self.statement_name = statement_name
        self.source_attempted = source_attempted
        self.fallbacks_attempted = tuple(fallbacks_attempted or ())
        self.suggested_override = suggested_override
        super().__init__(f"{statement_name} is unavailable for ticker '{ticker}'.")
