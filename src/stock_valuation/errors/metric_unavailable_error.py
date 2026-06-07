"""Exception raised when a required metric cannot be resolved."""

from __future__ import annotations

from collections.abc import Sequence

from stock_valuation.errors.stock_valuation_error import StockValuationError


class MetricUnavailableError(StockValuationError):
    """Raised when a required metric or lookup result is unavailable.

    Parameters
    ----------
    ticker:
        Ticker or query context associated with the missing metric.
    metric_name:
        Human-readable metric name that could not be resolved.
    source_attempted:
        Source attempted for the metric.
    fallbacks_attempted:
        Fallback sources attempted before failing.
    suggested_override:
        Optional user action that can resolve the failure.
    """

    def __init__(
        self,
        ticker: str,
        metric_name: str,
        *,
        source_attempted: str | None = None,
        fallbacks_attempted: Sequence[str] | None = None,
        suggested_override: str | None = None,
    ) -> None:
        self.ticker = ticker
        self.symbol = ticker
        self.metric_name = metric_name
        self.metric = metric_name
        self.source_attempted = source_attempted
        self.fallbacks_attempted = tuple(fallbacks_attempted or ())
        self.suggested_override = suggested_override
        super().__init__(f"{metric_name} is unavailable for ticker '{ticker}'.")
