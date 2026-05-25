"""Typed exceptions for stock valuation data access."""


class StockValuationError(Exception):
    """Base exception for stock valuation failures.

    Parameters
    ----------
    message:
        Human-readable error summary.
    ticker:
        Ticker symbol associated with the failure, when available.
    metric_name:
        Metric, statement, or provider name associated with the failure.
    source:
        Data source that was attempted.
    fallbacks:
        Fallback sources that were attempted before failing.
    suggested_override:
        User override that may resolve the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        ticker: str | None = None,
        metric_name: str | None = None,
        source: str | None = None,
        fallbacks: tuple[str, ...] = (),
        suggested_override: str | None = None,
    ) -> None:
        super().__init__(message)
        self.ticker = ticker
        self.metric_name = metric_name
        self.source = source
        self.fallbacks = fallbacks
        self.suggested_override = suggested_override


class TickerNotFoundError(StockValuationError):
    """Raised when a ticker is empty, invalid, or cannot be resolved."""


class StatementUnavailableError(StockValuationError):
    """Raised when a required yfinance financial statement is unavailable.

    Parameters
    ----------
    message:
        Human-readable error summary.
    statement_name:
        Statement name associated with the failure.
    ticker:
        Ticker symbol associated with the failure, when available.
    source:
        Data source that was attempted.
    fallbacks:
        Fallback sources that were attempted before failing.
    suggested_override:
        User override that may resolve the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        statement_name: str,
        ticker: str | None = None,
        source: str | None = None,
        fallbacks: tuple[str, ...] = (),
        suggested_override: str | None = None,
    ) -> None:
        super().__init__(
            message,
            ticker=ticker,
            metric_name=statement_name,
            source=source,
            fallbacks=fallbacks,
            suggested_override=suggested_override,
        )
        self.statement_name = statement_name


class MetricUnavailableError(StockValuationError):
    """Raised when a required yfinance metric is unavailable."""
