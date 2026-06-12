"""Exception raised when a raw value cannot be normalized."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from stock_valuation.errors.metric_unavailable_error import MetricUnavailableError
from stock_valuation.errors.stock_valuation_error import StockValuationError
from stock_valuation.redaction import redact_secrets


class NormalizationError(MetricUnavailableError):
    """Raised when source data cannot satisfy a normalized convention.

    Parameters
    ----------
    metric_name:
        Metric being normalized.
    source_attempted:
        Source or conversion attempted.
    raw_value:
        Optional raw value summarized after secret redaction.
    ticker:
        Optional ticker associated with the raw value.
    fallbacks_attempted:
        Fallback normalization paths attempted.
    suggested_override:
        Optional user action that can resolve the failure.
    """

    def __init__(
        self,
        metric_name: str,
        *,
        source_attempted: str,
        raw_value: Any | None = None,
        ticker: str | None = None,
        fallbacks_attempted: Sequence[str] = (),
        suggested_override: str | None = None,
    ) -> None:
        self.raw_value = redact_secrets(raw_value, field_name=metric_name)
        StockValuationError.__init__(
            self,
            f"Could not normalize {metric_name}.",
            ticker=ticker,
            metric=metric_name,
            source_attempted=source_attempted,
            fallbacks_attempted=fallbacks_attempted,
            suggested_override=suggested_override,
            metadata={"raw_value": self.raw_value},
        )
