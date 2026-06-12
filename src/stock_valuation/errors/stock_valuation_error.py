"""Base application exception type."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from stock_valuation.redaction import redact_secrets


class StockValuationError(Exception):
    """Base exception carrying secret-safe valuation failure context.

    Parameters
    ----------
    message:
        Concise public failure message.
    ticker:
        Optional ticker associated with the failure.
    metric:
        Optional metric associated with the failure.
    provider:
        Optional provider associated with the failure.
    source_attempted:
        Optional source attempted before failure.
    fallbacks_attempted:
        Fallback sources attempted before failure.
    suggested_override:
        Optional user action that can resolve the failure.
    metadata:
        Optional secret-safe structured context.
    """

    def __init__(
        self,
        message: str,
        *,
        ticker: str | None = None,
        metric: str | None = None,
        provider: str | None = None,
        source_attempted: str | None = None,
        fallbacks_attempted: Sequence[str] = (),
        suggested_override: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.safe_message = str(redact_secrets(message))
        self.ticker = ticker
        self.metric = metric
        self.metric_name = metric
        self.provider = provider
        self.source_attempted = (
            str(redact_secrets(source_attempted))
            if source_attempted is not None
            else None
        )
        self.fallbacks_attempted = tuple(
            str(redact_secrets(item)) for item in fallbacks_attempted
        )
        self.suggested_override = (
            str(redact_secrets(suggested_override))
            if suggested_override is not None
            else None
        )
        self.metadata = redact_secrets(dict(metadata or {}))
        super().__init__(self.safe_message)
