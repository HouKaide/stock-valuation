"""Exception raised when an external provider is unavailable."""

from __future__ import annotations

from collections.abc import Sequence

from stock_valuation.errors.stock_valuation_error import StockValuationError


class ProviderUnavailableError(StockValuationError):
    """Raised when provider setup or runtime data access fails.

    Parameters
    ----------
    provider_name:
        Human-readable provider name.
    input_name:
        Provider input or operation that failed.
    source_attempted:
        Provider source or endpoint attempted.
    fallbacks_attempted:
        Fallback providers or sources attempted.
    suggested_override:
        Secret-safe setup or override guidance.
    """

    def __init__(
        self,
        provider_name: str,
        input_name: str,
        *,
        source_attempted: str | None = None,
        fallbacks_attempted: Sequence[str] | None = None,
        suggested_override: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.provider = provider_name
        self.input_name = input_name
        self.metric_name = input_name
        self.source_attempted = source_attempted
        self.fallbacks_attempted = tuple(fallbacks_attempted or ())
        self.suggested_override = suggested_override
        super().__init__(f"Provider '{provider_name}' is unavailable for {input_name}.")
