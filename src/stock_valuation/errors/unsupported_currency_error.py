"""Exception raised when a currency cannot be normalized."""

from __future__ import annotations

from collections.abc import Sequence

from stock_valuation.errors.stock_valuation_error import StockValuationError


class UnsupportedCurrencyError(StockValuationError):
    """Raised when a currency value is missing or unsupported.

    Parameters
    ----------
    currency:
        Raw currency value that could not be normalized.
    metric_name:
        Metric requiring the currency.
    ticker:
        Optional ticker associated with the currency.
    source_attempted:
        Source attempted for the currency.
    fallbacks_attempted:
        Fallback currency sources attempted.
    suggested_override:
        Optional valuation-currency override guidance.
    """

    def __init__(
        self,
        currency: object,
        metric_name: str,
        *,
        ticker: str | None = None,
        source_attempted: str | None = None,
        fallbacks_attempted: Sequence[str] = (),
        suggested_override: str | None = (
            "Provide a supported valuation_currency_override."
        ),
    ) -> None:
        self.currency = currency
        super().__init__(
            f"Unsupported currency {currency!r} for {metric_name}.",
            ticker=ticker,
            metric=metric_name,
            source_attempted=source_attempted,
            fallbacks_attempted=fallbacks_attempted,
            suggested_override=suggested_override,
            metadata={"currency": currency},
        )
