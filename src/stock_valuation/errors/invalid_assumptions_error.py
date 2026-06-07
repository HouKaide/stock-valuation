"""Exception raised when valuation assumptions are invalid."""

from __future__ import annotations

from stock_valuation.errors.stock_valuation_error import StockValuationError


class InvalidAssumptionsError(StockValuationError):
    """Raised when valuation assumptions fail validation.

    Parameters
    ----------
    field_name:
        Name of the invalid assumption field.
    message:
        Human-readable validation failure.
    suggested_override:
        Optional user action that can resolve the failure.
    """

    def __init__(
        self,
        field_name: str,
        message: str,
        *,
        suggested_override: str | None = None,
    ) -> None:
        self.field_name = field_name
        self.suggested_override = suggested_override
        super().__init__(f"Invalid valuation assumption '{field_name}': {message}")
