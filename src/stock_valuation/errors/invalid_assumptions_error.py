"""Exception raised when valuation assumptions are invalid."""

from __future__ import annotations

from typing import Any

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
    period:
        Optional period associated with the invalid value.
    value:
        Optional invalid value.
    """

    def __init__(
        self,
        field_name: str,
        message: str,
        *,
        suggested_override: str | None = None,
        period: Any | None = None,
        value: Any | None = None,
    ) -> None:
        self.field_name = field_name
        self.metric_name = field_name
        self.suggested_override = suggested_override
        self.period = period
        self.value = value
        context = ""
        if period is not None:
            context += f" Period: {period}."
        if value is not None:
            context += f" Value: {value}."
        super().__init__(
            f"Invalid valuation assumption '{field_name}': {message}{context}"
        )
