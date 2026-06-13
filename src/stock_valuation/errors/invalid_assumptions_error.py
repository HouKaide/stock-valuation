"""Exception raised when valuation assumptions are invalid."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from stock_valuation.errors.stock_valuation_error import StockValuationError
from stock_valuation.redaction import redact_secrets


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
    wacc:
        Optional WACC associated with a terminal-growth validation failure.
    terminal_growth_rate:
        Optional terminal growth associated with a validation failure.
    source_rows:
        Optional normalized source rows used for the invalid calculation.
    """

    def __init__(
        self,
        field_name: str,
        message: str,
        *,
        suggested_override: str | None = None,
        period: Any | None = None,
        value: Any | None = None,
        wacc: Any | None = None,
        terminal_growth_rate: Any | None = None,
        source_rows: Sequence[str] = (),
    ) -> None:
        self.field_name = field_name
        self.period = period
        self.value = redact_secrets(value, field_name=field_name)
        self.wacc = redact_secrets(wacc)
        self.terminal_growth_rate = redact_secrets(terminal_growth_rate)
        self.source_rows = tuple(str(redact_secrets(row)) for row in source_rows)
        context = ""
        if period is not None:
            context += f" Period: {period}."
        if self.value is not None:
            context += f" Value: {self.value}."
        if self.wacc is not None:
            context += f" WACC: {self.wacc}."
        if self.terminal_growth_rate is not None:
            context += f" Terminal growth rate: {self.terminal_growth_rate}."
        super().__init__(
            f"Invalid valuation assumption '{field_name}': {message}{context}",
            metric=field_name,
            suggested_override=suggested_override,
            metadata={
                "period": period,
                "value": self.value,
                "wacc": self.wacc,
                "terminal_growth_rate": self.terminal_growth_rate,
                "source_rows": self.source_rows,
            },
        )
