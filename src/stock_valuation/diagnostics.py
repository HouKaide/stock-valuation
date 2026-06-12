"""Diagnostic factories, error conversion, and JSON-safe rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from stock_valuation.contracts import (
    Diagnostic,
    DiagnosticCategory,
    to_json_safe,
)
from stock_valuation.errors import StockValuationError
from stock_valuation.redaction import redact_secrets


def source_diagnostic(message: str, **context: Any) -> Diagnostic:
    """Create a source-selection diagnostic.

    Parameters
    ----------
    message:
        Concise source-selection message.
    **context:
        Additional ``Diagnostic`` fields.

    Returns
    -------
    Diagnostic
        Secret-safe source diagnostic.
    """

    return _diagnostic(DiagnosticCategory.SOURCE, message, context)


def fallback_diagnostic(message: str, **context: Any) -> Diagnostic:
    """Create a fallback-selection diagnostic.

    Parameters
    ----------
    message:
        Concise fallback message.
    **context:
        Additional ``Diagnostic`` fields.

    Returns
    -------
    Diagnostic
        Secret-safe fallback diagnostic.
    """

    return _diagnostic(DiagnosticCategory.FALLBACK, message, context)


def override_diagnostic(message: str, **context: Any) -> Diagnostic:
    """Create an explicit-override diagnostic.

    Parameters
    ----------
    message:
        Concise override message.
    **context:
        Additional ``Diagnostic`` fields.

    Returns
    -------
    Diagnostic
        Secret-safe override diagnostic.
    """

    return _diagnostic(DiagnosticCategory.OVERRIDE, message, context)


def provider_diagnostic(message: str, **context: Any) -> Diagnostic:
    """Create a provider-result diagnostic.

    Parameters
    ----------
    message:
        Concise provider message.
    **context:
        Additional ``Diagnostic`` fields.

    Returns
    -------
    Diagnostic
        Secret-safe provider diagnostic.
    """

    return _diagnostic(DiagnosticCategory.PROVIDER, message, context)


def failure_diagnostic(message: str, **context: Any) -> Diagnostic:
    """Create a failure diagnostic.

    Parameters
    ----------
    message:
        Concise failure message.
    **context:
        Additional ``Diagnostic`` fields.

    Returns
    -------
    Diagnostic
        Secret-safe failure diagnostic.
    """

    return _diagnostic(DiagnosticCategory.FAILURE, message, context)


def error_to_diagnostic(error: StockValuationError) -> Diagnostic:
    """Convert a typed project error into a failure diagnostic.

    Parameters
    ----------
    error:
        Typed project error to convert.

    Returns
    -------
    Diagnostic
        Failure diagnostic without traceback or secret content.
    """

    return failure_diagnostic(
        error.safe_message,
        ticker=error.ticker,
        metric=error.metric,
        provider=error.provider,
        source_attempted=error.source_attempted,
        fallbacks_attempted=error.fallbacks_attempted,
        suggested_override=error.suggested_override,
        metadata={
            "error_type": type(error).__name__,
            **dict(error.metadata),
        },
    )


def error_to_dict(error: StockValuationError) -> dict[str, Any]:
    """Serialize a typed error into a stable JSON-safe dictionary.

    Parameters
    ----------
    error:
        Typed project error to serialize.

    Returns
    -------
    dict[str, Any]
        Stable secret-safe error payload.
    """

    return {
        "type": type(error).__name__,
        "message": error.safe_message,
        "ticker": error.ticker,
        "metric": error.metric,
        "provider": error.provider,
        "source_attempted": error.source_attempted,
        "fallbacks_attempted": list(error.fallbacks_attempted),
        "suggested_override": error.suggested_override,
        "metadata": to_json_safe(redact_secrets(error.metadata)),
    }


def diagnostic_to_dict(diagnostic: Diagnostic) -> dict[str, Any]:
    """Serialize a diagnostic into a stable JSON-safe dictionary.

    Parameters
    ----------
    diagnostic:
        Diagnostic to serialize.

    Returns
    -------
    dict[str, Any]
        Stable secret-safe diagnostic payload.
    """

    return {
        "category": str(diagnostic.kind),
        "message": diagnostic.message,
        "ticker": diagnostic.ticker,
        "metric": diagnostic.metric,
        "provider": diagnostic.provider,
        "source_attempted": diagnostic.source_attempted,
        "fallbacks_attempted": list(diagnostic.fallbacks_attempted),
        "selected_fallback": diagnostic.selected_fallback,
        "suggested_override": diagnostic.suggested_override,
        "metadata": to_json_safe(redact_secrets(diagnostic.metadata)),
    }


def _diagnostic(
    category: DiagnosticCategory,
    message: str,
    context: Mapping[str, Any],
) -> Diagnostic:
    allowed_fields = {
        "ticker",
        "metric",
        "provider",
        "source_attempted",
        "fallbacks_attempted",
        "suggested_override",
        "selected_fallback",
        "metadata",
    }
    unexpected = set(context) - allowed_fields
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"Unexpected diagnostic fields: {names}")
    values = dict(context)
    fallbacks = values.get("fallbacks_attempted", ())
    if isinstance(fallbacks, Sequence) and not isinstance(fallbacks, str):
        values["fallbacks_attempted"] = tuple(fallbacks)
    return Diagnostic(kind=category, message=message, **values)


__all__ = [
    "Diagnostic",
    "DiagnosticCategory",
    "diagnostic_to_dict",
    "error_to_diagnostic",
    "error_to_dict",
    "failure_diagnostic",
    "fallback_diagnostic",
    "override_diagnostic",
    "provider_diagnostic",
    "redact_secrets",
    "source_diagnostic",
]
