"""Tests for diagnostic categories, factories, redaction, and serialization."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from stock_valuation import (
    Diagnostic,
    DiagnosticCategory,
    diagnostic_to_dict,
    error_to_diagnostic,
    error_to_dict,
    failure_diagnostic,
    fallback_diagnostic,
    override_diagnostic,
    provider_diagnostic,
    redact_secrets,
    source_diagnostic,
)
from stock_valuation.errors import (
    InvalidAssumptionsError,
    MarketDataUnavailableError,
    MetricUnavailableError,
    NormalizationError,
    ProviderUnavailableError,
    StatementUnavailableError,
    TickerNotFoundError,
    UnsupportedCurrencyError,
)


def test_diagnostic_category_values_are_stable() -> None:
    """Diagnostic category values should remain stable for JSON consumers."""

    assert {category.value for category in DiagnosticCategory} == {
        "source",
        "fallback",
        "override",
        "provider",
        "normalization",
        "warning",
        "failure",
        "calculation",
    }


@pytest.mark.parametrize("category", list(DiagnosticCategory))
def test_diagnostic_accepts_every_category(category: DiagnosticCategory) -> None:
    """Diagnostics should construct for every supported category."""

    diagnostic = Diagnostic(kind=category, message="Safe message.")

    assert diagnostic.kind == category


def test_diagnostic_preserves_selected_fallback_and_safe_metadata() -> None:
    """Diagnostics should retain useful metadata while redacting credentials."""

    diagnostic = Diagnostic(
        kind="fallback",
        message="Primary source failed.",
        ticker="TEST",
        metric="beta",
        source_attempted="info.beta",
        fallbacks_attempted=("override",),
        selected_fallback="override",
        suggested_override="Set token=secret-value",
        metadata={"api_key": "secret-value", "sample_size": 3},
    )

    assert diagnostic.kind == DiagnosticCategory.FALLBACK
    assert diagnostic.selected_fallback == "override"
    assert diagnostic.metadata == {"api_key": "[REDACTED]", "sample_size": 3}
    assert "secret-value" not in diagnostic.suggested_override


@pytest.mark.parametrize(
    ("factory", "category"),
    [
        (source_diagnostic, DiagnosticCategory.SOURCE),
        (fallback_diagnostic, DiagnosticCategory.FALLBACK),
        (override_diagnostic, DiagnosticCategory.OVERRIDE),
        (provider_diagnostic, DiagnosticCategory.PROVIDER),
        (failure_diagnostic, DiagnosticCategory.FAILURE),
    ],
)
def test_diagnostic_factories_create_expected_category(
    factory: object,
    category: DiagnosticCategory,
) -> None:
    """Each explicit factory should create its documented category."""

    diagnostic = factory(  # type: ignore[operator]
        "Resolved input.",
        ticker="TEST",
        metric="beta",
    )

    assert diagnostic.kind == category
    assert diagnostic.ticker == "TEST"


def test_diagnostic_factory_rejects_unknown_fields() -> None:
    """Factories should fail clearly on misspelled context fields."""

    with pytest.raises(TypeError, match="unexpected"):
        source_diagnostic("Resolved.", unexpected="value")


def test_redaction_handles_names_values_urls_and_safe_metadata() -> None:
    """Redaction should remove credentials without discarding safe values."""

    value = {
        "FRED_API_KEY": "fred-secret",
        "accessToken": "token-secret",
        "password": "password-secret",
        "endpoint": "https://user:password@example.com/rates",
        "message": "token=inline-secret",
        "status": 503,
    }

    redacted = redact_secrets(value)
    rendered = json.dumps(redacted)

    assert "fred-secret" not in rendered
    assert "token-secret" not in rendered
    assert "password-secret" not in rendered
    assert "inline-secret" not in rendered
    assert "user:password" not in rendered
    assert redacted["status"] == 503


def _typed_errors() -> list[Exception]:
    return [
        TickerNotFoundError("TEST", source_attempted="ticker input"),
        StatementUnavailableError("TEST", "income statement"),
        MetricUnavailableError("TEST", "beta"),
        ProviderUnavailableError("macro", "yield"),
        InvalidAssumptionsError("wacc", "Invalid.", value=Decimal("-1")),
        UnsupportedCurrencyError("XYZ", "valuation currency"),
        NormalizationError(
            "price",
            source_attempted="numeric scalar conversion",
            raw_value="invalid",
        ),
        MarketDataUnavailableError("TEST", "history"),
    ]


@pytest.mark.parametrize("error", _typed_errors())
def test_each_typed_error_converts_to_failure_diagnostic(error: Exception) -> None:
    """Every public typed error should convert without traceback content."""

    diagnostic = error_to_diagnostic(error)  # type: ignore[arg-type]

    assert diagnostic.kind == DiagnosticCategory.FAILURE
    assert diagnostic.metadata["error_type"] == type(error).__name__
    assert "Traceback" not in diagnostic.message


def test_error_and_diagnostic_payloads_are_json_safe_and_stable() -> None:
    """CLI payloads should serialize exact scalars, dates, and safe metadata."""

    error = InvalidAssumptionsError(
        "terminal_growth_rate",
        "Growth must be below WACC.",
        value=Decimal("0.10"),
        wacc=Decimal("0.08"),
        suggested_override="Provide terminal_growth_rate_override.",
    )
    diagnostic = Diagnostic(
        kind=DiagnosticCategory.WARNING,
        message="Comparison date is stale.",
        metadata={
            "as_of": date(2026, 1, 31),
            "difference": Decimal("0.02"),
            "api_key": "secret-value",
        },
    )

    error_payload = error_to_dict(error)
    diagnostic_payload = diagnostic_to_dict(diagnostic)

    assert error_payload["type"] == "InvalidAssumptionsError"
    assert error_payload["metadata"]["value"] == "0.10"
    assert diagnostic_payload["category"] == "warning"
    assert diagnostic_payload["metadata"] == {
        "as_of": "2026-01-31",
        "difference": "0.02",
        "api_key": "[REDACTED]",
    }
    assert "secret-value" not in json.dumps(diagnostic_payload)
    json.dumps(error_payload)
    json.dumps(diagnostic_payload)


def test_public_diagnostic_imports_are_available() -> None:
    """Root exports should expose diagnostic contracts and helpers."""

    assert source_diagnostic("Resolved.").kind == "source"
    assert diagnostic_to_dict(Diagnostic("source", "Resolved."))["category"] == (
        "source"
    )
