"""Secret-safe provider failure helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from stock_valuation.errors import ProviderUnavailableError


def raise_provider_unavailable(
    provider_name: str,
    input_name: str,
    *,
    source_attempted: str | None = None,
    fallbacks_attempted: Sequence[str] = (),
    api_key_env_var: str | None = None,
    cause: Exception | None = None,
) -> NoReturn:
    """Raise a secret-safe provider error for setup or runtime failures.

    Parameters
    ----------
    provider_name:
        Human-readable provider name.
    input_name:
        Provider input or operation that failed.
    source_attempted:
        Secret-free source or endpoint identifier.
    fallbacks_attempted:
        Fallback providers or sources already attempted.
    api_key_env_var:
        Environment variable name required for provider setup.
    cause:
        Optional underlying exception used only for exception chaining. Its
        message is never copied into the public error.

    Raises
    ------
    ProviderUnavailableError
        Always raised with secret-safe setup guidance.
    """

    guidance = "Configure the provider or supply the documented valuation override."
    if api_key_env_var:
        guidance = f"Set environment variable {api_key_env_var} or supply the documented valuation override."
    error = ProviderUnavailableError(
        provider_name,
        input_name,
        source_attempted=source_attempted,
        fallbacks_attempted=fallbacks_attempted,
        suggested_override=guidance,
    )
    if cause is None:
        raise error
    raise error from cause
