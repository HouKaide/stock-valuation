"""Secret-safe configuration references for external providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    """Reference provider configuration without storing secret values.

    Parameters
    ----------
    name:
        Provider name.
    base_url:
        Optional provider base URL.
    api_key_env_var:
        Optional environment variable name containing the API key.

    Notes
    -----
    This contract stores only an environment variable name. It never reads,
    stores, or renders the corresponding secret value.
    """

    name: str
    base_url: str | None = None
    api_key_env_var: str | None = None
