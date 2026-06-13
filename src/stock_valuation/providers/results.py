"""Data contracts for external provider results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from stock_valuation.contracts import Diagnostic


@dataclass(frozen=True)
class SovereignYieldCandidate:
    """Candidate considered for deterministic sovereign-yield selection.

    Attributes
    ----------
    symbol:
        Provider instrument symbol.
    currency:
        Instrument currency.
    country:
        Sovereign issuer country when known.
    maturity_years:
        Instrument maturity in years.
    instrument_type:
        Provider instrument classification.
    confidence:
        Provider confidence score in decimal representation.
    yield_value:
        Candidate yield in decimal representation.
    rejection_reason:
        Reason the candidate was rejected, or ``None`` when selected.
    """

    symbol: str
    currency: str
    country: str | None
    maturity_years: Decimal
    instrument_type: str
    confidence: Decimal
    yield_value: Decimal
    rejection_reason: str | None = None


@dataclass(frozen=True)
class SovereignYieldResult:
    """Result of deterministic sovereign-yield candidate selection.

    Attributes
    ----------
    selected:
        Candidate selected using currency, country, maturity, instrument type,
        and provider confidence.
    rejected:
        Candidates rejected during deterministic selection.
    provider:
        Provider that performed discovery and selection.
    valuation_date:
        Date for which the yield was resolved.
    diagnostics:
        Selection and rejection diagnostics.
    """

    selected: SovereignYieldCandidate
    provider: str
    valuation_date: date
    rejected: tuple[SovereignYieldCandidate, ...] = ()
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def yield_value(self) -> Decimal:
        """Return the selected yield.

        Returns
        -------
        Decimal
            Selected sovereign yield in decimal representation.
        """

        return self.selected.yield_value
