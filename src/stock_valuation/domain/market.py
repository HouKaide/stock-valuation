"""Market-facing domain models used by valuation services."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class MarketSnapshot:
    """Snapshot of the current market inputs required for valuation."""

    symbol: str
    company_name: str
    currency: str
    current_price: Decimal
    market_cap: Decimal | None
    beta_reference: Decimal | None
    shares_outstanding: Decimal
