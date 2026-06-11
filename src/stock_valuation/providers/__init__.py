"""External valuation provider contracts and secret-safe configuration."""

from stock_valuation.providers.config import ProviderConfig
from stock_valuation.providers.errors import raise_provider_unavailable
from stock_valuation.providers.protocols import (
    EquityRiskPremiumProvider,
    FxRateProvider,
    MacroRateProvider,
    MarketDebtProvider,
    SovereignYieldProvider,
    TaxRateProvider,
)
from stock_valuation.providers.results import (
    SovereignYieldCandidate,
    SovereignYieldResult,
)

__all__ = [
    "EquityRiskPremiumProvider",
    "FxRateProvider",
    "MacroRateProvider",
    "MarketDebtProvider",
    "ProviderConfig",
    "SovereignYieldCandidate",
    "SovereignYieldProvider",
    "SovereignYieldResult",
    "TaxRateProvider",
    "raise_provider_unavailable",
]
