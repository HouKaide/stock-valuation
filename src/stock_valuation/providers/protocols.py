"""Protocols for external valuation data providers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol, runtime_checkable

from stock_valuation.providers.results import SovereignYieldResult


@runtime_checkable
class TaxRateProvider(Protocol):
    """Provide country-level marginal corporate tax rates.

    Provider implementations return rates as decimals, where ``0.21`` means
    21 percent. Setup and runtime failures must raise
    ``ProviderUnavailableError``.
    """

    def get_corporate_tax_rate(self, country: str, valuation_date: date) -> Decimal:
        """Return the marginal corporate tax rate.

        Parameters
        ----------
        country:
            Company headquarters country.
        valuation_date:
            Date for which the rate is required.

        Returns
        -------
        Decimal
            Corporate tax rate in decimal representation.
        """


@runtime_checkable
class EquityRiskPremiumProvider(Protocol):
    """Provide equity risk premiums for a company risk country."""

    def get_equity_risk_premium(self, country: str, valuation_date: date) -> Decimal:
        """Return the country equity risk premium.

        Parameters
        ----------
        country:
            Company risk country.
        valuation_date:
            Date for which the premium is required.

        Returns
        -------
        Decimal
            Equity risk premium in decimal representation.
        """


@runtime_checkable
class MacroRateProvider(Protocol):
    """Provide long-term government yields used as risk-free rates.

    Consumers use the US 10-year Treasury method as a documented fallback
    when a currency-matched long-term government yield is unavailable.
    """

    def get_long_term_government_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> Decimal:
        """Return a currency-matched long-term government yield.

        Parameters
        ----------
        currency:
            Valuation currency.
        country:
            Optional country used to refine yield selection.
        valuation_date:
            Date for which the yield is required.

        Returns
        -------
        Decimal
            Government yield in decimal representation.
        """

    def get_us_10y_treasury_yield(self, valuation_date: date) -> Decimal:
        """Return the US 10-year Treasury fallback yield.

        Parameters
        ----------
        valuation_date:
            Date for which the yield is required.

        Returns
        -------
        Decimal
            US 10-year Treasury yield in decimal representation.
        """


@runtime_checkable
class SovereignYieldProvider(Protocol):
    """Discover a deterministic 10-year sovereign-yield instrument.

    Implementations rank candidates by currency, country, ten-year maturity,
    instrument type, and provider confidence. Missing usable candidates must
    raise ``ProviderUnavailableError`` with terminal-growth override guidance.
    """

    def find_10y_sovereign_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> SovereignYieldResult:
        """Return the selected 10-year sovereign-yield result.

        Parameters
        ----------
        currency:
            Valuation currency.
        country:
            Optional country used to refine instrument selection.
        valuation_date:
            Date for which the yield is required.

        Returns
        -------
        SovereignYieldResult
            Selected candidate, rejected candidates, provider, and yield.
        """


@runtime_checkable
class FxRateProvider(Protocol):
    """Provide exact decimal foreign-exchange rates and conversions.

    Identity conversion returns the input amount, and an identity rate is
    ``Decimal("1")``.
    """

    def convert(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        valuation_date: date,
    ) -> Decimal:
        """Convert an amount between currencies.

        Parameters
        ----------
        amount:
            Decimal amount to convert.
        from_currency:
            Source ISO currency.
        to_currency:
            Target ISO currency.
        valuation_date:
            Date for which the rate is required.

        Returns
        -------
        Decimal
            Converted decimal amount.
        """

    def get_rate(
        self,
        from_currency: str,
        to_currency: str,
        valuation_date: date,
    ) -> Decimal:
        """Return the exchange rate between two currencies.

        Parameters
        ----------
        from_currency:
            Source ISO currency.
        to_currency:
            Target ISO currency.
        valuation_date:
            Date for which the rate is required.

        Returns
        -------
        Decimal
            Exchange rate in decimal representation.
        """


@runtime_checkable
class MarketDebtProvider(Protocol):
    """Provide market debt when available.

    ``None`` means no market value is available. Consumers fall back to book
    debt outside this provider contract.
    """

    def get_market_value_of_debt(
        self, ticker: str, valuation_date: date
    ) -> Decimal | None:
        """Return market value of debt when available.

        Parameters
        ----------
        ticker:
            Company ticker.
        valuation_date:
            Date for which market debt is required.

        Returns
        -------
        Decimal | None
            Market value of debt, or ``None`` when unavailable.
        """


__all__ = [
    "EquityRiskPremiumProvider",
    "FxRateProvider",
    "MacroRateProvider",
    "MarketDebtProvider",
    "SovereignYieldProvider",
    "TaxRateProvider",
]
