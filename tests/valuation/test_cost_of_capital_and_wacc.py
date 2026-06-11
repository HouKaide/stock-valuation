"""Tests for cost-of-capital and WACC calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from stock_valuation import DamodaranValuationProcessor
from stock_valuation.contracts import ValuationAssumptions
from stock_valuation.errors import (
    InvalidAssumptionsError,
    MetricUnavailableError,
    ProviderUnavailableError,
)

VALUATION_DATE = date(2026, 1, 31)


@dataclass
class FakeStock:
    """Expose normalized cost-of-capital inputs."""

    beta_value: Decimal | None = Decimal("1.2")
    market_cap_value: Decimal | None = Decimal("800")
    debt: pd.Series = field(
        default_factory=lambda: pd.Series(
            [Decimal("100"), Decimal("200")],
            index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
            dtype=object,
        )
    )
    interest: pd.Series = field(
        default_factory=lambda: pd.Series(
            [Decimal("10"), Decimal("15")],
            index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
            dtype=object,
        )
    )
    tax_rate_provider: FakeTaxRateProvider | None = None

    def normalized_ticker(self) -> str:
        """Return a deterministic ticker."""

        return "TEST"

    def valuation_currency(self) -> str:
        """Return the valuation currency."""

        return "USD"

    def headquarters_country(self) -> str:
        """Return the company country."""

        return "United States"

    def beta(self) -> Decimal | None:
        """Return normalized beta."""

        return self.beta_value

    def market_cap(self) -> Decimal | None:
        """Return normalized market capitalization."""

        return self.market_cap_value

    def debt_series(self) -> pd.Series:
        """Return normalized annual debt."""

        return self.debt

    def interest_expense_series(self) -> pd.Series:
        """Return normalized annual interest expense."""

        return self.interest


@dataclass
class FakeMacroRateProvider:
    """Return deterministic macro rates or typed failures."""

    long_term_yield: Decimal = Decimal("0.04")
    us_10y_yield: Decimal = Decimal("0.045")
    fail_primary: bool = False
    fail_fallback: bool = False

    def get_long_term_government_yield(
        self,
        currency: str,
        country: str | None,
        valuation_date: date,
    ) -> Decimal:
        """Return the currency-matched yield."""

        if self.fail_primary:
            raise ProviderUnavailableError("macro", "currency yield")
        return self.long_term_yield

    def get_us_10y_treasury_yield(self, valuation_date: date) -> Decimal:
        """Return the US 10-year fallback yield."""

        if self.fail_fallback:
            raise ProviderUnavailableError("macro", "US 10Y yield")
        return self.us_10y_yield


@dataclass
class FakeEquityRiskPremiumProvider:
    """Return a deterministic ERP or typed failure."""

    premium: Decimal = Decimal("0.05")
    fail: bool = False

    def get_equity_risk_premium(
        self,
        country: str,
        valuation_date: date,
    ) -> Decimal:
        """Return the country ERP."""

        if self.fail:
            raise ProviderUnavailableError("erp", "equity risk premium")
        return self.premium


@dataclass
class FakeTaxRateProvider:
    """Return a deterministic corporate tax rate or typed failure."""

    rate: Decimal = Decimal("0.20")
    fail: bool = False

    def get_corporate_tax_rate(
        self,
        country: str,
        valuation_date: date,
    ) -> Decimal:
        """Return the corporate tax rate."""

        if self.fail:
            raise ProviderUnavailableError("tax", "corporate tax rate")
        return self.rate


@dataclass
class FakeMarketDebtProvider:
    """Return deterministic market debt."""

    debt: Decimal | None = Decimal("180")

    def get_market_value_of_debt(
        self,
        ticker: str,
        valuation_date: date,
    ) -> Decimal | None:
        """Return market debt or no value."""

        return self.debt


def _assumptions(**overrides: Decimal | str | None) -> ValuationAssumptions:
    values: dict[str, object] = {"valuation_date": VALUATION_DATE}
    values.update(overrides)
    return ValuationAssumptions(**values)  # type: ignore[arg-type]


def _processor(
    *,
    stock: FakeStock | None = None,
    assumptions: ValuationAssumptions | None = None,
    macro_provider: FakeMacroRateProvider | None = None,
    erp_provider: FakeEquityRiskPremiumProvider | None = None,
    tax_provider: FakeTaxRateProvider | None = None,
    debt_provider: FakeMarketDebtProvider | None = None,
) -> DamodaranValuationProcessor:
    return DamodaranValuationProcessor(
        stock=stock or FakeStock(),  # type: ignore[arg-type]
        assumptions=assumptions or _assumptions(),
        macro_provider=macro_provider,
        erp_provider=erp_provider,
        tax_rate_provider=tax_provider,
        market_debt_provider=debt_provider,
    )


def test_risk_free_rate_prefers_override() -> None:
    """Risk-free override should avoid the provider."""

    processor = _processor(
        assumptions=_assumptions(risk_free_rate_override=Decimal("0.03")),
        macro_provider=FakeMacroRateProvider(fail_primary=True, fail_fallback=True),
    )

    assert processor.risk_free_rate() == Decimal("0.03")
    assert processor.diagnostics[-1].kind == "override"


def test_risk_free_rate_uses_currency_matched_provider_value() -> None:
    """Primary macro yield should be used when available."""

    processor = _processor(macro_provider=FakeMacroRateProvider())

    assert processor.risk_free_rate() == Decimal("0.04")
    assert processor.diagnostics[-1].kind == "provider"


def test_risk_free_rate_falls_back_to_us_10y() -> None:
    """US 10-year yield should be used after primary failure."""

    processor = _processor(
        macro_provider=FakeMacroRateProvider(fail_primary=True),
    )

    assert processor.risk_free_rate() == Decimal("0.045")
    assert processor.diagnostics[-1].kind == "fallback"
    assert "US 10-year" in processor.diagnostics[-1].message


def test_risk_free_rate_raises_when_both_provider_paths_fail() -> None:
    """Exhausted risk-free provider paths should raise a typed error."""

    processor = _processor(
        macro_provider=FakeMacroRateProvider(
            fail_primary=True,
            fail_fallback=True,
        )
    )

    with pytest.raises(ProviderUnavailableError) as captured:
        processor.risk_free_rate()

    assert captured.value.fallbacks_attempted == ("US 10-year Treasury yield",)


def test_erp_resolution_supports_override_provider_and_failure() -> None:
    """ERP should resolve by override, provider, or typed failure."""

    override_processor = _processor(
        assumptions=_assumptions(
            equity_risk_premium_override=Decimal("0.06"),
        ),
        erp_provider=FakeEquityRiskPremiumProvider(fail=True),
    )
    provider_processor = _processor(
        erp_provider=FakeEquityRiskPremiumProvider(),
    )
    failure_processor = _processor(
        erp_provider=FakeEquityRiskPremiumProvider(fail=True),
    )

    assert override_processor.equity_risk_premium() == Decimal("0.06")
    assert override_processor.diagnostics[-1].kind == "override"
    assert provider_processor.equity_risk_premium() == Decimal("0.05")
    assert provider_processor.diagnostics[-1].kind == "provider"
    with pytest.raises(ProviderUnavailableError):
        failure_processor.equity_risk_premium()


def test_beta_resolution_supports_override_stock_and_missing_value() -> None:
    """Beta should resolve from override or stock and fail when absent."""

    override_processor = _processor(
        assumptions=_assumptions(beta_override=Decimal("1.1")),
        stock=FakeStock(beta_value=None),
    )
    stock_processor = _processor(stock=FakeStock(beta_value=Decimal("1.3")))
    missing_processor = _processor(stock=FakeStock(beta_value=None))

    assert override_processor.beta() == Decimal("1.1")
    assert override_processor.diagnostics[-1].kind == "override"
    assert stock_processor.beta() == Decimal("1.3")
    assert stock_processor.diagnostics[-1].source_attempted == "Stock.beta"
    with pytest.raises(MetricUnavailableError):
        missing_processor.beta()


def test_cost_of_equity_returns_formula_sources_and_steps() -> None:
    """Cost of equity should expose its exact component calculation."""

    processor = _processor(
        macro_provider=FakeMacroRateProvider(),
        erp_provider=FakeEquityRiskPremiumProvider(),
    )

    result = processor.cost_of_equity()

    assert result.risk_free_rate == Decimal("0.04")
    assert result.beta == Decimal("1.2")
    assert result.equity_risk_premium == Decimal("0.05")
    assert result.cost_of_equity == Decimal("0.10")
    assert len(result.source_details) == 3
    assert result.calculation_steps


@pytest.mark.parametrize("market_cap", [None, Decimal("0"), Decimal("-1")])
def test_market_value_of_equity_requires_positive_market_cap(
    market_cap: Decimal | None,
) -> None:
    """Missing and non-positive market caps should fail."""

    processor = _processor(stock=FakeStock(market_cap_value=market_cap))

    with pytest.raises(MetricUnavailableError):
        processor.market_value_of_equity()


def test_market_value_of_debt_prefers_override() -> None:
    """Market-debt override should avoid provider and book values."""

    processor = _processor(
        assumptions=_assumptions(market_debt_override=Decimal("220")),
        debt_provider=FakeMarketDebtProvider(debt=Decimal("180")),
    )

    assert processor.market_value_of_debt() == Decimal("220")
    assert processor.diagnostics[-1].kind == "override"


def test_market_value_of_debt_uses_provider_value() -> None:
    """Provider market debt should be used when available."""

    processor = _processor(debt_provider=FakeMarketDebtProvider())

    assert processor.market_value_of_debt() == Decimal("180")
    assert processor.diagnostics[-1].kind == "provider"


@pytest.mark.parametrize(
    "provider",
    [FakeMarketDebtProvider(debt=None), None],
)
def test_market_value_of_debt_falls_back_to_latest_book_debt(
    provider: FakeMarketDebtProvider | None,
) -> None:
    """Missing provider debt should use latest normalized book debt."""

    processor = _processor(debt_provider=provider)

    assert processor.market_value_of_debt() == Decimal("200")
    assert processor.diagnostics[-1].kind == "fallback"
    assert "book debt" in processor.diagnostics[-1].message


def test_market_value_of_debt_fails_without_book_debt() -> None:
    """Missing provider and book debt should raise a typed error."""

    processor = _processor(
        stock=FakeStock(debt=pd.Series([], dtype=object)),
    )

    with pytest.raises(MetricUnavailableError):
        processor.market_value_of_debt()


def test_average_debt_uses_latest_adjacent_periods() -> None:
    """Average debt should use the latest two chronological values."""

    processor = _processor()

    assert processor.average_debt() == Decimal("150")
    assert "2024-12-31" in processor.diagnostics[-1].message
    assert "2025-12-31" in processor.diagnostics[-1].message


@pytest.mark.parametrize(
    "debt",
    [
        pd.Series([Decimal("100")], index=[pd.Timestamp("2025-12-31")]),
        pd.Series(
            [Decimal("100"), Decimal("-100")],
            index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
        ),
        pd.Series(
            [Decimal("-100"), Decimal("-200")],
            index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
        ),
    ],
)
def test_average_debt_rejects_insufficient_or_non_positive_values(
    debt: pd.Series,
) -> None:
    """Invalid adjacent debt values should fail."""

    with pytest.raises(MetricUnavailableError):
        _processor(stock=FakeStock(debt=debt)).average_debt()


def test_tax_rate_resolution_supports_override_provider_and_failure() -> None:
    """Tax rate should resolve by override, provider, or typed failure."""

    override_processor = _processor(
        assumptions=_assumptions(tax_rate_override=Decimal("0.25")),
        tax_provider=FakeTaxRateProvider(fail=True),
    )
    provider_processor = _processor(tax_provider=FakeTaxRateProvider())
    failure_processor = _processor(tax_provider=FakeTaxRateProvider(fail=True))

    assert override_processor.tax_rate() == Decimal("0.25")
    assert override_processor.diagnostics[-1].kind == "override"
    assert provider_processor.tax_rate() == Decimal("0.20")
    assert provider_processor.diagnostics[-1].kind == "provider"
    with pytest.raises(ProviderUnavailableError):
        failure_processor.tax_rate()


def test_cost_of_debt_returns_pretax_after_tax_periods_and_steps() -> None:
    """Cost of debt should expose formula inputs and adjacent periods."""

    processor = _processor(
        assumptions=_assumptions(tax_rate_override=Decimal("0.20")),
    )

    result = processor.cost_of_debt()

    assert result.interest_expense == Decimal("15")
    assert result.average_debt == Decimal("150")
    assert result.pretax_cost_of_debt == Decimal("0.1")
    assert result.tax_rate == Decimal("0.20")
    assert result.after_tax_cost_of_debt == Decimal("0.08")
    assert result.source_periods == tuple(pd.to_datetime(["2024-12-31", "2025-12-31"]))
    assert len(result.calculation_steps) == 3


@pytest.mark.parametrize(
    ("equity", "debt"),
    [
        (Decimal("0"), Decimal("0")),
        (Decimal("-200"), Decimal("100")),
    ],
)
def test_capital_weights_reject_non_positive_total(
    equity: Decimal,
    debt: Decimal,
) -> None:
    """Zero and negative total capital should fail."""

    with pytest.raises(InvalidAssumptionsError):
        _processor().capital_weights(equity, debt)


def test_capital_weights_preserve_exact_decimal_values() -> None:
    """Capital weights should sum exactly to one."""

    total, equity_weight, debt_weight = _processor().capital_weights(
        Decimal("800"),
        Decimal("200"),
    )

    assert total == Decimal("1000")
    assert equity_weight == Decimal("0.8")
    assert debt_weight == Decimal("0.2")
    assert equity_weight + debt_weight == Decimal("1")


def test_wacc_returns_complete_breakdown_and_diagnostics() -> None:
    """WACC should combine exact component rates and capital weights."""

    processor = _processor(
        assumptions=_assumptions(
            risk_free_rate_override=Decimal("0.04"),
            equity_risk_premium_override=Decimal("0.05"),
            beta_override=Decimal("1.2"),
            tax_rate_override=Decimal("0.20"),
            market_debt_override=Decimal("200"),
        )
    )

    result = processor.wacc()

    assert result.market_value_of_equity == Decimal("800")
    assert result.market_value_of_debt == Decimal("200")
    assert result.total_capital == Decimal("1000")
    assert result.equity_weight == Decimal("0.8")
    assert result.debt_weight == Decimal("0.2")
    assert result.cost_of_equity == Decimal("0.10")
    assert result.pretax_cost_of_debt == Decimal("0.1")
    assert result.after_tax_cost_of_debt == Decimal("0.08")
    assert result.tax_rate == Decimal("0.20")
    assert result.wacc == Decimal("0.096")
    assert len(result.calculation_steps) == 4
    assert {diagnostic.kind for diagnostic in result.diagnostics} >= {
        "override",
        "source",
        "calculation",
    }


def test_wacc_diagnostics_include_provider_and_book_debt_fallback() -> None:
    """WACC diagnostics should retain major provider and fallback paths."""

    processor = _processor(
        macro_provider=FakeMacroRateProvider(),
        erp_provider=FakeEquityRiskPremiumProvider(),
        tax_provider=FakeTaxRateProvider(),
        debt_provider=FakeMarketDebtProvider(debt=None),
    )

    result = processor.wacc()

    assert any(
        diagnostic.kind == "provider" and diagnostic.metric == "risk-free rate"
        for diagnostic in result.diagnostics
    )
    assert any(
        diagnostic.kind == "provider" and diagnostic.metric == "equity risk premium"
        for diagnostic in result.diagnostics
    )
    assert any(
        diagnostic.kind == "fallback" and diagnostic.metric == "market value of debt"
        for diagnostic in result.diagnostics
    )
