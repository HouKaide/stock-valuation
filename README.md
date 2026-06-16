# Stock Valuation App

A Python CLI for Damodaran-style intrinsic stock valuation. The project values
public companies using an FCFF discounted cash flow model, yfinance company
data, explicit user assumptions, and optional external provider contracts for
macro, tax, FX, debt, and sovereign-yield inputs.

## Requirements

- Python 3.12 or newer
- `uv` for the documented setup commands
- Internet access for live yfinance market and financial statement data

The package dependencies are defined in `pyproject.toml`:

- `pandas`
- `yfinance`
- `pytest` when installing the `dev` extra

## Install

From the repository root:

```bash
uv sync --extra dev
```

This creates or updates the project virtual environment and installs the package
in editable form. If you already have another virtual environment activated,
`uv` may warn that it is using the project `.venv`; that is expected.

You can also install with standard Python packaging tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run The CLI

The installed command is:

```bash
stock-valuation
```

When using `uv`, you can run it without manually activating the environment:

```bash
uv run stock-valuation --help
```

The application currently exposes one subcommand:

```bash
uv run stock-valuation value TICKER
```

Example:

```bash
uv run stock-valuation value AAPL
```

The default output is a human-readable valuation report with these sections:

- Company Summary
- FCFF
- Growth
- Cost of Equity
- Cost of Debt
- WACC
- Forecast FCFF
- Terminal Value
- Discounting
- Equity Bridge
- Diagnostics

For machine-readable output, add `--json`:

```bash
uv run stock-valuation value AAPL --json
```

## Valuation Assumptions

Most rate inputs use decimal notation. For example, use `0.21` for 21%, not
`21`.

The CLI accepts these options:

```bash
uv run stock-valuation value AAPL \
  --forecast-years 5 \
  --company-country "United States" \
  --valuation-currency USD \
  --beta 1.2 \
  --tax-rate 0.21 \
  --erp 0.05 \
  --risk-free-rate 0.04 \
  --terminal-growth-rate 0.025 \
  --shares-outstanding 15400000000 \
  --market-debt 110000000000
```

Important runtime note: yfinance supplies company and statement data, but this
project does not ship concrete runtime implementations for every external
provider protocol. If a valuation needs macro rates, equity risk premium,
corporate tax rate, sovereign yield, FX, or market debt and no provider is
registered in code, provide the corresponding override from the CLI or an
assumptions file.

Common overrides:

- `--tax-rate`: corporate tax rate as a decimal
- `--erp`: equity risk premium as a decimal
- `--risk-free-rate`: risk-free rate as a decimal
- `--terminal-growth-rate`: terminal growth rate as a decimal
- `--market-debt`: market value of interest-bearing debt
- `--shares-outstanding`: share count if yfinance cannot provide one
- `--valuation-currency`: valuation currency override
- `--company-country`: country override for country-sensitive inputs

## Assumptions File

Instead of passing many flags, create a JSON assumptions file and pass it with
`--assumptions-file`.

Example `assumptions.json`:

```json
{
  "valuation_date": "2026-06-13",
  "forecast_years": 5,
  "company_country_override": "United States",
  "valuation_currency_override": "USD",
  "beta_override": "1.2",
  "tax_rate_override": "0.21",
  "equity_risk_premium_override": "0.05",
  "risk_free_rate_override": "0.04",
  "terminal_growth_rate_override": "0.025",
  "shares_outstanding_override": "15400000000",
  "market_debt_override": "110000000000"
}
```

Run:

```bash
uv run stock-valuation value AAPL --assumptions-file assumptions.json
```

CLI flags override values from the file only when the flag is supplied:

```bash
uv run stock-valuation value AAPL \
  --assumptions-file assumptions.json \
  --forecast-years 7 \
  --json
```

The assumptions file is strict. Unknown top-level fields, invalid dates, invalid
decimal values, and unsupported provider kinds raise typed errors instead of
being ignored.

## Provider Configuration

The assumptions file may include provider references:

```json
{
  "providers": {
    "macro": {
      "name": "example",
      "base_url": "https://example.test",
      "api_key_env_var": "EXAMPLE_API_KEY"
    }
  }
}
```

Supported provider kinds are:

- `macro`
- `erp`
- `fx`
- `tax_rate`
- `market_debt`
- `sovereign_yield`

Provider references are secret-safe. The JSON stores the environment variable
name, not the secret value. Set the secret separately:

```bash
export EXAMPLE_API_KEY="..."
uv run stock-valuation value AAPL --assumptions-file assumptions.json
```

Provider names must also be registered with provider factories in the Python
entrypoint. The stock CLI validates provider references and API key environment
variables, but it does not dynamically import arbitrary provider packages from
JSON.

The same provider references can be supplied through environment variables:

```bash
export STOCK_VALUATION_MACRO_PROVIDER=example
export STOCK_VALUATION_MACRO_PROVIDER_BASE_URL=https://example.test
export STOCK_VALUATION_MACRO_PROVIDER_API_KEY_ENV_VAR=EXAMPLE_API_KEY
export EXAMPLE_API_KEY="..."
```

Use the corresponding provider kind in the variable name, uppercased:
`MACRO`, `ERP`, `FX`, `TAX_RATE`, `MARKET_DEBT`, or `SOVEREIGN_YIELD`.

## Test

Run the full test suite:

```bash
uv run pytest
```

Run a specific test file:

```bash
uv run pytest tests/test_cli.py
```

The tests use fake data and provider doubles where needed, so they should not
depend on live yfinance responses.

## Troubleshooting

If `uv run ...` fails with a cache permission error, make sure `uv` can access
its cache directory or set a writable cache location:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

If `stock-valuation value TICKER` fails with a provider setup error, either
register a concrete provider implementation in code or pass the documented
override for the missing input.

If yfinance returns missing or empty data for a ticker, try a different public
ticker or provide explicit overrides for the missing valuation inputs. The CLI
prints typed errors and suggested actions without tracebacks or secrets.

## Project Layout

```text
src/stock_valuation/
  cli.py                 command-line entrypoint
  yfinance_client.py     ticker-bound yfinance access
  stock.py               normalized company and statement metrics
  processor.py           FCFF DCF valuation workflow
  contracts/             valuation result and assumption dataclasses
  providers/             external provider protocols and config
  errors/                typed project errors

tests/
  test_cli.py
  acceptance/
  clients/
  contracts/
  errors/
  providers/
  valuation/
```
