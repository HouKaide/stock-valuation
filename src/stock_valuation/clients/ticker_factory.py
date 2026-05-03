"""Factory helpers for constructing Yahoo Finance ticker objects."""

from __future__ import annotations

from typing import Callable, cast
import yfinance as yf

from stock_valuation.clients.ticker_protocol import TickerProtocol

TickerFactory = Callable[[str], TickerProtocol]


def default_ticker_factory(symbol: str) -> TickerProtocol:
    """Create a `yfinance.Ticker` instance on demand.

    Args:
        symbol: Normalized ticker symbol.

    Returns:
        A `TickerProtocol`-compatible Yahoo Finance ticker object.
    """

    return cast(TickerProtocol, yf.Ticker(symbol))
