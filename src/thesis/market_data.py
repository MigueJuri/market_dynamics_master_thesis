"""Carga de precios de cierre ajustado con caché CSV local."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

from thesis.config import DATA_DIR, END, START, TICKER


def load_single_price(
    ticker: str = TICKER,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
) -> pd.Series:
    """Precio de cierre ajustado de ``ticker`` con caché CSV en ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ticker}_{start}_{end}.csv"

    if cache.exists() and not refresh:
        px = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze("columns")
    else:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=False, progress=False)
        if raw.empty:
            return pd.Series(dtype="float64", name=ticker)
        px = raw["Adj Close"].dropna().squeeze()
        px.to_csv(cache)

    px.name = ticker
    return px


def load_multiple_prices(
    tickers: Sequence[str],
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Descarga cada ticker, persiste su CSV y devuelve un frame combinado."""
    series_list = [
        s for s in (load_single_price(t, start, end, cache_dir, refresh)
                    for t in tickers)
        if not s.empty
    ]
    return pd.concat(series_list, axis=1) if series_list else pd.DataFrame()
