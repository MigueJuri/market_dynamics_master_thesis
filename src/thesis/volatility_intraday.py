"""Ingeniería de características para barras intradía de Alpaca.

Este módulo descarga barras OHLCV de Alpaca y las reduce a **estimadores
diarios**: cada sesión se resume con OHLCV diario y una varianza realizada
intradía que excluye el salto overnight de la primera barra.  A partir de
ese frame diario se reutilizan las features y el target de
:mod:`thesis.volatility`, de modo que el notebook Alpaca comparte la misma
estructura diaria ``(lookback=5 días, n_features=13)`` que el notebook de
Yahoo Finance.

Convenciones
------------
* El timestamp de Alpaca es el **inicio** de la barra.  Las barras se indexan
  por la hora local del mercado (``America/New_York``) sin zona horaria, para
  que las fronteras temporales se interpreten en hora de mercado.
* ``rv_intraday`` es la suma de los retornos al cuadrado **dentro de la
  sesión**; el salto de la primera barra del día respecto al cierre del día
  anterior no se incluye.
* ``rv_daily``, ``rv_weekly`` y ``rv_monthly`` se calculan a partir de
  ``rv_intraday`` (no de los retornos cierre-a-cierre), de modo que los
  componentes HAR-RV miden puramente volatilidad intradía.  ``gap`` y
  ``skew_5`` siguen usando retornos diarios cierre-a-cierre, lo que documenta
  explícitamente la convención mixta.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thesis.config import TRADING_DAYS
from thesis.timeseries import log_returns
from thesis.volatility import (
    LOG_COLS,
    OHLCV_COLUMNS,
    compute_volatility_features,
    validate_ohlcv,
)

#: Zona horaria del mercado de acciones de EE. UU.
MARKET_TZ: str = "America/New_York"


def _parse_timeframe_minutes(timeframe: str) -> int:
    """Extrae los minutos por barra de un timeframe de Alpaca.

    Soporta timeframes en minutos y horas (p. ej. ``'1Min'``, ``'5Min'``,
    ``'15Min'``, ``'1Hour'``).

    Parameters
    ----------
    timeframe : str
        Temporalidad con el formato ``<cantidad><Unidad>``.

    Returns
    -------
    int
        Minutos contenidos en cada barra.

    Raises
    ------
    ValueError
        Si el timeframe no tiene cantidad numérica o no está en minutos/horas.
    """
    text = timeframe.strip()
    amount_text = "".join(ch for ch in text if ch.isdigit())
    unit_text = "".join(ch for ch in text if not ch.isdigit()).lower()
    if not amount_text:
        raise ValueError(f"Timeframe no válido: {timeframe!r}")
    amount = int(amount_text)
    if unit_text in ("min", "m", "minute", "minutes", "minuto", "minutos"):
        return amount
    if unit_text in ("hour", "h", "hours", "hora", "horas"):
        return amount * 60
    raise ValueError(
        f"Timeframe no soportado para inferencia automática: {timeframe!r}. "
        "Usa minutos u horas, o proporciona bars_per_day explícitamente."
    )


def infer_frequency_params(
    timeframe: str,
    trading_days: int = TRADING_DAYS,
    market_minutes: float = 390.0,
) -> dict[str, int]:
    """Infiere ``bars_per_day`` y ``bars_per_year`` a partir del timeframe.

    Asume una sesión de mercado de 6.5 horas (390 minutos), que es el
    horario habitual de NYSE/Nasdaq.  Para 1, 5 y 15 minutos la división es
    exacta (390, 78 y 26 barras respectivamente); para otros timeframes el
    resultado se redondea al entero más cercano y puede sobreescribirse
    pasando ``bars_per_day`` de forma explícita.

    Parameters
    ----------
    timeframe : str
        Temporalidad de las barras (p. ej. ``'1Min'``, ``'5Min'``,
        ``'15Min'``).
    trading_days : int
        Días de mercado por año (252 por defecto).
    market_minutes : float
        Minutos de mercado por sesión (390 por defecto).

    Returns
    -------
    dict[str, int]
        ``{"timeframe", "minutes_per_bar", "bars_per_day", "bars_per_year"}``.

    Raises
    ------
    ValueError
        Si no se puede inferir un ``bars_per_day`` positivo.
    """
    minutes_per_bar = _parse_timeframe_minutes(timeframe)
    bars_per_day = round(market_minutes / minutes_per_bar)
    if bars_per_day <= 0:
        raise ValueError(
            f"No se pudo inferir bars_per_day para {timeframe!r} "
            f"({minutes_per_bar} min/bar)."
        )
    return {
        "timeframe": timeframe,
        "minutes_per_bar": minutes_per_bar,
        "bars_per_day": bars_per_day,
        "bars_per_year": bars_per_day * trading_days,
    }


def _build_timeframe(timeframe: str) -> Any:
    """Traduce ``'15Min'`` al ``TimeFrame`` de ``alpaca-py``.

    Parameters
    ----------
    timeframe : str
        Temporalidad con el formato ``<cantidad><Unidad>``, donde ``Unidad``
        es ``Min``, ``Hour``, ``Day``, ``Week`` o ``Month``.

    Returns
    -------
    Any
        Instancia de ``alpaca.data.timeframe.TimeFrame``.
    """
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    units = {
        "min": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
        "week": TimeFrameUnit.Week,
        "month": TimeFrameUnit.Month,
    }
    text = timeframe.strip()
    amount_text = "".join(ch for ch in text if ch.isdigit())
    unit_text = "".join(ch for ch in text if not ch.isdigit()).lower()
    if not amount_text or unit_text not in units:
        raise ValueError(
            f"Timeframe no soportado: {timeframe!r}. Usa '<cantidad><Unidad>' "
            "con Unidad en {Min, Hour, Day, Week, Month}."
        )
    return TimeFrame(int(amount_text), units[unit_text])


def _utc_timestamp(value: str) -> pd.Timestamp:
    """Normaliza una fecha a UTC (naive ⇒ se interpreta como UTC)."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _cache_path_alpaca(
    ticker: str,
    start: str,
    end: str | None,
    timeframe: str,
    feed: str,
    adjustment: str,
    cache_dir: Path,
) -> Path:
    """Ruta de caché CSV siguiendo el patrón de ``market_data.load_single_price``."""
    end_tag = end if end is not None else "now"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}_{timeframe}_{feed}_{adjustment}_{start}_{end_tag}.csv"


def download_ohlcv_alpaca(
    ticker: str,
    start: str,
    end: str | None = None,
    timeframe: str = "15Min",
    feed: str = "iex",
    adjustment: str = "all",
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Descarga barras OHLCV de Alpaca y las valida, con caché CSV opcional.

    Las credenciales se leen de las variables de entorno ``ALPACA_API_KEY`` y
    ``ALPACA_SECRET_KEY`` (el archivo ``.env`` está ignorado por Git; con
    ``uv`` se puede pasar con ``uv run --env-file .env ...``).

    El timestamp de Alpaca marca el **inicio** de la barra.  El índice
    devuelto está en hora local del mercado (``America/New_York``) y sin zona
    horaria.

    Parameters
    ----------
    ticker : str
        Símbolo del activo (p. ej. ``'SPY'``).
    start : str
        Fecha de inicio en formato ``YYYY-MM-DD`` (UTC).
    end : str | None
        Fecha de fin (exclusiva).  Si es ``None`` se descarga hasta ahora.
    timeframe : str
        Temporalidad de las barras (p. ej. ``'15Min'``).
    feed : str
        Fuente de datos (``'iex'`` para el feed gratuito, ``'sip'`` para el
        feed de pago).
    adjustment : str
        Ajuste por acciones corporativas (``'raw'``, ``'split'``,
        ``'dividend'`` o ``'all'``).
    cache_dir : Path | None
        Directorio donde leer/escribir un CSV de caché.  Si es ``None`` no
        hay persistencia.
    refresh : bool
        Si ``True`` fuerza la descarga aunque exista caché.

    Returns
    -------
    pd.DataFrame
        Datos validados con columnas ``Open, High, Low, Close, Volume``.

    Raises
    ------
    RuntimeError
        Si faltan las credenciales de Alpaca.
    ValueError
        Si Alpaca no devuelve barras para el periodo solicitado.
    """
    if cache_dir is not None:
        cache = _cache_path_alpaca(
            ticker, start, end, timeframe, feed, adjustment, Path(cache_dir)
        )
        if cache.exists() and not refresh:
            raw = pd.read_csv(cache, index_col=0, parse_dates=True)
            return validate_ohlcv(raw)

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Faltan las credenciales de Alpaca: define ALPACA_API_KEY y "
            "ALPACA_SECRET_KEY en el entorno (p. ej. `uv run --env-file .env ...`)."
        )

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=_build_timeframe(timeframe),
        start=_utc_timestamp(start),
        end=None if end is None else _utc_timestamp(end),
        adjustment=Adjustment(adjustment),
        feed=DataFeed(feed),
    )

    bars = client.get_stock_bars(request)
    raw = bars.df
    if raw.empty:
        raise ValueError(
            f"Alpaca no devolvió barras para {ticker} "
            f"({timeframe}, feed={feed}, {start} → {end})."
        )

    # ``bars.df`` viene indexado por (símbolo, timestamp) con columnas en
    # minúsculas: se conserva sólo el símbolo pedido y se deja el timestamp
    # como índice.
    if isinstance(raw.index, pd.MultiIndex):
        symbols = raw.index.get_level_values(0)
        raw = raw[symbols == ticker]
        raw.index = raw.index.droplevel(0)
    if raw.empty:
        raise ValueError(f"Alpaca no devolvió barras para {ticker}.")
    raw = raw.rename(columns={c: str(c).capitalize() for c in raw.columns})

    # Hora local del mercado, sin zona horaria (el mercado no opera en el
    # instante del cambio de horario, así que no hay timestamps duplicados).
    raw.index = (
        pd.to_datetime(raw.index, utc=True).tz_convert(MARKET_TZ).tz_localize(None)
    )

    raw = validate_ohlcv(raw)

    if cache_dir is not None:
        raw.to_csv(cache)

    return raw


def aggregate_intraday_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce barras intradía a un frame diario con ``rv_intraday``.

    La agregación OHLCV es la convencional: ``Open`` = primera barra,
    ``High`` = máximo, ``Low`` = mínimo, ``Close`` = última barra y
    ``Volume`` = suma.  ``rv_intraday`` es la suma de los retornos al
    cuadrado **dentro de la sesión**, excluyendo el salto overnight de la
    primera barra del día.

    Parameters
    ----------
    df : pd.DataFrame
        Datos intradía validados con columnas OHLCV.

    Returns
    -------
    pd.DataFrame
        Frame diario con columnas ``Open, High, Low, Close, Volume,
        rv_intraday``.
    """
    df = validate_ohlcv(df)

    close = df["Close"]
    log_ret = log_returns(close)
    day = pd.Series(df.index.normalize(), index=df.index)
    overnight = day != day.shift(1)
    intraday_ret = log_ret.mask(overnight)
    rv_intraday = (intraday_ret**2).groupby(day).sum()

    daily = df.groupby(pd.Grouper(freq="D")).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    daily["rv_intraday"] = rv_intraday
    return daily.dropna()


def compute_volatility_features_from_intraday_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Features de volatilidad diaria usando la RV intradía.

    Reutiliza :func:`thesis.volatility.compute_volatility_features` sobre el
    OHLCV diario y sobrescribe ``rv_daily``, ``rv_weekly``, ``rv_monthly`` y
    ``vol_ratio`` con estimadores construidos a partir de ``rv_intraday``.
    La compresión logarítmica es la misma que en el módulo diario.

    Parameters
    ----------
    daily : pd.DataFrame
        Frame diario producido por :func:`aggregate_intraday_to_daily`.

    Returns
    -------
    pd.DataFrame
        Características causales con las mismas columnas que el módulo
        diario.
    """
    ohlcv = daily[list(OHLCV_COLUMNS)]
    feat = compute_volatility_features(ohlcv)

    rv = daily["rv_intraday"]
    feat["rv_daily"] = rv
    feat["rv_weekly"] = rv.rolling(5, min_periods=5).mean()
    feat["rv_monthly"] = rv.rolling(22, min_periods=22).mean()
    feat["vol_ratio"] = rv / (feat["rv_weekly"] + 1e-10)

    for col in LOG_COLS:
        if col in feat.columns:
            feat[col] = np.log(feat[col].clip(lower=1e-12))

    return feat


def compute_log_vol_target_from_intraday_daily(
    daily: pd.DataFrame,
    horizon: int,
    annualize: int = TRADING_DAYS,
) -> pd.Series:
    """Log-volatilidad realizada anualizada a ``horizon`` días hacia adelante.

    Usa ``rv_intraday`` como medida de varianza realizada diaria, sin
    incluir el salto overnight:

    .. math::
        y_t = \\log\\!\\left(\\sqrt{\\frac{A}{H}\\sum_{i=1}^{H} RV_{t+i}^{intradía}}\\right)

    Parameters
    ----------
    daily : pd.DataFrame
        Frame diario con columna ``rv_intraday``.
    horizon : int
        Horizonte de predicción en sesiones.
    annualize : int
        Factor de anualización (por defecto 252).

    Returns
    -------
    pd.Series
        Target indexado por fecha.  Las últimas ``horizon`` observaciones
        son ``NaN`` porque requieren datos futuros.
    """
    rv = daily["rv_intraday"]
    fwd_rv = rv.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    target = np.sqrt((annualize / horizon) * fwd_rv)
    return np.log(target.clip(lower=1e-12)).rename("target_log_vol")


def build_volatility_dataset_from_intraday(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Combina features y target diarios en un único DataFrame limpio.

    Parameters
    ----------
    df : pd.DataFrame
        Datos intradía validados.
    horizon : int
        Horizonte del target en sesiones.

    Returns
    -------
    pd.DataFrame
        Dataset completo sin ``NaN``.
    """
    daily = aggregate_intraday_to_daily(df)
    features = compute_volatility_features_from_intraday_daily(daily)
    target = compute_log_vol_target_from_intraday_daily(daily, horizon)
    return pd.concat([features, target], axis=1).dropna()
