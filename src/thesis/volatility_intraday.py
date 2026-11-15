"""Ingeniería de características para barras intradía (p. ej. 15 minutos).

Espejo de :mod:`thesis.volatility` para datos de frecuencia intradía: las
ventanas expresadas en sesiones se reescalan a **barras** mediante
``bars_per_day`` y la anualización usa el número de barras por año.

Convenciones
------------
* El timestamp de Alpaca es el **inicio** de la barra.  Las barras se indexan
  por la hora local del mercado (``America/New_York``) sin zona horaria, para
  que las fronteras temporales (``temporal_split``) se interpreten en hora
  de mercado.
* ``rv_daily`` es la suma de los retornos al cuadrado de las barras del día
  (26 barras con ``bars_per_day=26``); ``rv_weekly`` y ``rv_monthly`` son
  medias de los retornos al cuadrado sobre 130 y 572 barras.  En el módulo
  diario las mismas magnitudes son la media de 5 y 22 observaciones diarias:
  la diferencia es un factor constante (26) que se absorbe en el intercepto
  de la regresión HAR-RV y en la estandarización de la LSTM.
* Los estimadores de rango (Parkinson, Garman-Klass) quedan en unidades por
  barra, igual que en el módulo diario quedan en unidades por sesión.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from thesis.config import TRADING_DAYS
from thesis.timeseries import log_returns
from thesis.volatility import LOG_COLS, validate_ohlcv

#: Zona horaria del mercado de acciones de EE. UU.
MARKET_TZ: str = "America/New_York"

#: Ventanas del módulo diario, expresadas en sesiones.
DAILY_WINDOWS: dict[str, int] = {
    "rv_daily": 1,
    "rv_weekly": 5,
    "rv_monthly": 22,
    "vol_20": 20,
    "rsi_14": 14,
    "rel_volume": 20,
    "volume_trend": 5,
    "skew_5": 5,
    "mean_ret_5": 5,
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


def download_ohlcv_alpaca(
    ticker: str,
    start: str,
    end: str | None = None,
    timeframe: str = "15Min",
    feed: str = "iex",
    adjustment: str = "all",
) -> pd.DataFrame:
    """Descarga barras OHLCV de Alpaca y las valida.

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
        Fuente de datos (``'iex'`` para el feed gratuito).
    adjustment : str
        Ajuste por acciones corporativas (``'raw'``, ``'split'``,
        ``'dividend'`` o ``'all'``).

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

    return validate_ohlcv(raw)


def compute_volatility_features_intraday(
    df: pd.DataFrame,
    bars_per_day: int = 26,
    bars_per_year: int | None = None,
) -> pd.DataFrame:
    """Construye características predictoras causales sobre barras.

    Espejo de :func:`thesis.volatility.compute_volatility_features` con las
    ventanas diarias reescaladas a barras:

    ==================  ==========  ==========================================
    Característica      Barras      Equivalente diario
    ==================  ==========  ==========================================
    ``rv_daily``        26          suma de ``r²`` de las barras del día
    ``rv_weekly``       130         media de ``r²`` de 5 sesiones
    ``rv_monthly``      572         media de ``r²`` de 22 sesiones
    ``vol_20``          520         desviación de 20 sesiones (anualizada)
    ``parkinson_vol``   520         Parkinson de 20 sesiones
    ``gk_vol``          520         Garman-Klass de 20 sesiones
    ``rsi_14``          364         RSI de 14 sesiones
    ``rel_volume``      520         volumen relativo a 20 sesiones
    ``volume_trend``    130         tendencia de volumen de 5 sesiones
    ``skew_5``          130         asimetría de 5 sesiones
    ``mean_ret_5``      130         retorno medio de 5 sesiones
    ==================  ==========  ==========================================

    ``gap`` e ``intraday_range`` no tienen ventana: comparan cada barra con
    la anterior.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV validados.
    bars_per_day : int
        Número de barras por sesión (26 para barras de 15 minutos).
    bars_per_year : int | None
        Barras por año para anualizar.  Por defecto ``bars_per_day * 252``.

    Returns
    -------
    pd.DataFrame
        Características indexadas por fecha.  Las primeras filas contienen
        ``NaN`` por las ventanas de warm-up.
    """
    if bars_per_day <= 0:
        raise ValueError("bars_per_day debe ser un entero positivo.")
    if bars_per_year is None:
        bars_per_year = bars_per_day * TRADING_DAYS

    w = {name: days * bars_per_day for name, days in DAILY_WINDOWS.items()}

    feat = pd.DataFrame(index=df.index)

    log_ret = log_returns(df["Close"])
    feat["log_return"] = log_ret

    # Componentes HAR-RV: la varianza realizada diaria es la suma de los
    # retornos al cuadrado de las barras del día; las medias de 5 y 22
    # sesiones se calculan sobre los retornos al cuadrado de las barras.
    rv_sq = log_ret**2
    feat["rv_daily"] = rv_sq.rolling(w["rv_daily"], min_periods=w["rv_daily"]).sum()
    feat["rv_weekly"] = rv_sq.rolling(w["rv_weekly"], min_periods=w["rv_weekly"]).mean()
    feat["rv_monthly"] = rv_sq.rolling(w["rv_monthly"], min_periods=w["rv_monthly"]).mean()

    feat["vol_20"] = (
        log_ret.rolling(w["vol_20"], min_periods=w["vol_20"]).std()
        * np.sqrt(bars_per_year)
    )
    feat["vol_ratio"] = feat["rv_daily"] / (feat["rv_weekly"] + 1e-10)

    # Estimador Parkinson con corrección C2.
    hl_log_sq = np.log(df["High"] / df["Low"]) ** 2
    feat["parkinson_vol"] = np.sqrt(
        hl_log_sq.rolling(w["vol_20"], min_periods=w["vol_20"]).mean()
        / (4.0 * np.log(2))
    )

    # Estimador Garman-Klass.
    hl2 = 0.5 * np.log(df["High"] / df["Low"]) ** 2
    co2 = (2.0 * np.log(2) - 1.0) * np.log(df["Close"] / df["Open"]) ** 2
    feat["gk_vol"] = np.sqrt(
        (hl2 - co2).clip(lower=0).rolling(w["vol_20"], min_periods=w["vol_20"]).mean()
    )

    # RSI de 14 sesiones normalizado a [0, 1].
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(w["rsi_14"], min_periods=w["rsi_14"]).mean()
    loss = (-delta.clip(upper=0)).rolling(w["rsi_14"], min_periods=w["rsi_14"]).mean()
    rs = gain / (loss + 1e-10)
    feat["rsi_14"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0

    # Volumen relativo y tendencia.
    vol_ma = df["Volume"].rolling(w["rel_volume"], min_periods=w["rel_volume"]).mean()
    feat["rel_volume"] = df["Volume"] / (vol_ma + 1e-10)
    feat["volume_trend"] = (
        df["Volume"].rolling(w["volume_trend"], min_periods=w["volume_trend"]).mean()
        / (vol_ma + 1e-10)
    )

    # Rango de la barra normalizado por el cierre previo (causal).
    feat["intraday_range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)

    # Gap respecto a la barra anterior.
    feat["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Estadísticos de retornos recientes.
    feat["skew_5"] = log_ret.rolling(w["skew_5"], min_periods=w["skew_5"]).skew()
    feat["mean_ret_5"] = log_ret.rolling(w["mean_ret_5"], min_periods=w["mean_ret_5"]).mean()

    # Compresión logarítmica de variables estrictamente positivas (las mismas
    # columnas que en el módulo diario).
    for col in LOG_COLS:
        if col in feat.columns:
            feat[col] = np.log(feat[col].clip(lower=1e-12))

    return feat


def compute_log_vol_target_intraday(
    df: pd.DataFrame,
    horizon_bars: int = 130,
    annualize: int = 6552,
) -> pd.Series:
    """Log-volatilidad realizada anualizada a ``horizon_bars`` barras.

    Para cada barra ``t``:

    .. math::
        y_t = \\log\\!\\left(\\sqrt{\\frac{A}{H}\\sum_{i=1}^{H} r_{t+i}^2}\\right)

    donde ``A`` es el factor de anualización (6552 para barras de 15
    minutos) y ``H`` el horizonte en barras.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV con columna ``Close``.
    horizon_bars : int
        Horizonte de predicción en barras.
    annualize : int
        Factor de anualización (por defecto 6552).

    Returns
    -------
    pd.Series
        Target indexado por fecha.  Las últimas ``horizon_bars``
        observaciones son ``NaN`` porque requieren datos futuros.
    """
    log_ret = log_returns(df["Close"])
    rv_sq = log_ret**2
    fwd_rv = rv_sq.rolling(horizon_bars, min_periods=horizon_bars).sum().shift(
        -horizon_bars
    )
    target = np.sqrt((annualize / horizon_bars) * fwd_rv)
    return np.log(target.clip(lower=1e-12)).rename("target_log_vol")


def build_volatility_dataset_intraday(
    df: pd.DataFrame,
    horizon_bars: int,
    bars_per_day: int = 26,
    bars_per_year: int | None = None,
) -> pd.DataFrame:
    """Combina features y target intradía en un único DataFrame limpio.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV validados.
    horizon_bars : int
        Horizonte del target en barras.
    bars_per_day : int
        Número de barras por sesión.
    bars_per_year : int | None
        Barras por año para anualizar.  Por defecto ``bars_per_day * 252``.

    Returns
    -------
    pd.DataFrame
        Dataset completo sin ``NaN``.
    """
    if bars_per_year is None:
        bars_per_year = bars_per_day * TRADING_DAYS
    features = compute_volatility_features_intraday(
        df, bars_per_day=bars_per_day, bars_per_year=bars_per_year
    )
    target = compute_log_vol_target_intraday(
        df, horizon_bars=horizon_bars, annualize=bars_per_year
    )
    return pd.concat([features, target], axis=1).dropna()
