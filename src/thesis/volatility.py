"""Ingeniería de características y utilidades para modelado de volatilidad.

Todas las transformaciones son estrictamente causales: en el tiempo ``t``
solo se usa información disponible hasta ``t`` (inclusive).  El módulo no
depende de ``torch`` para que las features puedan construirse y probarse
sin aceleradoras ni redes neuronales.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from thesis.config import TRADING_DAYS
from thesis.timeseries import log_returns

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y limpia un DataFrame OHLCV descargado.

    - Asegura que las columnas ``Open, High, Low, Close, Volume`` existan.
    - Convierte el índice a ``DatetimeIndex`` y ordena cronológicamente.
    - Aplica ``ffill`` sobre NaN y descarta filas con precios no positivos.
    - Reconstruye la consistencia OHLC: ``High`` se eleva al máximo de
      ``Open``/``Close`` y ``Low`` se reduce al mínimo de ``Open``/``Close``.
      ``yfinance`` puede publicar barras intradía inconsistentes (por
      ejemplo, el cierre de la sesión en curso antes que su máximo), lo
      que sesgaría los estimadores de Parkinson y Garman-Klass.

    Parameters
    ----------
    df : pd.DataFrame
        Datos en bruto con al menos las columnas OHLCV.

    Returns
    -------
    pd.DataFrame
        Datos validados y ordenados, con las columnas en el orden canónico
        ``Open, High, Low, Close, Volume``.

    Raises
    ------
    ValueError
        Si faltan columnas obligatorias o hay precios no positivos.
    """
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {missing}")

    df = df[list(OHLCV_COLUMNS)].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    n_nan = df.isnull().sum().sum()
    if n_nan:
        df = df.ffill().dropna()

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Precio no positivo detectado en los datos OHLCV.")

    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


def download_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Descarga datos OHLCV desde Yahoo Finance y los valida.

    Parameters
    ----------
    ticker : str
        Símbolo del activo.
    start : str
        Fecha de inicio en formato ``YYYY-MM-DD``.
    end : str | None
        Fecha de fin (exclusiva).  Si es ``None`` se descarga hasta la
        fecha actual.
    auto_adjust : bool
        Si ``True`` ajusta por dividendos y splits.
    progress : bool
        Muestra la barra de progreso de ``yfinance``.

    Returns
    -------
    pd.DataFrame
        DataFrame validado con columnas ``Open, High, Low, Close, Volume``.
    """
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=auto_adjust, progress=progress
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return validate_ohlcv(raw)


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye características predictoras causales de volatilidad.

    Las características incluyen componentes HAR-RV (Corsi, 2009),
    estimadores de Parkinson y Garman-Klass, RSI, métricas de volumen,
    rango intradía, gap de apertura y estadísticos de retornos recientes.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV validados.

    Returns
    -------
    pd.DataFrame
        Características indexadas por fecha.  Las primeras filas contienen
        ``NaN`` por las ventanas de warm-up.
    """
    feat = pd.DataFrame(index=df.index)

    log_ret = log_returns(df["Close"])
    feat["log_return"] = log_ret

    # Componentes HAR-RV: ventanas que incluyen el día corriente.
    rv_sq = log_ret**2
    feat["rv_daily"] = rv_sq
    feat["rv_weekly"] = rv_sq.rolling(5, min_periods=5).mean()
    feat["rv_monthly"] = rv_sq.rolling(22, min_periods=22).mean()

    feat["vol_20"] = log_ret.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    feat["vol_ratio"] = feat["rv_daily"] / (feat["rv_weekly"] + 1e-10)

    # Estimador Parkinson con corrección C2.
    hl_log_sq = np.log(df["High"] / df["Low"]) ** 2
    feat["parkinson_vol"] = np.sqrt(
        hl_log_sq.rolling(20, min_periods=20).mean() / (4.0 * np.log(2))
    )

    # Estimador Garman-Klass.
    hl2 = 0.5 * np.log(df["High"] / df["Low"]) ** 2
    co2 = (2.0 * np.log(2) - 1.0) * np.log(df["Close"] / df["Open"]) ** 2
    feat["gk_vol"] = np.sqrt(
        (hl2 - co2).clip(lower=0).rolling(20, min_periods=20).mean()
    )

    # RSI 14 días normalizado a [0, 1].
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    feat["rsi_14"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0

    # Volumen relativo y tendencia.
    vol_ma20 = df["Volume"].rolling(20, min_periods=20).mean()
    feat["rel_volume"] = df["Volume"] / (vol_ma20 + 1e-10)
    feat["volume_trend"] = df["Volume"].rolling(5, min_periods=5).mean() / (
        vol_ma20 + 1e-10
    )

    # Rango intradía normalizado por cierre previo (causal).
    feat["intraday_range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)

    # Gap de apertura.
    feat["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Estadísticos de retornos recientes.
    feat["skew_5"] = log_ret.rolling(5, min_periods=5).skew()
    feat["mean_ret_5"] = log_ret.rolling(5, min_periods=5).mean()

    # Compresión logarítmica de variables estrictamente positivas.
    log_cols = [
        "rv_daily",
        "rv_weekly",
        "rv_monthly",
        "vol_ratio",
        "parkinson_vol",
        "gk_vol",
        "rel_volume",
        "volume_trend",
        "intraday_range",
    ]
    for col in log_cols:
        if col in feat.columns:
            feat[col] = np.log(feat[col].clip(lower=1e-12))

    return feat


def compute_log_vol_target(
    df: pd.DataFrame,
    horizon: int,
    annualize: int = TRADING_DAYS,
) -> pd.Series:
    """Log-volatilidad realizada anualizada a ``horizon`` días hacia adelante.

    Para cada fecha ``t``:

    .. math::
        y_t = \\log\\!\\left(\\sqrt{\\frac{A}{H}\\sum_{i=1}^{H} r_{t+i}^2}\\right)

    donde ``A`` es el factor de anualización y ``H`` el horizonte.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV con columna ``Close``.
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
    log_ret = log_returns(df["Close"])
    rv_sq = log_ret**2
    fwd_rv = rv_sq.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    target = np.sqrt((annualize / horizon) * fwd_rv)
    return np.log(target.clip(lower=1e-12)).rename("target_log_vol")


def build_volatility_dataset(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Combina features y target en un único DataFrame limpio.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV validados.
    horizon : int
        Horizonte del target.

    Returns
    -------
    pd.DataFrame
        Dataset completo sin ``NaN``.
    """
    features = compute_volatility_features(df)
    target = compute_log_vol_target(df, horizon)
    return pd.concat([features, target], axis=1).dropna()


def temporal_split(
    data: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide ``data`` en train, validation y test de forma cronológica.

    Las particiones son estrictamente disjuntas.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset con índice temporal.
    train_end : str
        Fecha final (inclusive) del conjunto de entrenamiento.
    val_end : str
        Fecha final (inclusive) del conjunto de validación.

    Returns
    -------
    tuple
        ``(train, validation, test)``.
    """
    train = data.loc[data.index <= train_end].copy()
    val = data.loc[(data.index > train_end) & (data.index <= val_end)].copy()
    test = data.loc[data.index > val_end].copy()
    return train, val, test


def purge_boundary_labels(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Elimina las filas cuyo label forward cruza la frontera del split.

    El target de :func:`compute_log_vol_target` en la posición ``t`` usa los
    retornos de las posiciones ``t+1 ... t+horizon``.  Por tanto, las últimas
    ``horizon`` filas de una partición tienen labels apoyados en observaciones
    de la partición siguiente: los últimos labels del train usan retornos de
    validación y los últimos de validación usan retornos de test.

    La purga es **posicional** porque el target se construye con ventanas
    móviles sobre filas (no sobre calendario): aunque falten sesiones, la
    ventana del label sigue siendo de ``horizon`` observaciones consecutivas.

    Parameters
    ----------
    df : pd.DataFrame
        Partición de entrenamiento o validación.
    horizon : int
        Horizonte de predicción en sesiones.

    Returns
    -------
    pd.DataFrame
        Copia de ``df`` sin sus últimas ``horizon`` filas.
    """
    if horizon <= 0:
        return df.copy()
    if horizon >= len(df):
        return df.iloc[:0].copy()
    return df.iloc[:-horizon].copy()


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye pares (secuencia, target) con ventana deslizante.

    Para la secuencia ``i``:

    - Características: ``X[i : i + lookback]``.
    - Target: ``y[i + lookback]``.

    Parameters
    ----------
    X : np.ndarray
        Características de forma ``(T, n_features)``.
    y : np.ndarray
        Target de forma ``(T,)``.
    lookback : int
        Longitud de la ventana histórica.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X_seq, y_seq)`` con formas ``(T - lookback, lookback, n_features)``
        y ``(T - lookback,)``.
    """
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def align_har_features_with_sequences(
    features_df: pd.DataFrame,
    y_seq: np.ndarray,
    lookback: int,
) -> pd.DataFrame:
    """Alinea características raw del HAR-RV con las secuencias LSTM.

    Cada secuencia ``i`` usa información hasta el índice ``i + lookback - 1``
    dentro de su partición.  Esta función devuelve un DataFrame con el vector
    de características de ese último día para cada target.

    Parameters
    ----------
    features_df : pd.DataFrame
        Features de la partición (sin estandarizar).
    y_seq : np.ndarray
        Targets de las secuencias de la misma partición.
    lookback : int
        Longitud de la ventana histórica.

    Returns
    -------
    pd.DataFrame
        Características HAR alineadas, una fila por secuencia.
    """
    start = lookback - 1
    end = start + len(y_seq)
    return features_df.iloc[start:end].reset_index(drop=True)


def scale_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Escala train/val/test y devuelve arrays y escaladores.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Características de cada partición.
    y_train, y_val, y_test : np.ndarray
        Targets de cada partición.

    Returns
    -------
    dict
        Diccionario con arrays escalados y los objetos ``StandardScaler``.
    """
    from sklearn.preprocessing import StandardScaler

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = X_scaler.fit_transform(X_train)
    X_val_s = X_scaler.transform(X_val)
    X_test_s = X_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train_s,
        "y_val": y_val_s,
        "y_test": y_test_s,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
    }


def count_extreme_moves(
    df: pd.DataFrame,
    threshold: float = 0.15,
) -> int:
    """Cuenta movimientos diarios absolutos mayores a ``threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Datos OHLCV.
    threshold : float
        Umbral absoluto en retornos logarítmicos.

    Returns
    -------
    int
        Número de movimientos extremos.
    """
    log_ret = log_returns(df["Close"])
    return int((np.abs(log_ret) > threshold).sum())
