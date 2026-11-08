"""Análisis de BTC: descarga OHLCV, power-law y ciclos de mercado.

El módulo concentra la lógica reutilizable del cuaderno ``notebooks/btc/BTC.py``:

1. descarga del OHLCV diario de Bitstamp con caché local en ``DATA_DIR``;
2. regresión cuantílica power-law ``log10(price) = α + β·log10(days)``;
3. detección de picos y valles macro con ``scipy.signal.find_peaks``;
4. detrend por banda de cuantil, candidatos de extremos locales y
   construcción del **ciclo idealizado** (poligonal por los extremos
   elegidos por el autor).

Importar el módulo **no** descarga datos ni dibuja figuras: los efectos con
red, disco o pantalla viven dentro de funciones explícitas.  El módulo no se
exporta en ``thesis/__init__.py`` para no arrastrar ``statsmodels`` en cada
import del paquete; se usa ``from thesis.btc import ...``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from scipy.signal import find_peaks

from thesis.config import DATA_DIR

#: Columnas canónicas del OHLCV diario de BTC (minúsculas, como en Bitstamp).
BTC_OHLCV_COLUMNS: tuple[str, ...] = ("open", "close", "high", "low", "volume")

#: Nombre del CSV de caché dentro de ``cache_dir``.
BTC_CACHE_FILENAME: str = "btc_daily_ohlcv.csv"

#: Plantilla de la API OHLC de Bitstamp.
BITSTAMP_OHLC_URL: str = "https://www.bitstamp.net/api/v2/ohlc/{symbol}/"

#: Fecha de génesis usada para medir los días transcurridos.
BTC_GENESIS: str = "2009-01-03"

#: Cuantililes por defecto de la regresión power-law.
BTC_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

#: Cuantil de la banda usada para detrender el precio (suelo del power-law).
BTC_CYCLE_QUANTILE: float = 0.10

#: Distancia mínima (días) entre extremos candidatos del ciclo idealizado.
BTC_CYCLE_DISTANCE: int = 200

#: Prominencia mínima (en log10) de los extremos candidatos del ciclo.
BTC_CYCLE_PROMINENCE: float = 0.05

#: Ventana por defecto (en días) del promedio móvil del precio dettrendizado.
BTC_CYCLE_SMOOTH_WINDOW: int = 30

#: Halvings del protocolo (anotaciones descriptivas, no señales).
BTC_HALVINGS: tuple[str, ...] = (
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
)

#: Picos macro anotados por el autor (incluye fechas futuras: sólo son
#: líneas de referencia del gráfico, no señales de mercado).
BTC_PEAKS: tuple[str, ...] = (
    "2013-12-04",
    "2017-12-16",
    "2021-11-08",
    "2025-10-06",
)

#: Valles macro anotados por el autor (ídem: anotaciones descriptivas).
BTC_BOTTOMS: tuple[str, ...] = (
    "2015-01-14",
    "2018-12-15",
    "2022-11-21",
    "2026-10-06",
)

#: Pico de Mt. Gox, anterior a los datos de Bitstamp; se inyecta a mano.
MTGOX_PEAK_DATE: str = "2011-06-08"
MTGOX_PEAK_CLOSE: float = 31.90

_QUANTILE_COLORS: dict[float, str] = {0.10: "green", 0.50: "blue", 0.90: "red"}
_QUANTILE_STYLES: dict[float, str] = {0.10: "--", 0.50: "-.", 0.90: "--"}


def _as_dates(dates: Sequence[Any]) -> list[pd.Timestamp]:
    """Convierte una secuencia de fechas (texto o ``Timestamp``) a timestamps."""
    return [pd.Timestamp(str(d).strip()) for d in dates]


def validate_btc_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el OHLCV diario de BTC y devuelve el frame canónico.

    - Exige las columnas ``open, close, high, low, volume``.
    - Convierte el índice a ``DatetimeIndex`` y exige que sea monótono.
    - Rechaza ``NaN`` y precios no positivos.

    Parameters
    ----------
    df : pd.DataFrame
        Datos en bruto con las columnas OHLCV de BTC.

    Returns
    -------
    pd.DataFrame
        Datos validados con las columnas en el orden canónico.

    Raises
    ------
    ValueError
        Si faltan columnas, hay NaN, precios no positivos o el índice no es
        monótono creciente.
    """
    missing = set(BTC_OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {missing}")

    out = df[list(BTC_OHLCV_COLUMNS)].copy()
    out.index = pd.to_datetime(out.index)

    if not out.index.is_monotonic_increasing:
        raise ValueError(
            "El índice de los datos OHLCV de BTC no es monótono creciente."
        )

    if out.isna().any().any():
        raise ValueError("Valores NaN detectados en los datos OHLCV de BTC.")

    if (out[["open", "close", "high", "low"]] <= 0).any().any():
        raise ValueError("Precio no positivo detectado en los datos OHLCV de BTC.")

    return out


def _fetch_raw_ohlcv(
    symbol: str,
    session: requests.Session | None = None,
    step: int = 86400,
    limit: int = 1000,
    pause: float = 0.1,
) -> list[dict[str, Any]]:
    """Pagina la API OHLC de Bitstamp y devuelve las velas en bruto."""
    sess = session if session is not None else requests.Session()
    url = BITSTAMP_OHLC_URL.format(symbol=symbol)

    # Bitstamp sólo publica datos desde agosto de 2011.
    start_timestamp = int(datetime(2010, 8, 1, tzinfo=timezone.utc).timestamp())
    end_timestamp = int(datetime.now(tz=timezone.utc).timestamp())

    all_data: list[dict[str, Any]] = []
    current_start = start_timestamp

    while current_start < end_timestamp:
        params = {"step": step, "limit": limit, "start": current_start}
        response = sess.get(url, params=params)
        response.raise_for_status()

        data = response.json().get("data", {}).get("ohlc", [])
        if not data:
            break

        all_data.extend(data)

        # Continúa desde la última vela recibida más un paso.
        current_start = int(data[-1]["timestamp"]) + step

        # Pausa para respetar el límite de la API (800 peticiones/minuto).
        time.sleep(pause)

    return all_data


def _raw_to_frame(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """Convierte las velas crudas de Bitstamp en un frame OHLCV indexado."""
    if not raw:
        raise ValueError("Bitstamp no devolvió ninguna vela OHLCV.")

    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s")
    df = df.set_index("timestamp")

    missing = set(BTC_OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLCV en la respuesta: {missing}")

    df = df[list(BTC_OHLCV_COLUMNS)].astype(float)

    # Descarta duplicados por ventanas solapadas y ordena cronológicamente.
    return df[~df.index.duplicated(keep="first")].sort_index()


def fetch_btc_ohlcv(
    symbol: str = "btcusd",
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Descarga el OHLCV diario de BTC de Bitstamp con caché CSV local.

    Si la caché existe y ``refresh`` es ``False`` se lee del disco; si no, se
    pagina la API y se escribe ``btc_daily_ohlcv.csv`` en ``cache_dir``
    (anclado a ``thesis.config.DATA_DIR``, no al directorio de trabajo).

    Parameters
    ----------
    symbol : str
        Par de Bitstamp, por defecto ``"btcusd"``.
    cache_dir : Path
        Directorio de caché (por defecto ``thesis.config.DATA_DIR``).
    refresh : bool
        Si ``True`` ignora la caché y vuelve a descargar.
    session : requests.Session | None
        Sesión HTTP inyectable (para tests o para reutilizar conexiones).

    Returns
    -------
    pd.DataFrame
        OHLCV diario validado, indexado por fecha.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / BTC_CACHE_FILENAME

    if cache.exists() and not refresh:
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index)
        return validate_btc_ohlcv(cached)

    frame = _raw_to_frame(_fetch_raw_ohlcv(symbol, session=session))
    validated = validate_btc_ohlcv(frame)
    validated.to_csv(cache)
    return validated


def prepare_power_law_frame(
    df: pd.DataFrame,
    genesis: str = BTC_GENESIS,
) -> pd.DataFrame:
    """Añade las variables del power-law: ``days``, ``log_days``, ``log_price``.

    La transformación es causal: ``days`` depende sólo de la fecha de cada
    fila, de modo que truncar la serie no altera las filas ya calculadas.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV diario de BTC con columna ``close``.
    genesis : str
        Fecha de génesis del activo (por defecto ``"2009-01-03"``).

    Returns
    -------
    pd.DataFrame
        Copia del frame con las filas ``days > 0`` y ``close > 0`` y las tres
        columnas adicionales.
    """
    if "close" not in df.columns:
        raise ValueError("El DataFrame de BTC debe incluir la columna 'close'.")

    out = df.copy()
    index = pd.to_datetime(out.index)
    out["days"] = (index - pd.Timestamp(genesis)).days

    # Filtra días no positivos (pre-genesis) y precios no positivos.
    out = out[(out["days"] > 0) & (out["close"] > 0)].copy()

    out["log_days"] = np.log10(out["days"])
    out["log_price"] = np.log10(out["close"])
    return out


def fit_power_law_quantiles(
    df: pd.DataFrame,
    quantiles: Sequence[float] = BTC_QUANTILES,
) -> dict[float, dict[str, Any]]:
    """Ajusta la regresión cuantílica ``log_price ~ log_days``.

    Parameters
    ----------
    df : pd.DataFrame
        Frame OHLCV de BTC o frame ya preparado con ``prepare_power_law_frame``.
    quantiles : Sequence[float]
        Cuantililes a ajustar (por defecto 0.10, 0.50 y 0.90).

    Returns
    -------
    dict
        ``{cuantil: {"params": pd.Series, "prediction": pd.Series}}`` con los
        parámetros de la regresión y la predicción en escala lineal.
    """
    frame = df
    if "log_days" not in df.columns or "log_price" not in df.columns:
        frame = prepare_power_law_frame(df)

    if frame.empty:
        raise ValueError("No hay datos suficientes para ajustar la regresión.")

    mod = smf.quantreg("log_price ~ log_days", frame)

    results: dict[float, dict[str, Any]] = {}
    for q in quantiles:
        res = mod.fit(q=q)
        results[q] = {
            "params": res.params,
            "prediction": 10 ** res.predict(frame[["log_days"]]),
        }
    return results


def detrend_power_law_quantile(
    df: pd.DataFrame,
    fit: dict[float, dict[str, Any]] | None = None,
    quantile: float = BTC_CYCLE_QUANTILE,
) -> pd.DataFrame:
    """Resta la banda de un cuantil del power-law al precio de BTC.

    La banda ajustada es la recta ``log10(band) = α + β·log10(days)`` del
    cuantil elegido (0,10 por defecto: el "suelo" del power-law).  Al
    restarla, el precio queda en escala logarítmica y sus **mínimos locales
    quedan cerca de cero**: los valles que tocan la banda inferior se
    materializan alrededor de 0.

    Columnas añadidas al frame de entrada:

    - ``band``: precio de la banda ajustada (misma unidad que ``close``);
    - ``log_band``: ``log10(band)``;
    - ``resid_log``: ``log_price - log_band`` (residuo logarítmico);
    - ``ratio``: ``close / band`` (precio relativo a la banda; equivale a
      ``10 ** resid_log``).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV diario de BTC o frame ya preparado con ``prepare_power_law_frame``.
    fit : dict[float, dict[str, Any]] | None
        Ajuste de ``fit_power_law_quantiles``.  Si es ``None`` se recalcula
        para el cuantil pedido.
    quantile : float
        Cuantil de la banda que se resta (0,10 por defecto).

    Returns
    -------
    pd.DataFrame
        Copia del frame con ``band``, ``log_band``, ``resid_log`` y ``ratio``.

    Raises
    ------
    ValueError
        Si el cuantil no está en ``fit`` o si la banda no cubre el índice
        del frame (ajustes incompatibles).
    """
    frame = df
    if "log_days" not in df.columns or "log_price" not in df.columns:
        frame = prepare_power_law_frame(df)

    if fit is None:
        fit = fit_power_law_quantiles(frame, quantiles=[quantile])
    if quantile not in fit:
        available = sorted(float(q) for q in fit)
        raise ValueError(
            f"El cuantil {quantile} no está en el ajuste; disponibles: {available}."
        )

    prediction = fit[quantile]["prediction"].astype(float)
    band = prediction.reindex(frame.index)
    if band.isna().any():
        raise ValueError(
            "La banda ajustada no cubre el índice del frame "
            "(¿se ajustó la regresión sobre otra serie?)."
        )

    out = frame.copy()
    out["band"] = band
    out["log_band"] = np.log10(out["band"])
    out["resid_log"] = out["log_price"] - out["log_band"]
    out["ratio"] = out["close"] / out["band"]
    return out


def smooth_cycle_residual(
    resid: pd.Series,
    window: int = BTC_CYCLE_SMOOTH_WINDOW,
    min_periods: int | None = None,
    centered: bool = False,
) -> pd.Series:
    """Promedio móvil (ventana fija) del residuo logarítmico del ciclo.

    Suaviza el precio dettrendizado con una media móvil **trailing** (sólo
    datos hasta ``t``): los primeros ``window - 1`` valores quedan en ``NaN``
    porque la ventana aún no está completa.  La serie suavizada es la que se
    usa después para detectar los extremos locales y para construir el ciclo
    idealizado, de modo que el ruido diario no genere candidatos espurios.

    Con ``centered=True`` la ventana se **centra** en cada fecha: la media
    usa también datos posteriores, de modo que los giros de la serie
    suavizada coinciden con los del precio y no aparecen desplazados.  Es un
    uso descriptivo (alinear los extremos con el ciclo real), no una señal
    causal: con la ventana completa quedan ``NaN`` en los bordes, del orden de
    ``window // 2`` al principio y al final.

    Parameters
    ----------
    resid : pd.Series
        Residuo logarítmico (``resid_log``) indexado por fecha.
    window : int
        Tamaño de la ventana en días (por defecto ``30``).
    min_periods : int | None
        Mínimo de observaciones por ventana.  Por defecto es ``window``
        (ventana completa); con un valor menor la media usa ventanas
        parciales en los bordes de la serie.
    centered : bool
        Si es ``True``, la ventana se centra en cada fecha en lugar de
        terminar en ella (media móvil centrada en lugar de *trailing*).

    Returns
    -------
    pd.Series
        Media móvil del residuo, en el índice de ``resid``.

    Raises
    ------
    ValueError
        Si ``window`` o ``min_periods`` no son positivos.
    """
    win = int(window)
    if win < 1:
        raise ValueError(f"La ventana debe ser >= 1, se recibió {window}.")

    mp = win if min_periods is None else int(min_periods)
    if mp < 1:
        raise ValueError(f"`min_periods` debe ser >= 1, se recibió {min_periods}.")

    series = resid.astype(float)
    smoothed = series.rolling(window=win, min_periods=mp, center=bool(centered)).mean()
    smoothed.name = "resid_ma"
    return smoothed


def find_local_extrema(
    series: pd.Series,
    distance: int = BTC_CYCLE_DISTANCE,
    prominence: float = BTC_CYCLE_PROMINENCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detecta mínimos y máximos locales candidatos de una serie.

    Usa ``scipy.signal.find_peaks`` sobre la serie (máximos) y sobre su
    negación (mínimos), con una distancia mínima en muestras y una
    prominencia mínima.  Los candidatos se numeran por orden cronológico
    (``B1``, ``B2``, ... para los valles; ``P1``, ``P2``, ... para los
    picos) para que el autor pueda elegir los que interesen.

    Los valores ``NaN`` (por ejemplo, el comienzo de una serie suavizada con
    ventana) se descartan antes de la detección: no son candidatos ni
    desplazan la numeración.

    Parameters
    ----------
    series : pd.Series
        Serie a analizar (por ejemplo, el residuo logarítmico).
    distance : int
        Distancia mínima entre candidatos, en muestras (días).
    prominence : float
        Prominencia mínima, en las unidades de la serie.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(bottoms, peaks)`` indexados por fecha, con las columnas ``id``,
        ``value`` y ``prominence``.
    """
    clean = series.dropna()
    values = clean.to_numpy(dtype=float)

    def _candidates(
        indices: np.ndarray,
        props: dict[str, Any],
        prefix: str,
    ) -> pd.DataFrame:
        sub = clean.iloc[indices]
        out = pd.DataFrame(
            {
                "value": sub.to_numpy(dtype=float),
                "prominence": np.asarray(props["prominences"], dtype=float),
            },
            index=sub.index,
        )
        out.index.name = series.index.name
        out = out.sort_index()
        out.insert(0, "id", [f"{prefix}{i}" for i in range(1, len(out) + 1)])
        return out

    min_idx, min_props = find_peaks(
        -values, distance=distance, prominence=prominence
    )
    max_idx, max_props = find_peaks(
        values, distance=distance, prominence=prominence
    )

    return _candidates(min_idx, min_props, "B"), _candidates(max_idx, max_props, "P")


def select_extrema(
    candidates: pd.DataFrame,
    selection: Sequence[Any],
) -> pd.DataFrame:
    """Resuelve la selección del autor sobre una tabla de candidatos.

    Cada elemento de ``selection`` puede ser:

    - el ``id`` del candidato (``"B3"``, ``"P7"``);
    - una fecha (texto ISO ``"2015-01-14"`` o ``pandas.Timestamp``);
    - el número de orden (1-based) tal como aparece en la tabla impresa.

    Los candidatos resueltos se devuelven sin duplicados y ordenados por
    fecha.

    Parameters
    ----------
    candidates : pd.DataFrame
        Tabla de candidatos con columna ``id`` e índice de fechas.
    selection : Sequence[Any]
        Selección del autor (ids, fechas o números de orden).

    Returns
    -------
    pd.DataFrame
        Subconjunto de ``candidates`` con la selección resuelta.

    Raises
    ------
    ValueError
        Si algún elemento no corresponde a un candidato.
    """
    if candidates.empty:
        return candidates.copy()

    ids = candidates["id"].astype(str)
    positions_by_id = {rid: pos for pos, rid in enumerate(ids)}

    chosen: list[int] = []
    for item in selection:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            pos = int(item) - 1
            if not 0 <= pos < len(candidates):
                raise ValueError(
                    f"Selección {item} fuera de rango (1..{len(candidates)})."
                )
        else:
            key = str(item).strip()
            if key in positions_by_id:
                pos = positions_by_id[key]
            else:
                try:
                    stamp = pd.Timestamp(key)
                except ValueError as exc:
                    disponibles = list(positions_by_id)
                    vista = ", ".join(disponibles[:12])
                    if len(disponibles) > 12:
                        vista += ", ..."
                    raise ValueError(
                        f"'{key}' no es un id válido ni una fecha. "
                        f"Ids disponibles: {vista}."
                    ) from exc
                matches = np.flatnonzero(candidates.index == stamp)
                if len(matches) == 0:
                    raise ValueError(
                        f"'{key}' no está en la tabla de candidatos "
                        f"({len(candidates)} candidatos disponibles)."
                    )
                pos = int(matches[0])
        if pos not in chosen:
            chosen.append(pos)

    return candidates.iloc[sorted(chosen)].copy()


_NS_PER_DAY: float = 86_400_000_000_000.0


def _to_days_since_epoch(
    index: pd.DatetimeIndex,
    origin: float | None = None,
) -> np.ndarray:
    """Convierte fechas a días (float) desde una época de referencia.

    Normaliza a nanoseconds porque ``pandas`` 3 usa microsegundos en
    ``date_range`` y nanosegundos en ``Timestamp.value``: mezclar ambas
    escalas rompería la interpolación.
    """
    nanos = index.to_numpy(dtype="datetime64[ns]").astype("float64")
    if origin is None:
        origin = 0.0
    return (nanos - origin) / _NS_PER_DAY


def idealized_cycle(
    resid: pd.Series,
    bottoms: pd.DataFrame,
    peaks: pd.DataFrame,
) -> pd.Series:
    """Construye el ciclo idealizado a partir de los extremos elegidos.

    La poligonal une los extremos seleccionados **en orden cronológico**:
    cada valle (mínimo local elegido) queda anclado en ``0`` y cada pico
    queda a la **amplitud de su ciclo**, es decir, su residuo menos el
    residuo del valle elegido que lo precede.  Entre vértices la interpolación
    es lineal y fuera del rango cubierto la serie vale ``NaN``.

    Parameters
    ----------
    resid : pd.Series
        Residuo logarítmico (o cualquier serie) indexada por fecha.
    bottoms : pd.DataFrame
        Tabla de mínimos elegidos (con índice de fechas).
    peaks : pd.DataFrame
        Tabla de máximos elegidos (con índice de fechas).

    Returns
    -------
    pd.Series
        Ciclo idealizado en el índice de ``resid``.

    Raises
    ------
    TypeError
        Si el residuo no está indexado por fechas.
    ValueError
        Si se seleccionan menos de dos extremos.
    """
    series = resid.astype(float)
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("El residuo debe estar indexado por fechas.")

    events: list[tuple[pd.Timestamp, str]] = []
    events.extend((pd.Timestamp(ts), "B") for ts in bottoms.index)
    events.extend((pd.Timestamp(ts), "P") for ts in peaks.index)
    events.sort(key=lambda item: item[0])

    if len(events) < 2:
        raise ValueError(
            "Se necesitan al menos dos extremos seleccionados para ajustar "
            "el ciclo idealizado."
        )

    floor = 0.0
    vertices: dict[pd.Timestamp, float] = {}
    for stamp, kind in events:
        value = float(series.loc[stamp]) if stamp in series.index else 0.0
        if kind == "B":
            floor = value
            vertices[stamp] = 0.0
        else:
            vertices[stamp] = value - floor

    ordered = sorted(vertices)
    stamps = pd.DatetimeIndex(ordered)
    origin = stamps.to_numpy(dtype="datetime64[ns]").astype("float64")[0]
    x = _to_days_since_epoch(stamps, origin)
    y = np.array([vertices[stamp] for stamp in ordered], dtype=float)

    full_x = _to_days_since_epoch(series.index, origin)
    interpolated = np.interp(full_x, x, y, left=np.nan, right=np.nan)
    return pd.Series(interpolated, index=series.index, name="idealized")


def plot_btc_quantreg(
    df: pd.DataFrame,
    halvings: Sequence[Any] | None = None,
    peaks: Sequence[Any] | None = None,
    bottoms: Sequence[Any] | None = None,
    ax: Sequence[plt.Axes] | None = None,
    quantiles: Sequence[float] = BTC_QUANTILES,
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Dibuja el precio de BTC con las rectas de la regresión quantílica.

    Devuelve tres gráficos independientes: la evolución temporal del precio
    en escala lineal, el eje temporal en escala logarítmica y los días desde
    la génesis en escala log-log.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV diario de BTC.
    halvings, peaks, bottoms : Sequence[Any] | None
        Fechas anotadas en el gráfico.  Por defecto se usan
        ``BTC_HALVINGS``, ``BTC_PEAKS`` y ``BTC_BOTTOMS``.
    ax : Sequence[plt.Axes] | None
        Tres ejes donde dibujar; si es ``None`` se crean tres figuras nuevas.
    quantiles : Sequence[float]
        Cuantililes del power-law que se ajustan y se dibujan (por defecto
        ``BTC_QUANTILES``).

    Returns
    -------
    tuple[plt.Figure, plt.Figure, plt.Figure]
        Las figuras lineal, temporal logarítmica y power-law, respectivamente.
    """
    reg_df = prepare_power_law_frame(df)
    fit = fit_power_law_quantiles(reg_df, quantiles=quantiles)

    halving_dates = _as_dates(BTC_HALVINGS if halvings is None else halvings)
    peak_dates = _as_dates(BTC_PEAKS if peaks is None else peaks)
    bottom_dates = _as_dates(BTC_BOTTOMS if bottoms is None else bottoms)

    if ax is None:
        linear_fig, ax0 = plt.subplots(figsize=(12, 5))
        temporal_fig, ax1 = plt.subplots(figsize=(12, 5))
        power_law_fig, ax2 = plt.subplots(figsize=(12, 5))
    else:
        axes = list(ax)
        if len(axes) != 3:
            raise ValueError("`ax` debe contener exactamente tres ejes.")
        ax0, ax1, ax2 = axes
        linear_fig = ax0.get_figure()
        temporal_fig = ax1.get_figure()
        power_law_fig = ax2.get_figure()

    # --- Evolución temporal en escala lineal ---
    ax0.plot(
        reg_df.index,
        reg_df["close"],
        label="Cierre de BTC",
        color="black",
        alpha=0.7,
        linewidth=1,
    )
    ax0.set_title("Evolución del precio de Bitcoin")
    ax0.set_xlabel("Fecha")
    ax0.set_ylabel("Precio (USD)")
    ax0.legend()
    ax0.grid(True, ls="--", alpha=0.4)

    # --- Gráfico temporal ---
    ax1.plot(
        reg_df.index,
        reg_df["close"],
        label="Cierre de BTC",
        color="black",
        alpha=0.6,
        linewidth=1,
    )
    for q, res in fit.items():
        ax1.plot(
            reg_df.index,
            res["prediction"],
            label=f"Reg. percentil {int(q * 100)}",
            color=_QUANTILE_COLORS.get(q, "gray"),
            linestyle=_QUANTILE_STYLES.get(q, "--"),
            linewidth=2,
        )

    for i, halving_date in enumerate(halving_dates):
        ax1.axvline(
            halving_date,
            color="purple",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
            label="Halving" if i == 0 else None,
        )

    for i, peak_date in enumerate(peak_dates):
        ax1.axvline(
            peak_date,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label="Pico" if i == 0 else None,
        )

    for i, bottom_date in enumerate(bottom_dates):
        ax1.axvline(
            bottom_date,
            color="red",
            linestyle="-.",
            alpha=0.5,
            linewidth=1.5,
            label="Valle" if i == 0 else None,
        )

    ax1.set_yscale("log")
    ax1.set_title("Dinámica del precio de BTC — regresión cuantílica power-law")
    ax1.set_xlabel("Días desde la génesis")
    ax1.set_ylabel("Precio (USD)")
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # --- Gráfico power-law ---
    ax2.plot(
        reg_df["days"],
        reg_df["close"],
        label="Cierre de BTC",
        color="black",
        alpha=0.6,
        linewidth=1,
    )
    for q, res in fit.items():
        ax2.plot(
            reg_df["days"],
            res["prediction"],
            label=f"Reg. percentil {int(q * 100)}",
            color=_QUANTILE_COLORS.get(q, "gray"),
            linestyle=_QUANTILE_STYLES.get(q, "--"),
            linewidth=2,
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Dinámica del precio de BTC — regresión cuantílica power-law")
    ax2.set_xlabel("Días desde la génesis")
    ax2.set_ylabel("Precio (USD)")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    linear_fig.tight_layout()
    temporal_fig.tight_layout()
    power_law_fig.tight_layout()
    return linear_fig, temporal_fig, power_law_fig


def _print_cycles_summary(
    peaks: pd.DataFrame,
    bottoms: pd.DataFrame,
) -> None:
    """Imprime el resumen de picos y valles macro detectados."""
    print("=== PICOS MACRO HISTÓRICOS ===")
    for date, row in peaks.iterrows():
        dist = (
            f"{int(row['Days_Since_Last_Peak'])} días"
            if pd.notna(row["Days_Since_Last_Peak"])
            else "N/A"
        )
        print(
            f"Fecha: {date.strftime('%Y-%m-%d')} | "
            f"Precio: ${row['close']:>9.2f} | Distancia al previo: {dist}"
        )

    peak_mean = peaks["Days_Since_Last_Peak"].mean()
    peak_std = peaks["Days_Since_Last_Peak"].std()
    print(
        f"\nPico a pico -> Media: {peak_mean:.1f} días | Desv. típica: {peak_std:.1f} días\n"
    )

    print("=== VALLES MACRO HISTÓRICOS ===")
    for date, row in bottoms.iterrows():
        dist = (
            f"{int(row['Days_Since_Last_Bottom'])} días"
            if pd.notna(row["Days_Since_Last_Bottom"])
            else "N/A"
        )
        print(
            f"Fecha: {date.strftime('%Y-%m-%d')} | "
            f"Precio: ${row['close']:>9.2f} | Distancia al previo: {dist}"
        )

    bottom_mean = bottoms["Days_Since_Last_Bottom"].mean()
    bottom_std = bottoms["Days_Since_Last_Bottom"].std()
    print(
        f"\nValle a valle -> Media: {bottom_mean:.1f} días | "
        f"Desv. típica: {bottom_std:.1f} días"
    )


def detect_market_cycles(
    df: pd.DataFrame,
    distance: int = 300,
    prominence: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detecta picos y valles macro sobre ``log10(close)``.

    Usa ``scipy.signal.find_peaks`` con una distancia mínima (en muestras) y
    una prominencia mínima.  El pico de Mt. Gox (2011-06-08), anterior a los
    datos de Bitstamp, se inyecta a mano.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV diario de BTC.
    distance : int
        Distancia mínima entre picos, en muestras (días).
    prominence : float
        Prominencia mínima en escala logarítmica.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(peaks, bottoms)`` con las columnas ``Days_Since_Last_Peak`` y
        ``Days_Since_Last_Bottom`` respectivamente.
    """
    if "close" not in df.columns:
        raise ValueError("El DataFrame de BTC debe incluir la columna 'close'.")

    frame = df.copy()
    frame.index = pd.to_datetime(frame.index)

    log_prices = np.log10(frame["close"])
    peak_idx, _ = find_peaks(log_prices, distance=distance, prominence=prominence)
    bottom_idx, _ = find_peaks(-log_prices, distance=distance, prominence=prominence)

    peaks = frame.iloc[peak_idx].copy()
    bottoms = frame.iloc[bottom_idx].copy()

    mtgox_peak = pd.DataFrame(
        {"close": [MTGOX_PEAK_CLOSE]},
        index=pd.to_datetime([MTGOX_PEAK_DATE]),
    )
    peaks = pd.concat([mtgox_peak, peaks]).sort_index()

    peaks["Days_Since_Last_Peak"] = peaks.index.to_series().diff().dt.days
    bottoms["Days_Since_Last_Bottom"] = bottoms.index.to_series().diff().dt.days

    _print_cycles_summary(peaks, bottoms)

    return peaks, bottoms
