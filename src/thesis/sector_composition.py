"""Composición sectorial del S&P 500 y del ETF SPY.

El módulo concentra la lógica reutilizable del cuaderno
``notebooks/spy/SPY_sectores.py``:

1. **snapshot** de los pesos sectoriales actuales de un ETF (yfinance);
2. **reconstrucción histórica** aproximada de la composición sectorial del
   S&P 500 por capitalización de mercado (Wikipedia → constituyentes;
   yfinance → precios y acciones; ponderación por sector);
3. **entropía de Shannon** de los pesos sectoriales a lo largo del tiempo;
4. **HHI a nivel de empresa**: concentración de la capitalización del S&P 500
   en pocas compañías a lo largo del tiempo.

Importar el módulo **no** descarga datos ni dibuja figuras: los efectos con
red, disco o pantalla viven dentro de funciones explícitas.  El módulo no se
exporta en ``thesis/__init__.py`` para no arrastrar ``lxml``/``requests`` en
cada import del paquete; se usa ``from thesis.sector_composition import ...``.

.. warning::
   La reconstrucción histórica es una **aproximación** con sesgo de
   supervivencia y clasificación sectorial retroactiva: se usan la lista de
   constituyentes y las etiquetas GICS **actuales**, y las acciones en
   circulación se mantienen constantes en el tiempo.  Es una herramienta
   descriptiva de la rotación sectorial, no una reconstrucción point-in-time.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

#: Traducción al español de los sectores GICS que usa yfinance.
GICS_SECTOR_ES: dict[str, str] = {
    "technology": "Tecnología",
    "financial_services": "Servicios Financieros",
    "healthcare": "Salud",
    "consumer_cyclical": "Consumo Cíclico",
    "industrials": "Industriales",
    "communication_services": "Servicios de Comunicación",
    "consumer_defensive": "Consumo Defensivo",
    "energy": "Energía",
    "realestate": "Bienes Raíces",
    "utilities": "Servicios Públicos",
    "basic_materials": "Materiales Básicos",
}

#: Página de Wikipedia con la lista de constituyentes del S&P 500.
SP500_WIKI_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

#: Peso (en fracción, no en %) por debajo del cual un sector se agrupa en
#: "Otros".  El valor original del cuaderno-era 0,08 en el código y "9%" en
#: el comentario: aquí se unifica en 0,08.
MINOR_SECTOR_THRESHOLD: float = 0.08

#: Nombre del CSV de caché con los pesos sectoriales históricos.
SECTOR_COMPOSITION_CACHE_FILENAME: str = "sp500_sector_composition.csv"

#: Plantilla del nombre del CSV de caché con el **frame de precios**
#: histórico (ancho, fecha × ticker).  La clave de caché es la ventana
#: ``(start, end)``: ``end`` debe ser **explícito**, porque con ``end=None``
#: ("hasta hoy") no se puede congelar la ventana en el nombre del archivo.
PRICE_CACHE_FILENAME_TMPL: str = "sp500_prices_{start}_{end}.csv"

#: Nombre del CSV de caché con las acciones en circulación.
SHARES_CACHE_FILENAME: str = "sp500_shares.csv"

#: Etiqueta del grupo que agrupa los sectores por debajo del umbral.
OTHERS_LABEL: str = "Otros"

#: User-Agent enviado a Wikipedia (evita el bloqueo de la API pública).
_WIKIPEDIA_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; thesis-sector-composition/1.0)"
}


def aggregate_small_sectors(
    weights: pd.Series,
    threshold: float = MINOR_SECTOR_THRESHOLD,
    translation: Mapping[str, str] = GICS_SECTOR_ES,
) -> pd.Series:
    """Agrupa en "Otros" los sectores con peso menor que ``threshold``.

    Los pesos se esperan en **fracción** (0,30 = 30 %), como los devuelve
    ``yf.Ticker(...).funds_data``.  Los sectores con peso ``< threshold`` se
    suman en una única entrada "Otros"; el resto se traducen con
    ``translation`` (si no hay traducción, se usa el nombre legible del
    sector).  El resultado queda **ordenado de mayor a menor**.

    Los pesos ``NaN`` se descartan y los pesos exactamente nulos no generan
    una entrada "Otros" vacía.

    Parameters
    ----------
    weights : pd.Series
        Pesos sectoriales en fracción, indexados por el nombre del sector.
    threshold : float
        Umbral por debajo del cual un sector se agrupa en "Otros"
        (por defecto ``0.08``).
    translation : Mapping[str, str]
        Traducción de los nombres de sector al español.

    Returns
    -------
    pd.Series
        Pesos por sector (traducidos, con "Otros" incluido si aplica),
        ordenados de mayor a menor y con la misma suma que la entrada.
    """
    if weights.empty:
        return weights.copy()

    series = weights.astype(float).dropna()
    small = series[series < threshold]
    large = series[series >= threshold]

    labels = {
        sector: translation.get(sector, str(sector).replace("_", " ").title())
        for sector in large.index
    }
    out = large.rename(index=labels)

    others = float(small.sum()) if not small.empty else 0.0
    if others > 0:
        out = pd.concat([out, pd.Series({OTHERS_LABEL: others})])

    out = out.sort_values(ascending=False, kind="stable")
    out.name = "weight"
    return out


def fetch_spy_sector_weights(ticker: str = "SPY") -> pd.Series:
    """Descarga los pesos sectoriales actuales de un ETF.

    Usa ``yf.Ticker(ticker).funds_data.sector_weightings``, que devuelve un
    diccionario ``sector -> peso en fracción``.

    Parameters
    ----------
    ticker : str
        Símbolo del ETF (por defecto ``"SPY"``).

    Returns
    -------
    pd.Series
        Pesos sectoriales en fracción, ordenados de mayor a menor.

    Raises
    ------
    ValueError
        Si yfinance no devuelve la composición sectorial del ETF.
    """
    data = yf.Ticker(ticker).funds_data
    raw = getattr(data, "sector_weightings", None) if data is not None else None
    if not raw:
        raise ValueError(
            f"yfinance no devolvió la composición sectorial de {ticker!r}."
        )

    weights = pd.Series(
        {str(sector): float(value) for sector, value in raw.items()},
        dtype=float,
        name="weight",
    )
    return weights.sort_values(ascending=False, kind="stable")


def get_sp500_constituents(
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Descarga la lista actual de constituyentes del S&P 500 de Wikipedia.

    Devuelve las columnas ``Symbol`` (con los puntos sustituidos por
    guiones: ``BRK.B`` → ``BRK-B``), ``Security`` y ``Sector`` (la columna
    "GICS Sector" de Wikipedia).

    Parameters
    ----------
    session : requests.Session | None
        Sesión HTTP inyectable (para tests).  Si es ``None`` se crea una
        sesión nueva.

    Returns
    -------
    pd.DataFrame
        Constituyentes con las columnas ``Symbol``, ``Security`` y
        ``Sector``.

    Raises
    ------
    ValueError
        Si Wikipedia no devuelve ninguna tabla o le faltan columnas.
    """
    sess = session if session is not None else requests.Session()
    response = sess.get(SP500_WIKI_URL, headers=_WIKIPEDIA_HEADERS, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError("Wikipedia no devolvió ninguna tabla de constituyentes.")

    table = tables[0]
    required = {"Symbol", "Security", "GICS Sector"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en la tabla de Wikipedia: {sorted(missing)}."
        )

    out = table[["Symbol", "Security", "GICS Sector"]].copy()
    out["Symbol"] = out["Symbol"].astype(str).str.replace(".", "-", regex=False)
    out = out.rename(columns={"GICS Sector": "Sector"})
    return out.reset_index(drop=True)


def _extract_close(
    data: pd.DataFrame | pd.Series | None,
    batch: Sequence[str],
) -> pd.DataFrame | None:
    """Extrae la columna de cierres de la respuesta de ``yf.download``."""
    if data is None or len(data) == 0:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return None
        close = data["Close"]
    elif "Close" in data.columns:
        # Con un único ticker, yfinance devuelve las columnas planas y no
        # incluye el nombre del símbolo: se restituye desde el lote.
        close = data[["Close"]].copy()
        close.columns = [str(batch[0])] * close.shape[1]
    else:
        return None

    if isinstance(close, pd.Series):
        close = close.to_frame(batch[0])

    close = close.copy()
    close.columns = [str(column) for column in close.columns]
    return close


def price_cache_path(cache_dir: Path | str, start: str, end: str) -> Path:
    """Ruta del CSV de caché de precios para la ventana ``start:end``.

    Parameters
    ----------
    cache_dir : Path | str
        Directorio donde se guarda (o se busca) el CSV.
    start : str
        Fecha inicial de la ventana (ISO).
    end : str
        Fecha final de la ventana (ISO), **explícita**.

    Returns
    -------
    Path
        Ruta del CSV ``sp500_prices_<start>_<end>.csv``.
    """
    return Path(cache_dir) / PRICE_CACHE_FILENAME_TMPL.format(start=start, end=end)


def download_price_history(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
    batch_size: int = 100,
    sleep: float = 1.0,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Descarga el cierre ajustado de varios tickers por lotes.

    Las descargas se hacen en lotes de ``batch_size`` tickers (yfinance
    limita el tamaño de la petición) con una pausa de ``sleep`` segundos
    entre lotes.

    Si se pasa ``cache_dir`` **y** ``end`` es explícito, el frame-wide
    (fecha × ticker) se cachea en un único CSV
    (``sp500_prices_<start>_<end>.csv``): en las siguientes llamadas el CSV
    se lee y **no** se toca la red.  La caché sólo se considera válida si
    contiene todos los tickers pedidos (si el universo de constituyentes ha
    cambiado, se vuelve a descargar).  Con ``end=None`` ("hasta hoy") no se
    cachea nada: la ventana no se puede congelar en el nombre del archivo.

    Parameters
    ----------
    tickers : Sequence[str]
        Símbolos a descargar.
    start : str
        Fecha inicial (ISO).
    end : str | None
        Fecha final (ISO).  ``None`` significa "hasta hoy" (y desactiva la
        caché).
    batch_size : int
        Número de tickers por petición.
    sleep : float
        Segundos de pausa entre lotes.
    cache_dir : Path | None
        Directorio del CSV de caché.  ``None`` desactiva la caché.
    refresh : bool
        Si es ``True``, se ignora (y se sobrescribe) la caché existente.

    Returns
    -------
    pd.DataFrame
        Cierres ajustados en formato ancho (fecha × ticker), ordenados por
        fecha y sin columnas duplicadas.

    Raises
    ------
    ValueError
        Si ``batch_size`` no es positivo.
    """
    if batch_size < 1:
        raise ValueError(f"`batch_size` debe ser >= 1, se recibió {batch_size}.")

    # Se eliminan los duplicados (mismo ticker pedido dos veces) para que el
    # frame ancho y su caché no tengan columnas repetidas.
    symbols = list(dict.fromkeys(str(ticker) for ticker in tickers))
    if not symbols:
        return pd.DataFrame()

    cache_path: Path | None = None
    if cache_dir is not None and end is not None:
        cache_path = price_cache_path(cache_dir, start, end)
        if cache_path.exists() and not refresh:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if set(symbols).issubset(cached.columns):
                # La caché cubre el universo pedido: se devuelve sin red.
                return cached.loc[:, symbols].sort_index()
            # La caché es de otro universo de tickers: se vuelve a descargar.

    frames: list[pd.DataFrame] = []
    for start_idx in range(0, len(symbols), batch_size):
        batch = symbols[start_idx : start_idx + batch_size]
        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = _extract_close(data, batch)
        if close is not None and not close.empty:
            frames.append(close)
        if sleep > 0 and start_idx + batch_size < len(symbols):
            time.sleep(sleep)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Se guardan *todas* las columnas pedidas (las que no se pudieron
        # descargar quedan en NaN) para que la caché siempre cubra el
        # universo completo y no se re-descargue en cada ejecución.
        prices.reindex(columns=symbols).to_csv(cache_path)

    return prices


def get_shares_outstanding(
    tickers: Sequence[str],
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Descarga las acciones en circulación actuales de cada ticker.

    Es una consulta *best-effort*: los tickers que fallan quedan con
    ``NaN`` y se descartan al ponderar.

    Si se pasa ``cache_dir``, el resultado se cachea en
    ``sp500_shares.csv`` y las siguientes llamadas **no** consultan la red.
    La caché congela las acciones tal y como se observaron en la última
    descarga (el método ya las mantiene constantes en el tiempo): usa
    ``refresh=True`` para actualizarlas.

    Parameters
    ----------
    tickers : Sequence[str]
        Símbolos a consultar.
    cache_dir : Path | None
        Directorio del CSV de caché.  ``None`` desactiva la caché.
    refresh : bool
        Si es ``True``, se ignora (y se sobrescribe) la caché existente.

    Returns
    -------
    pd.Series
        Acciones en circulación por ticker, ordenadas alfabéticamente.
    """
    symbols = [str(ticker) for ticker in tickers]

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / SHARES_CACHE_FILENAME
        if cache_path.exists() and not refresh:
            cached = pd.read_csv(cache_path, index_col=0).iloc[:, 0]
            cached.index = cached.index.astype(str)
            cached = pd.to_numeric(cached, errors="coerce")
            out = cached.reindex(symbols).astype(float)
            out.name = "SharesOutstanding"
            out.index.name = None
            return out.sort_index()

    shares: dict[str, float] = {}
    for ticker in symbols:
        symbol = str(ticker)
        value: float = np.nan
        try:
            info = yf.Ticker(symbol).fast_info
            raw = info.get("shares", np.nan)
        except Exception:  # noqa: BLE001 - la consulta es best-effort
            raw = np.nan
        if raw is not None:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = np.nan
        shares[symbol] = value

    out = pd.Series(shares, dtype=float, name="SharesOutstanding").sort_index()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_frame().rename_axis("Ticker").to_csv(cache_path)

    return out


def _market_cap(prices: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """Capitalización de mercado aproximada (precio · acciones).

    Aplica la validación común a :func:`build_sector_weights` y
    :func:`company_hhi`: se descartan los tickers sin precio **o** sin
    acciones (columnas enteramente nulas) y se rechaza cualquier
    capitalización negativa (precios o acciones corruptos).

    Parameters
    ----------
    prices : pd.DataFrame
        Cierres en formato ancho (fecha × ticker).
    shares : pd.Series
        Acciones en circulación por ticker.

    Returns
    -------
    pd.DataFrame
        Capitalización de mercado por fecha y ticker.

    Raises
    ------
    ValueError
        Si ``prices`` no tiene columnas, si ningún ticker tiene precio y
        acciones en común o si hay capitalizaciones negativas.
    """
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("`prices` no contiene ninguna columna de ticker.")

    market_cap = prices.mul(shares.reindex(prices.columns), axis=1)
    market_cap = market_cap.dropna(axis=1, how="all")
    if market_cap.empty:
        raise ValueError(
            "Ningún ticker tiene precio y acciones en común: no hay datos "
            "para ponderar."
        )

    if (market_cap.to_numpy(dtype=float) < 0).any():
        raise ValueError(
            "Se han obtenido pesos negativos "
            "(precios o acciones corruptos)."
        )
    return market_cap


def build_sector_weights(
    prices: pd.DataFrame,
    shares: pd.Series,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """Calcula el peso porcentual de cada sector por fecha.

    La capitalización de mercado se aproxima como
    ``precio(t) · acciones_en_circulación`` (las acciones son las
    **actuales**, constantes en el tiempo).  Los tickers sin datos (precio o
    acciones) se descartan, y el resultado son porcentajes (0-100) que suman
    100 en cada fecha.

    Parameters
    ----------
    prices : pd.DataFrame
        Cierres en formato ancho (fecha × ticker).
    shares : pd.Series
        Acciones en circulación por ticker.
    sector_map : pd.Series
        Sector de cada ticker, indexado por el símbolo.

    Returns
    -------
    pd.DataFrame
        Pesos sectoriales en porcentaje (0-100), con una columna por
        sector.

    Raises
    ------
    ValueError
        Si no hay tickers con datos, si ``sector_map`` no cubre los tickers
        usados o si los pesos resultantes son negativos.
    """
    market_cap = _market_cap(prices, shares)

    sectors = sector_map.reindex(market_cap.columns)
    if sectors.isna().any():
        missing = [str(symbol) for symbol in sectors.index[sectors.isna()]]
        raise ValueError(
            f"`sector_map` no cubre los tickers con datos: {missing[:10]}."
        )

    sector_caps = market_cap.T.groupby(sectors.to_numpy()).sum().T

    # Fechas sin capitalización (p. ej. antes del primer dato útil) se
    # descartan: en ellas la normalización no está definida.
    totals = sector_caps.sum(axis=1)
    valid = totals > 0
    sector_caps = sector_caps.loc[valid]
    totals = totals.loc[valid]

    weights = sector_caps.div(totals, axis=0) * 100.0
    if (weights.to_numpy(dtype=float) < 0).any():
        raise ValueError("Se han obtenido pesos sectoriales negativos.")
    return weights


def company_hhi(prices: pd.DataFrame, shares: pd.Series) -> pd.Series:
    """Índice de Herfindahl-Hirschman (HHI) a nivel de empresa.

    El HHI mide la **concentración** de la capitalización de mercado:

    ``HHI(t) = Σ_i w_i(t)²``,  con ``w_i(t) = cap_i(t) / Σ_j cap_j(t)``.

    Toma valores en ``[1/N, 1]``: vale ``1/N`` si las ``N`` empresas con
    datos pesan lo mismo y tiende a ``1`` cuando la capitalización se
    concentra en pocas compañías.  Es adimensional (fracción); algunos
    informes usan la convención equivalente ``× 10 000``.

    El cálculo es **puntual** (sólo depende de la fila): truncar la serie no
    altera las filas ya calculadas.  Los tickers sin precio o sin acciones
    se descartan, y las fechas sin capitalización total positiva se
    omiten (en ellas la normalización no está definida).

    Parameters
    ----------
    prices : pd.DataFrame
        Cierres en formato ancho (fecha × ticker).
    shares : pd.Series
        Acciones en circulación por ticker.

    Returns
    -------
    pd.Series
        HHI por fecha (nombre ``"hhi"``), en el índice de la entrada.

    Raises
    ------
    ValueError
        Si ``prices`` no tiene columnas, si ningún ticker tiene precio y
        acciones en común o si hay capitalizaciones negativas.
    """
    market_cap = _market_cap(prices, shares)

    # Fechas sin capitalización (p. ej. antes del primer dato útil) se
    # descartan: en ellas la normalización no está definida.
    totals = market_cap.sum(axis=1)
    valid = totals > 0
    market_cap = market_cap.loc[valid]
    totals = totals.loc[valid]

    weights = market_cap.div(totals, axis=0)
    hhi = (weights**2).sum(axis=1)
    hhi.name = "hhi"
    return hhi


def sector_entropy(sector_weights: pd.DataFrame) -> pd.Series:
    """Entropía de Shannon de los pesos sectoriales, fecha a fecha.

    Con ``p = pesos / 100`` (probabilidades), la entropía es
    ``-Σ p·ln(p)``.  El límite ``p·ln(p) → 0`` cuando ``p → 0`` se aplica de
    forma explícita, de modo que un sector con peso nulo no produce
    ``-inf``.

    El cálculo es **puntual** (sólo depende de la fila): truncar la serie no
    altera las filas ya calculadas.

    Parameters
    ----------
    sector_weights : pd.DataFrame
        Pesos sectoriales en porcentaje (0-100), con una columna por sector.

    Returns
    -------
    pd.Series
        Entropía (en nats) por fecha, en el índice de la entrada.
    """
    probabilities = sector_weights.to_numpy(dtype=float) / 100.0

    log_p = np.zeros_like(probabilities)
    np.log(probabilities, out=log_p, where=probabilities > 0)
    p_log_p = np.where(probabilities > 0, probabilities * log_p, 0.0)

    entropy = -p_log_p.sum(axis=1)
    return pd.Series(entropy, index=sector_weights.index, name="entropy")
