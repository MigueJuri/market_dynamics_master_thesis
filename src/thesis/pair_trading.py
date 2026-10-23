"""Pair trading utilities: spread construction, Ornstein-Uhlenbeck fitting,
Mutual Information matrices and process simulations shared across notebooks.

Este módulo concentra la lógica común a los cuadernos de ``pair_trading``
para evitar duplicación.  Se apoya en ``thesis.market_data`` para la
descarga cacheada de precios, en ``thesis.timeseries`` para retornos y en
``thesis.config`` para parámetros globales.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from statsmodels.tsa.stattools import adfuller

from thesis.config import (
    DATA_DIR,
    END,
    PAIR_SELECTION_MIN_HISTORY_OBS,
    SEED,
    START,
    TABLES_DIR,
)
from thesis.market_data import load_multiple_prices

WIKIPEDIA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


KSG_CACHE_VERSION = "v1"


# =============================================================================
# Universo y precios
# =============================================================================


def get_sp500_constituents() -> pd.DataFrame:
    """Scrapea la tabla completa de constituyentes del S&P 500.

    Descarga ``https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`` y
    devuelve un ``DataFrame`` con la composición actual del índice.  El nombre
    del valor (``Security``) se conserva para que ``filter_duplicate_issuers``
    pueda detectar emisores con varias clases de acción (ej. ``GOOGL``/
    ``GOOG``).

    Returns
    -------
    pd.DataFrame
        Columnas ``Symbol`` (normalizado para ``yfinance``: puntos
        reemplazados por guiones), ``Security`` (nombre del valor) y
        ``GICS Sector``.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=15)
    if response.status_code != 200:
        raise ConnectionError(f"HTTP Error: {response.status_code}")

    df = pd.read_html(StringIO(response.text))[0]
    df = df[["Symbol", "Security", "GICS Sector"]].copy()
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


def _issuer_key(security: str) -> str:
    """Normaliza el nombre de un valor para identificar emisores duplicados.

    Pasos:
    - minúsculas;
    - eliminación de sufijos ``Class A``, ``Class B`` y ``Cl A``/``Cl B``;
    - eliminación de sufijos corporativos (``Inc``, ``Corp``, ``Co``, etc.);
    - sustitución de puntuación por espacios;
    - recorte de espacios.

    El resultado es una clave canónica que agrupa las filas que comparten la
    raíz de la empresa: ``Alphabet Inc. Class A`` y ``Alphabet Inc. Class C``
    colapsan en ``alphabet``.  La agrupación depende de que esa raíz esté en
    ``Security``: dos símbolos del mismo emisor escritos sólo con el ticker
    (``GOOGL Class A`` y ``GOOG Class C``) producen claves distintas y **no**
    se agrupan.
    """
    if not isinstance(security, str):
        return ""
    name = security.lower()
    name = re.sub(
        r"\bclass\s+[a-z]\b|\bcl\s+[a-z]\b",
        "",
        name,
    )
    name = re.sub(
        r"\b(inc|corp|corporation|company|co|ltd|plc|sa|ag|n\.v\.|"
        r"holdings|group|plc|the)\b\.?",
        "",
        name,
    )
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def filter_duplicate_issuers(
    constituents: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra emisores con múltiples clases de acciones (ej. ``GOOGL``/``GOOG``).

    Pasos:
    1. Añade la columna ``Issuer`` usando ``_issuer_key`` sobre ``Security``.
    2. Para cada emisor con más de un símbolo, conserva el primero de la
       tabla oficial del S&P 500 (criterio determinista).
    3. Devuelve la tabla con los símbolos restantes junto con la columna
       ``Issuer`` y otra tabla con los símbolos descartados y su
       representante.

    En el S&P 500 actual (octubre 2026) se descartan:

    - ``GOOG`` (representante ``GOOGL``, Alphabet)
    - ``FOX``  (representante ``FOXA``, Fox Corporation)
    - ``NWS``  (representante ``NWSA``, News Corp)

    Parameters
    ----------
    constituents : pd.DataFrame
        Salida de :func:`get_sp500_constituents`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``kept``: constituyentes filtrados, una fila por emisor, con la
        columna extra ``Issuer``.
        ``removed``: símbolos descartados y su representante (una fila por
        símbolo).
    """
    if "Security" not in constituents.columns:
        raise ValueError(
            "constituents debe contener la columna 'Security'. "
            "Usa get_sp500_constituents()."
        )

    work = constituents.copy()
    work["Issuer"] = work["Security"].map(_issuer_key)

    counts = work["Issuer"].value_counts()
    multi_issuer = counts[counts > 1].index
    if len(multi_issuer) == 0:
        work["Issuer"] = work["Issuer"].fillna("")
        removed = pd.DataFrame(columns=["Symbol", "Issuer", "Representative"])
        return work, removed

    first_per_issuer = (
        work[work["Issuer"].isin(multi_issuer)]
        .drop_duplicates(subset="Issuer", keep="first")
    )
    representative = first_per_issuer.set_index("Issuer")["Symbol"].to_dict()
    duplicates = work[
        work["Issuer"].isin(multi_issuer)
        & ~work["Symbol"].isin(representative.values())
    ].copy()
    duplicates["Representative"] = duplicates["Issuer"].map(representative)
    removed = (
        duplicates[["Symbol", "Issuer", "Representative"]]
        .reset_index(drop=True)
    )

    keepers_mask = ~work["Symbol"].isin(removed["Symbol"])
    kept = work.loc[keepers_mask].reset_index(drop=True)
    return kept, removed


def get_sp500_tickers_by_sector(
    target_sector: str = "Energy",
    unique_issuers: bool = False,
) -> tuple[list[str], pd.DataFrame] | list[str]:
    """Scrapea los constituyentes del S&P 500 y los filtra por sector GICS.

    Parameters
    ----------
    target_sector : str
        Nombre del sector GICS.  ``"all"`` devuelve todos los tickers sin
        filtrar (útil como universo completo).
    unique_issuers : bool
        Si ``True``, aplica :func:`filter_duplicate_issuers` antes de
        devolver los tickers.  En este caso la función devuelve también
        la tabla de símbolos descartados.

    Returns
    -------
    list[str] o tuple[list[str], pd.DataFrame]
        Tickers compatibles con ``yfinance``.  Si ``unique_issuers=True``
        devuelve una tupla ``(tickers, removed)``.
    """
    df = get_sp500_constituents()

    if target_sector.lower() != "all":
        available = df["GICS Sector"].unique()
        if target_sector not in available:
            raise ValueError(
                f"Sector '{target_sector}' no encontrado. Sectores válidos: {available}"
            )
        df = df.loc[df["GICS Sector"] == target_sector].copy()

    if unique_issuers:
        kept, removed = filter_duplicate_issuers(df)
        return kept["Symbol"].tolist(), removed

    return df["Symbol"].tolist()


def _download_prices(
    tickers: Sequence[str] | str,
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Descarga el frame de precios ``Adj Close`` sin filtrar columnas."""
    if isinstance(tickers, str):
        tickers = [tickers]

    prices = load_multiple_prices(list(tickers), start, end, cache_dir, refresh)
    if prices.empty:
        raise ValueError("No se descargaron datos para el rango solicitado.")
    return prices


def filter_by_history(
    prices: pd.DataFrame,
    min_history_obs: int = 0,
    min_valid_obs: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarta los tickers con historia insuficiente dentro de la ventana.

    El filtro se aplica **antes** de alinear las fechas.  Es la clave para
    que la intersección de fechas no la determine un único activo recién
    cotizado: si un ticker sólo tiene 46 observaciones, alinear todo el
    universo con él reduciría el panel a esas 46 sesiones.

    Parameters
    ----------
    prices : pd.DataFrame
        Precios sin alinear, con una columna por ticker.
    min_history_obs : int
        Mínimo de observaciones no nulas dentro de la ventana para conservar
        un ticker (historia real, no sólo datos presentes).
    min_valid_obs : int
        Mínimo absoluto de observaciones no nulas; si ``min_history_obs`` es
        menor, el umbral efectivo es el mayor de los dos.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Panel filtrado y tabla ``dropped`` con columnas ``n_obs``,
        ``primera_fecha``, ``ultima_fecha`` y ``motivo``, ordenada por número
        de observaciones.
    """
    umbral = max(min_history_obs, min_valid_obs)
    conteo = prices.notna().sum()

    if conteo.empty:
        return prices.copy(), _empty_drop_table()

    excedentes = conteo.index[conteo >= umbral]
    insuficientes = conteo.index[conteo < umbral]

    panel = prices[excedentes].copy()

    if len(insuficientes) == 0:
        return panel, _empty_drop_table()

    motivo = np.where(
        conteo.loc[insuficientes] < min_history_obs,
        "historia_insuficiente",
        "sin_datos",
    )
    dropped = pd.DataFrame(
        {
            "n_obs": conteo.loc[insuficientes],
            "primera_fecha": _extremes(prices[insuficientes], "min"),
            "ultima_fecha": _extremes(prices[insuficientes], "max"),
            "motivo": motivo,
        },
        index=pd.Index(insuficientes, name="ticker"),
    ).sort_values("n_obs")
    return panel, dropped


def _extremes(prices: pd.DataFrame, which: str) -> pd.Series:
    """Primera o última fecha con precio válido de cada columna."""
    return prices.apply(lambda col: col.dropna().index.min() if which == "min"
                        else col.dropna().index.max())


def _empty_drop_table() -> pd.DataFrame:
    """Tabla de descarte vacía con las columnas esperadas."""
    return pd.DataFrame(
        {
            "n_obs": pd.Series(dtype="int64"),
            "primera_fecha": pd.Series(dtype="datetime64[ns]"),
            "ultima_fecha": pd.Series(dtype="datetime64[ns]"),
            "motivo": pd.Series(dtype="object"),
        },
        index=pd.Index([], name="ticker"),
    )


@dataclass(frozen=True)
class DatasetReport:
    """Diagnóstico de cobertura de un panel de retornos alineado.

    Sirve para detectar el fallo más insidioso de estos cuadernos: pedir un
    rango de treinta años y acabar analysizando dos meses porque la
    intersección de fechas la fijó el activo con menos historia.

    Attributes
    ----------
    requested_start, requested_end : str
        Ventana temporal solicitada.
    first_date, last_date : pd.Timestamp
        Fechas efectivas de la primera y la última sesión del panel.
    n_sessions : int
        Número de sesiones (observaciones de retorno) del panel.
    n_assets : int
        Activos que sobreviven al filtro de historia.
    n_assets_requested : int
        Activos solicitados antes de filtrar.
    min_history_obs : int
        Umbral de historia aplicado.  Se registra en el informe para que la
        tabla exportada documente qué filtro produjo la muestra.
    union_sessions : int
        Fechas distintas en las que cotizó **algún** activo superviviente: el
        máximo de sesiones que permite alinear ese universo.  Comparado con
        ``n_sessions`` revela cuánto se pierde por exigir la intersección.
    coverage_pct : float
        ``n_sessions / union_sessions``.  Es la métrica que delata el fallo
        de alineación: un valor minúsculo significa que la fecha más temprana
        la fija un activo sin historia, no que falten datos.
    window_pct : float
        ``n_sessions / expected_sessions`` sobre la ventana solicitada.  Suele
        ser bajo por diseño, ya que la mayoría de los componentes actuales
        del S&P 500 no cotizaba todavía en el inicio del rango.
    expected_sessions : int
        Días hábiles entre las fechas solicitadas (límite superior, no
        descuenta festivos bursátiles).
    calendar_days : int
        Días naturales entre la primera y la última sesión efectivas.
    sessions_per_year : float
        Sesiones efectivas por año natural; un valor cercano a 252 confirma
        que el índice es de sesiones bursátiles y no de días calendario.
    history_per_asset : pd.Series
        Observaciones de precio disponibles por activo dentro de la ventana,
        antes de alinear.
    dropped_assets : pd.DataFrame
        Activos descartados con ``n_obs``, fechas y ``motivo``.
    """

    requested_start: str
    requested_end: str
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    n_sessions: int
    n_assets: int
    n_assets_requested: int
    min_history_obs: int
    union_sessions: int
    coverage_pct: float
    window_pct: float
    expected_sessions: int
    calendar_days: int
    sessions_per_year: float
    history_per_asset: pd.Series
    dropped_assets: pd.DataFrame


#: Mínimo de sesiones para que los estadísticos (correlación, ADF, OU) sean
#: interpretables; por debajo se advierte en lugar de devolver números sin
#: contexto.  Un año bursátil.
MIN_SESSIONS_WARN = 252


def _build_dataset_report(
    returns: pd.DataFrame,
    history: pd.DataFrame,
    dropped: pd.DataFrame,
    requested_start: str,
    requested_end: str,
    n_requested: int,
    min_history_obs: int,
) -> DatasetReport:
    """Construye el :class:`DatasetReport` de un panel ya alineado."""
    inicio, fin = pd.Timestamp(requested_start), pd.Timestamp(requested_end)
    # ``np.busday_count`` exige granularidad diaria.
    expected = int(
        np.busday_count(
            np.datetime64(inicio.to_datetime64(), "D"),
            np.datetime64(fin.to_datetime64(), "D"),
        )
    )
    n_sessions = len(returns)
    dias = (returns.index[-1] - returns.index[0]).days if n_sessions else 0
    años = dias / 365.25 if dias else float("nan")
    union = int(history.notna().any(axis=1).sum())

    return DatasetReport(
        requested_start=requested_start,
        requested_end=requested_end,
        first_date=returns.index[0],
        last_date=returns.index[-1],
        n_sessions=n_sessions,
        n_assets=returns.shape[1],
        n_assets_requested=n_requested,
        min_history_obs=min_history_obs,
        union_sessions=union,
        coverage_pct=n_sessions / union if union else float("nan"),
        window_pct=n_sessions / expected if expected else float("nan"),
        expected_sessions=expected,
        calendar_days=dias,
        sessions_per_year=n_sessions / años if años else float("nan"),
        history_per_asset=history.notna().sum(),
        dropped_assets=dropped,
    )


def _warn_on_low_coverage(report: DatasetReport) -> None:
    """Advierte si la muestra efectiva no permite interpretar los análisis.

    El criterio es el tamaño absoluto de la muestra, no la cobertura: un
    universo de componentes actuales no puede cubrir toda la ventana pedida
    porque muchas empresas no cotizaban todavía al principio del rango.  Lo
    que sí indica un fallo es una intersección minúscula, y eso queda
    reflejado en ``coverage_pct`` frente a ``union_sessions``.
    """
    if report.n_sessions < MIN_SESSIONS_WARN:
        warnings.warn(
            f"Panel de retornos degradado: sólo {report.n_sessions} sesiones "
            f"(mínimo recomendado {MIN_SESSIONS_WARN}) y una cobertura del "
            f"{report.coverage_pct:.1%} de las {report.union_sessions} sesiones "
            f"disponibles. Suele indicar que algún activo sin historia está "
            f"truncando la intersección: revisa min_history_obs y la ventana.",
            UserWarning,
            stacklevel=3,
        )


def print_dataset_report(
    report: DatasetReport,
    min_history_obs: int | None = None,
    max_dropped: int = 10,
) -> None:
    """Imprime el diagnóstico de cobertura de un panel de retornos.

    Existe para que los cuadernos que comparten la fase de preparación
    (Pearson e Información Mutua) no dupliquen el bloque de impresiones y
    cualquier recorte de la muestra quede visible antes de interpretar los
    resultados.

    Parameters
    ----------
    report : DatasetReport
        Diagnóstico devuelto por :func:`load_returns_panel`.
    min_history_obs : int, opcional
        Umbral de historia aplicado, sólo para mostrarlo en el resumen.
    max_dropped : int
        Número máximo de activos descartados que se listan en detalle.
    """
    print(f"\nVentana solicitada : {report.requested_start} → "
          f"{report.requested_end}")
    print(
        f"Ventana efectiva   : {report.first_date:%Y-%m-%d} → "
        f"{report.last_date:%Y-%m-%d}"
    )
    print(
        f"Sesiones           : {report.n_sessions} "
        f"({report.calendar_days} días naturales, "
        f"{report.sessions_per_year:.0f} sesiones/año)"
    )
    print(
        f"Cobertura          : {report.coverage_pct:.1%} de las "
        f"{report.union_sessions} fechas disponibles y "
        f"{report.window_pct:.1%} de la ventana solicitada"
    )
    umbral = (
        f" (exige ≥ {min_history_obs} sesiones)"
        if min_history_obs is not None
        else ""
    )
    print(
        f"Activos            : {report.n_assets} de "
        f"{report.n_assets_requested}{umbral}"
    )
    historia = report.history_per_asset
    if not historia.empty:
        print(
            f"Historia por activo: min {historia.min()}, "
            f"mediana {historia.median():.0f}, max {historia.max()}"
        )
    if not report.dropped_assets.empty:
        print(
            f"\nActivos descartados por historia insuficiente "
            f"({len(report.dropped_assets)}):"
        )
        print(report.dropped_assets.head(max_dropped).to_string())
        restantes = len(report.dropped_assets) - max_dropped
        if restantes > 0:
            print(f"... y {restantes} más.")


@dataclass(frozen=True)
class PairSelectionUniverse:
    """Universo preparado para los cuadernos de selección de pares.

    Agrupa lo que ambos cuaderos necesitan antes de estimar su métrica:
    los tickers únicos por emisor, sus sectores, sus emisores, el panel de
    log-retornos alineado y su diagnóstico de cobertura.

    Attributes
    ----------
    returns : pd.DataFrame
        Log-retornos alineados del panel común.
    report : DatasetReport
        Diagnóstico de cobertura del panel.
    sectors : pd.Series
        Sector GICS de cada ticker presente en el panel.
    issuers : pd.Series
        Emisor de cada ticker presente en el panel.
    n_sectors : int
        Número de sectores GICS distintos presentes.
    removed_duplicate_issuers : pd.DataFrame
        Filas descartadas por compartir emisor con otra clase de acción.
    """

    returns: pd.DataFrame
    report: DatasetReport
    sectors: pd.Series
    issuers: pd.Series
    n_sectors: int
    removed_duplicate_issuers: pd.DataFrame


def prepare_pair_selection_universe(
    constituents: pd.DataFrame | None = None,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
    min_history_obs: int = PAIR_SELECTION_MIN_HISTORY_OBS,
    min_valid_obs: int = 3,
    verbose: bool = True,
) -> PairSelectionUniverse:
    """Prepara el universo común de los cuadernos de selección de pares.

    Encadena la fase compartida por ambos cuadernos: elimina una clase de
    acción por emisor, construye el panel de log-retornos con
    :func:`load_returns_panel` (que filtra por historia **antes** de alinear)
    y alinea sectores y emisores con los tickers que sobreviven.

    Mantener esta fase en un único sitio garantiza que la matriz de Pearson
    y la de Información Mutua se estimen sobre exactamente la misma muestra.

    Parameters
    ----------
    constituents : pd.DataFrame, opcional
        Tabla de constituyentes del S&P 500.  Si se omite se descarga con
        :func:`get_sp500_constituents`.
    start, end : str
        Ventana temporal solicitada.
    cache_dir : Path
        Carpeta de la caché CSV de precios.
    refresh : bool
        Si ``True``, fuerza la descarga de precios aunque exista caché.
    min_history_obs : int
        Mínimo de sesiones por ticker dentro de la ventana.
    min_valid_obs : int
        Mínimo absoluto de observaciones válidas por ticker.
    verbose : bool
        Si ``True``, imprime el tamaño del universo, los emisores
        descartados y el diagnóstico completo de cobertura.

    Returns
    -------
    PairSelectionUniverse
        Panel de retornos, diagnóstico y metadatos alineados.
    """
    if constituents is None:
        constituents = get_sp500_constituents()

    kept, removed = filter_duplicate_issuers(constituents)
    tickers = kept["Symbol"].tolist()

    if verbose:
        print(f"Tamaño del universo S&P 500: {len(tickers)} activos únicos")
        if not removed.empty:
            print("Símbolos descartados (mismo emisor que otro ya presente):")
            print(removed.to_string(index=False))

    returns, report = load_returns_panel(
        tickers,
        start,
        end,
        cache_dir,
        refresh,
        min_valid_obs,
        min_history_obs,
    )

    sectors = kept.set_index("Symbol")["GICS Sector"].reindex(
        returns.columns
    ).dropna()
    issuers = kept.set_index("Symbol")["Issuer"].reindex(returns.columns)

    if verbose:
        print_dataset_report(report, min_history_obs=min_history_obs)
        print(f"Sectores GICS presentes: {sectors.nunique()}")

    return PairSelectionUniverse(
        returns=returns,
        report=report,
        sectors=sectors,
        issuers=issuers,
        n_sectors=int(sectors.nunique()),
        removed_duplicate_issuers=removed,
    )


# =============================================================================
# Exportación de tablas para la tesis
# =============================================================================

#: Caracteres con significado especial en LaTeX y su reemplazo.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text: object) -> str:
    """Escapa los caracteres especiales de LaTeX de una cadena."""
    return "".join(_LATEX_ESCAPES.get(char, char) for char in str(text))


def _format_latex_cell(value: object, decimals: int) -> str:
    """Formatea una celda para ``tabular``: ausentes, booleanos y números.

    Los booleanos se muestran como ``Sí``/``No`` y los enteros con separador
    de miles, para que las sesiones de la tabla se lean de un vistazo.
    """
    if isinstance(value, (bool, np.bool_)):
        return r"\textbf{Sí}" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", r"\,")
    if isinstance(value, (float, np.floating)):
        return "--" if np.isnan(value) else f"{float(value):.{decimals}f}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "--"
    return _latex_escape(value)


def _latex_tabular(display: pd.DataFrame, decimals: int) -> str:
    """Construye el ``tabular`` con ``booktabs`` de un DataFrame.

    Se implementa a mano en lugar de usar ``DataFrame.to_latex`` porque en
    pandas 3 ese método delega en ``Styler``, que exige ``jinja2``: una
    dependencia nueva sólo para formatear una tabla de veinte filas.  Así se
    controlan con exactitud el escapado, los valores ausentes y la
    alineación por tipo de columna.
    """
    aligns = "".join(
        "r" if pd.api.types.is_numeric_dtype(display[col]) else "l"
        for col in display.columns
    ) or "l"
    header = " & ".join(_latex_escape(col) for col in display.columns)
    # La cadena cruda es imprescindible: en una f-string normal ``"\\\\"``
    # colapsaría a un único "\", que rompe la alineación de LaTeX.
    fin_de_fila = r" \\"
    lines = [
        rf"\begin{{tabular}}{{{aligns}}}",
        r"\toprule",
        f"{header}" + fin_de_fila,
        r"\midrule",
    ]
    for _, row in display.iterrows():
        cells = [_format_latex_cell(value, decimals) for value in row]
        lines.append(" & ".join(cells) + fin_de_fila)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _jsonable(value: object) -> object:
    """Convierte tipos de pandas y numpy en valores serializables en JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Series):
        return {str(k): _jsonable(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="list"))
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def table_conditions(
    report: DatasetReport,
    method: str,
    n_pairs: int,
    notes: Sequence[str] = (),
) -> list[str]:
    """Redacta las condiciones de cálculo que acompañan a una tabla.

    Una tabla de resultados sin las condiciones con las que se produjo es
    irreproducible: aquí se recogen la ventana solicitada y la efectiva, el
    tamaño real de la muestra, el filtro de historia aplicado y la cobertura,
    de modo que el lector pueda reconstruir exactamente el panel.  Las
    ``notes`` del cuaderno (significancia del test, número de vecinos, ...)
    se añaden al final ya redactadas.
    """
    lineas = [
        f"Método de selección: {method}.",
        (
            f"Ventana solicitada: {report.requested_start} a "
            f"{report.requested_end}."
        ),
        (
            f"Ventana efectiva: {report.first_date:%Y-%m-%d} a "
            f"{report.last_date:%Y-%m-%d}."
        ),
        (
            f"Muestra: {report.n_sessions} sesiones "
            f"({report.calendar_days} días naturales, "
            f"{report.sessions_per_year:.1f} sesiones/año)."
        ),
        (
            f"Universo: {report.n_assets} activos de "
            f"{report.n_assets_requested} solicitados "
            f"({len(report.dropped_assets)} descartados por historia "
            f"insuficiente)."
        ),
        (
            f"Filtro de historia: al menos {report.min_history_obs} sesiones "
            f"por activo dentro de la ventana."
        ),
        (
            f"Cobertura: {report.coverage_pct:.1%} de las "
            f"{report.union_sessions} fechas disponibles y "
            f"{report.window_pct:.1%} de la ventana solicitada."
        ),
        f"Pares seleccionados: {n_pairs}.",
    ]
    lineas.extend(str(nota) for nota in notes)
    return lineas


def _with_rank(table: pd.DataFrame) -> pd.DataFrame:
    """Antepone la columna ``rank`` si la tabla todavía no la tiene."""
    if "rank" in table.columns:
        return table
    con_rank = table.copy()
    con_rank.insert(0, "rank", np.arange(1, len(con_rank) + 1))
    return con_rank


def select_table_columns(
    table: pd.DataFrame,
    columns: Sequence[str] | None,
) -> pd.DataFrame:
    """Selecciona y reordena las columnas de una tabla de resultados.

    Permite que cada cuaderno elija qué información acaba en el ``.tex`` y
    en el ``.csv`` sin perder de vista el resto: las condiciones de cálculo
    viajan aparte, en la nota y en el JSON, así que recortar columnas no
    hace la tabla menos reproducible.

    Parameters
    ----------
    table : pd.DataFrame
        Tabla de partida, ya con ``rank`` si procede.
    columns : Sequence[str] o None
        Columnas a exportar, en el orden deseado.  ``None`` las exporta
        todas.  La lista se respeta al pie de la letra: si no incluye
        ``rank``, la tabla exportada no lo lleva.

    Returns
    -------
    pd.DataFrame
        La tabla con exactamente las columnas pedidas, en ese orden.

    Raises
    ------
    TypeError
        Si ``columns`` es una cadena suelta en lugar de una secuencia.
    ValueError
        Si la lista está vacía, repite columnas o pide alguna inexistente.
    """
    if columns is None:
        return table
    if isinstance(columns, str):
        raise TypeError(
            "``columns`` debe ser una secuencia de nombres de columna, no "
            f"una cadena suelta: {columns!r}. Usa [{columns!r}] si sólo "
            "quieres una columna."
        )

    pedidas = list(columns)
    if not pedidas:
        raise ValueError("``columns`` no puede ser una lista vacía.")

    repetidas = [c for c in dict.fromkeys(pedidas) if pedidas.count(c) > 1]
    if repetidas:
        raise ValueError(f"Columnas repetidas en ``columns``: {repetidas}.")

    faltantes = [c for c in pedidas if c not in table.columns]
    if faltantes:
        raise ValueError(
            f"Columnas inexistentes en la tabla: {faltantes}. "
            f"Disponibles: {list(table.columns)}."
        )
    return table.loc[:, pedidas]


def export_pair_selection_table(
    table: pd.DataFrame,
    report: DatasetReport,
    output_dir: Path | None = None,
    stem: str = "parejas",
    method: str = "",
    caption: str = "",
    label: str = "",
    columns: Sequence[str] | None = None,
    extra_metadata: Mapping[str, object] | None = None,
    notes: Sequence[str] = (),
    decimals: int = 4,
) -> dict[str, Path]:
    """Exporta una tabla de pares a CSV, LaTeX y JSON de metadatos.

    Escribe tres ficheros en ``output_dir`` con el mismo ``stem``:

    - ``.csv`` con los valores sin formatear, para reutilizarlos;
    - ``.tex`` listo para ``\\input`` en la tesis, con las condiciones de
      cálculo como nota bajo la tabla;
    - ``.json`` con las condiciones estructuradas, para documentarlas o
      compararlas entre ejecuciones.

    Parameters
    ----------
    table : pd.DataFrame
        Tabla de pares.  Si no trae la columna ``rank``, se antepone.
    report : DatasetReport
        Diagnóstico del panel con el que se calculó la tabla.
    output_dir : Path, opcional
        Destino; por defecto ``TABLES_DIR`` (``outputs/tables``).
    stem : str
        Nombre base de los tres ficheros.
    method : str
        Descripción del método, por ejemplo ``"correlación de Pearson"``.
    caption, label : str
        Texto y etiqueta LaTeX de la tabla.
    columns : Sequence[str], opcional
        Columnas a exportar y su orden.  ``None`` (por defecto) exporta
        todas.  Ver :func:`select_table_columns`.
    extra_metadata : Mapping, opcional
        Claves y valores adicionales del cuaderno que se guardan en el JSON
        (número de vecinos KSG, significancia del ADF, ...).
    notes : Sequence[str], opcional
        Condiciones adicionales del cuaderno, ya redactadas, que se añaden
        a la nota de la tabla LaTeX.
    decimals : int
        Decimales de las columnas de punto flotante en el ``.tex``.

    Returns
    -------
    dict[str, Path]
        Rutas de los ficheros escritos, con las claves ``csv``, ``tex`` y
        ``json``.
    """
    if table.empty:
        raise ValueError("La tabla a exportar no tiene filas.")

    destino = Path(output_dir) if output_dir is not None else TABLES_DIR
    destino.mkdir(parents=True, exist_ok=True)

    tabla = select_table_columns(_with_rank(table), columns)

    csv_path = destino / f"{stem}.csv"
    tex_path = destino / f"{stem}.tex"
    json_path = destino / f"{stem}.json"

    tabla.to_csv(csv_path, index=False)

    lineas = table_conditions(report, method, len(tabla), notes)
    metadata = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "metodo": method,
        "columnas": list(tabla.columns),
        "n_pares": len(tabla),
        "ventana_solicitada": {
            "inicio": report.requested_start,
            "fin": report.requested_end,
        },
        "ventana_efectiva": {
            "inicio": report.first_date.strftime("%Y-%m-%d"),
            "fin": report.last_date.strftime("%Y-%m-%d"),
        },
        "sesiones": report.n_sessions,
        "dias_naturales": report.calendar_days,
        "sesiones_por_ano": report.sessions_per_year,
        "activos_solicitados": report.n_assets_requested,
        "activos_conservados": report.n_assets,
        "activos_descartados": len(report.dropped_assets),
        "umbral_historia_sesiones": report.min_history_obs,
        "fechas_disponibles_union": report.union_sessions,
        "cobertura_panel_pct": report.coverage_pct,
        "cobertura_ventana_pct": report.window_pct,
        "condiciones": lineas,
    }
    metadata.update(dict(extra_metadata or {}))
    json_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    caption = caption or method or stem
    cuerpo = [
        (
            "% Generado automáticamente por "
            "thesis.pair_trading.export_pair_selection_table"
        ),
        "% No editar a mano: se regenera en cada ejecución del cuaderno.",
        "% Requiere únicamente el paquete booktabs.",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
    ]
    if label:
        cuerpo.append(rf"\label{{{label}}}")
    cuerpo += [_latex_tabular(tabla, decimals), "", r"\vspace{2pt}"]
    # Las condiciones van en un ``minipage`` y no en ``tablenotes``: dentro
    # de ``threeparttable`` tanto ``itemize`` como los párrafos sueltos
    # abortan la compilación, y así sólo hace falta ``booktabs``.
    cuerpo += [r"\begin{minipage}{0.98\linewidth}", r"\footnotesize"]
    cuerpo += [
        rf"\textbullet\ {_latex_escape(linea)}\par" for linea in lineas
    ]
    cuerpo += [r"\end{minipage}", r"\end{table}"]
    tex_path.write_text("\n".join(cuerpo) + "\n", encoding="utf-8")

    print(f"Tabla exportada: {csv_path.name}, {tex_path.name}, {json_path.name}")
    return {"csv": csv_path, "tex": tex_path, "json": json_path}


def _load_aligned_prices(
    tickers: Sequence[str] | str,
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool,
    min_valid_obs: int,
    min_history_obs: int = 0,
) -> pd.DataFrame:
    """Descarga precios ``Adj Close`` y conserva sólo los tickers válidos.

    Aplica el filtro de historia antes de alinear y exige que sobrevivan al
    menos dos columnas.
    """
    prices = _download_prices(tickers, start, end, cache_dir, refresh)
    panel, _ = filter_by_history(prices, min_history_obs, min_valid_obs)

    if len(panel.columns) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 tickers válidos. Encontrados: "
            f"{list(panel.columns)}"
        )
    return panel


def load_returns_panel(
    tickers: Sequence[str] | str,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
    min_valid_obs: int = 3,
    min_history_obs: int = 0,
) -> tuple[pd.DataFrame, DatasetReport]:
    """Construye el panel de log-retornos alineado junto con su diagnóstico.

    Descarga los precios, descarta por historia los tickers demasiado
    recientes, alinea las fechas de los supervivientes y calcula
    $r_t = \\log(P_t / P_{t-1})$.

    El orden importa: alinear **antes** de filtrar haría que un único activo
    recién cotizado truncase el panel completo, ya que ``dropna`` conserva
    sólo las fechas en las que cotizan todos los activos.

    Parameters
    ----------
    tickers : Sequence[str] | str
        Lista de tickers (o un único ticker) compatible con ``yfinance``.
    start, end : str
        Ventana temporal solicitada.
    cache_dir : Path
        Carpeta donde ``load_multiple_prices`` guarda la caché CSV.
    refresh : bool
        Si ``True``, fuerza la descarga aunque exista caché.
    min_valid_obs : int
        Mínimo absoluto de observaciones válidas por ticker.
    min_history_obs : int
        Mínimo de observaciones dentro de la ventana.  Un valor de ``1260``
        (~5 años) mantiene la historia suficiente sin depender de que el
        activo más nuevo haya cotizado durante toda la ventana.

    Returns
    -------
    tuple[pd.DataFrame, DatasetReport]
        Retornos logarítmios alineados y el diagnóstico de cobertura.
    """
    prices = _download_prices(tickers, start, end, cache_dir, refresh)
    n_requested = len(prices.columns)

    panel, dropped = filter_by_history(prices, min_history_obs, min_valid_obs)
    if len(panel.columns) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 tickers válidos. Encontrados: "
            f"{list(panel.columns)}"
        )

    log_ret = np.log(panel / panel.shift(1)).dropna(how="any")
    if log_ret.empty:
        raise ValueError("No quedan observaciones tras el alineamiento.")

    report = _build_dataset_report(
        log_ret, panel, dropped, start, end, n_requested, min_history_obs,
    )
    _warn_on_low_coverage(report)
    return log_ret, report


def load_log_prices(
    tickers: Sequence[str] | str,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
    min_valid_obs: int = 3,
    min_history_obs: int = 0,
) -> pd.DataFrame:
    """Descarga precios ``Adj Close`` y devuelve los log-precios limpios.

    Reutiliza :func:`thesis.market_data.load_multiple_prices` para
    aprovechar la caché CSV y aplica $\\log P_t$ sobre las observaciones
    comunes a todos los tickers válidos.

    Es la serie que requiere el análisis de cointegración: el hedge ratio y
    el test ADF se estiman sobre $\\log P$, no sobre retornos.  Acumular los
    log-retornos con ``cumsum`` no es equivalente, porque descarta el
    primer nivel y pierde observaciones en las fechas donde un activo no
    cotizó, de modo que las series resultantes no coinciden con las que
    produce :func:`thesis.market_data.load_multiple_prices` más ``np.log``.

    Parameters
    ----------
    tickers : Sequence[str] | str
        Lista de tickers (o un único ticker) compatible con ``yfinance``.
    start, end : str
        Fechas en formato ``"YYYY-MM-DD"``.
    cache_dir : Path
        Carpeta donde ``load_multiple_prices`` guarda la caché CSV.
    refresh : bool
        Si ``True``, fuerza la descarga aunque exista caché.
    min_valid_obs : int
        Mínimo de observaciones válidas para conservar un ticker.
    min_history_obs : int
        Mínimo de observaciones dentro de la ventana.

    Returns
    -------
    pd.DataFrame
        Log-precios con índice temporal y un ticker por columna.
    """
    prices = _load_aligned_prices(
        tickers, start, end, cache_dir, refresh, min_valid_obs, min_history_obs,
    )

    log_prices = np.log(prices.where(prices > 0)).dropna(how="any")
    if log_prices.empty:
        raise ValueError("No quedan observaciones tras el alineamiento.")
    return log_prices


def load_log_returns(
    tickers: Sequence[str] | str,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
    min_valid_obs: int = 3,
    min_history_obs: int = 0,
) -> pd.DataFrame:
    """Descarga precios ``Adj Close`` y devuelve los log-retornos limpios.

    Atajo sobre :func:`load_returns_panel` para los consumidores que no
    necesitan el diagnóstico de cobertura.  El log-retorno es
    $r_t = \\log(P_t / P_{t-1})$.

    Parameters
    ----------
    tickers : Sequence[str] | str
        Lista de tickers (o un único ticker) compatible con ``yfinance``.
    start, end : str
        Fechas en formato ``"YYYY-MM-DD"``.
    cache_dir : Path
        Carpeta donde ``load_multiple_prices`` guarda la caché CSV.
    refresh : bool
        Si ``True``, fuerza la descarga aunque exista caché.
    min_valid_obs : int
        Mínimo de observaciones válidas por ticker.
    min_history_obs : int
        Mínimo de observaciones dentro de la ventana.

    Returns
    -------
    pd.DataFrame
        Retornos logarítmicos con índice temporal y un ticker por columna.
    """
    log_ret, _ = load_returns_panel(
        tickers, start, end, cache_dir, refresh, min_valid_obs, min_history_obs,
    )
    return log_ret


# =============================================================================
# Hedge ratio y spread
# =============================================================================


@dataclass(frozen=True)
class HedgeResult:
    """Resultado de la regresión OLS log-lineal ``log A = alpha + beta log B + epsilon``.

    Attributes
    ----------
    alpha : float
        Intercepto de la regresión.
    beta : float
        Coeficiente de hedge ratio.
    r_squared : float
        R² de la regresión.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo completo para diagnósticos adicionales.
    """

    alpha: float
    beta: float
    r_squared: float
    model: object


@dataclass(frozen=True)
class ADFResult:
    """Resumen del test Augmented Dickey-Fuller aplicado al spread.

    Attributes
    ----------
    statistic : float
        Estadístico ADF.  Valores muy negativos rechazan la hipótesis nula
        de raíz unitaria.
    pvalue : float
        P-value asociado al estadístico.
    is_stationary : bool
        ``True`` si ``pvalue < significance``.
    """

    statistic: float
    pvalue: float
    is_stationary: bool


def compute_hedge_ratio(
    log_a: pd.Series, log_b: pd.Series
) -> HedgeResult:
    """Estima ``log(A) = alpha + beta * log(B) + epsilon`` por OLS.

    El residuo de la regresión es el spread cointegrado candidato:
    $\\varepsilon_t = \\log A_t - \\alpha - \\beta \\log B_t$.

    Parameters
    ----------
    log_a, log_b : pd.Series
        Series de log-precios alineadas por índice.

    Returns
    -------
    HedgeResult
        Estimaciones de ``alpha``, ``beta``, R² y el modelo de
        ``statsmodels`` subyacente.
    """
    data = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    if data.empty:
        raise ValueError("No hay observaciones comunes para ajustar el hedge.")
    X = sm.add_constant(data["b"])
    model = sm.OLS(data["a"], X).fit()
    return HedgeResult(
        alpha=float(model.params["const"]),
        beta=float(model.params["b"]),
        r_squared=float(model.rsquared),
        model=model,
    )


def adf_test(spread: pd.Series, significance: float = 0.05) -> ADFResult:
    """Test Augmented Dickey-Fuller sobre el spread.

    Contrasta la hipótesis nula de raíz unitaria en la serie
    $\\Delta y_t = \\alpha + \\beta t + \\gamma y_{t-1} +
    \\sum_i \\delta_i \\Delta y_{t-i} + \\varepsilon_t$.  Si el p-value
    cae por debajo de ``significance`` el spread se considera
    estacionario.

    Parameters
    ----------
    spread : pd.Series
        Serie del spread ``log(A) - (alpha + beta * log(B))``.
    significance : float
        Nivel de significancia (por defecto 0.05).

    Returns
    -------
    ADFResult
        Estadístico ADF, p-value y bandera ``is_stationary``.
    """
    clean = spread.dropna()
    if clean.empty:
        raise ValueError("El spread no contiene observaciones válidas.")
    result = adfuller(clean, autolag="AIC", result_object=False)
    stat, pvalue, *_ = result
    return ADFResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        is_stationary=pvalue < significance,
    )


def extract_and_construct_spread(
    ticker_a: str,
    ticker_b: str,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
) -> tuple[pd.Series, float]:
    """Pipeline completo: descarga, hedge ratio, spread y diagnóstico ADF.

    Pasos:
    1. Descarga los precios ``Adj Close`` de los dos activos vía
       :func:`thesis.market_data.load_multiple_prices`.
    2. Calcula los log-precios $\\log A_t$ y $\\log B_t$.
    3. Ajusta la OLS ``log A = alpha + beta * log B + epsilon`` y
       construye el spread residual
       $\\varepsilon_t = \\log A_t - \\alpha - \\beta \\log B_t$.
    4. Aplica el test ADF sobre el spread.

    Imprime por consola el hedge ratio, el R², el estadístico ADF y el
    p-value.  Devuelve el spread y el coeficiente ``beta``.

    Returns
    -------
    spread : pd.Series
        Residuo ``log(A) - (alpha + beta * log(B))``.
    beta : float
        Coeficiente de hedge ratio estimado.
    """
    prices = load_multiple_prices([ticker_a, ticker_b], start, end, cache_dir, refresh)
    prices = prices.dropna()
    if prices.empty:
        raise ValueError("No se descargaron datos. Verifica los tickers o fechas.")

    log_a = np.log(prices[ticker_a])
    log_b = np.log(prices[ticker_b])

    hedge = compute_hedge_ratio(log_a, log_b)
    spread = log_a - (hedge.alpha + hedge.beta * log_b)

    print(f"Hedge Ratio (Beta): {hedge.beta:.4f}")
    print(f"Intercept (Alpha): {hedge.alpha:.4f}")
    print(f"R-squared: {hedge.r_squared:.4f}")

    adf = adf_test(spread)
    print(f"ADF Statistic: {adf.statistic:.4f}")
    print(f"p-value: {adf.pvalue:.4f}")
    if adf.is_stationary:
        print("ÉXITO: el spread es estacionario.")
    else:
        print("ALERTA: el spread NO es estacionario.")

    return spread, hedge.beta


# =============================================================================
# Ornstein-Uhlenbeck
# =============================================================================


@dataclass(frozen=True)
class OUParameters:
    """Parámetros continuos del proceso de Ornstein-Uhlenbeck.

    Attributes
    ----------
    theta : float
        Velocidad de reversión a la media (rigidez del muelle).
    mu : float
        Media de equilibrio del spread.
    sigma : float
        Volatilidad instantánea de los shocks.
    half_life : float
        Tiempo característico de reversión (en unidades de ``dt``).
    sigma_eq : float
        Desviación típica de la distribución estacionaria
        $\\sigma / \\sqrt{2\\theta}$.
    r_squared : float
        R² del AR(1) usado para la calibración.
    """

    theta: float
    mu: float
    sigma: float
    half_life: float
    sigma_eq: float
    r_squared: float


def fit_ou_process(spread: pd.Series, dt: float = 1.0) -> OUParameters:
    """Ajusta un OU a un spread estacionario mediante AR(1) → OU.

    El modelo discreto ``Z_t = a + b * Z_{t-1} + eps_t`` se traduce a
    los parámetros continuos ``(theta, mu, sigma)`` asumiendo ``dt``
    constante:

    - $\\theta = -\\log(b)/\\Delta t$
    - $\\mu = a / (1 - b)$
    - $\\sigma = \\sqrt{2\\theta\\,\\mathrm{Var}(\\varepsilon)/(1 - b^2)}$
    - Vida media: $t_{1/2} = \\log 2 / \\theta$
    - Desviación de equilibrio: $\\sigma_{eq} = \\sqrt{\\sigma^2/(2\\theta)}$

    Parameters
    ----------
    spread : pd.Series
        Serie del spread estacionario ``Z_t``.
    dt : float
        Paso temporal en las mismas unidades que ``half_life``
        (1.0 para datos diarios).

    Returns
    -------
    OUParameters
        Parámetros continuos del OU ajustados y métricas auxiliares.
    """
    clean = spread.dropna().to_numpy()
    if clean.size < 3:
        raise ValueError("Se requieren al menos 3 observaciones para ajustar OU.")

    y = clean[1:]
    x = clean[:-1]
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    a, b = float(model.params[0]), float(model.params[1])

    if b >= 1:
        raise ValueError(
            "Coeficiente AR(1) >= 1: el spread no es estacionario. "
            "Verifica el test ADF."
        )
    if b <= 0:
        raise ValueError("Coeficiente AR(1) <= 0: oscilación determinista.")

    var_eps = float(np.var(model.resid, ddof=2))
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = np.sqrt((2 * theta * var_eps) / (1 - b ** 2))
    half_life = np.log(2) / theta
    sigma_eq = np.sqrt(sigma ** 2 / (2 * theta))

    return OUParameters(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        half_life=float(half_life),
        sigma_eq=float(sigma_eq),
        r_squared=float(model.rsquared),
    )


def simulate_ou_process(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    dt: float,
) -> np.ndarray:
    """Simula una trayectoria del proceso de Ornstein-Uhlenbeck.

    Discretización Euler-Maruyama:

    $$X_{t+\\Delta t} = X_t + \\theta(\\mu - X_t)\\,\\Delta t
      + \\sigma\\,\\sqrt{\\Delta t}\\,Z_t,$$

    donde $Z_t \\sim \\mathcal{N}(0, 1)$.

    Parameters
    ----------
    theta, mu, sigma : float
        Parámetros continuos del OU.
    x0 : float
        Valor inicial $X_0$.
    T : float
        Horizonte total en las mismas unidades que ``dt``.
    dt : float
        Paso de discretización.

    Returns
    -------
    np.ndarray
        Trayectoria simulada con ``T/dt`` puntos.
    """
    n = int(T / dt)
    x = np.zeros(n)
    x[0] = x0
    sqrt_dt = np.sqrt(dt)
    for t in range(1, n):
        dW = sqrt_dt * np.random.normal()
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * dW
    return x


def simulate_gbm(
    s0: float, mu: float, sigma: float, T: float, dt: float
) -> np.ndarray:
    """Simula un Movimiento Browniano Geométrico en escala lineal.

    $$S_t = S_0 \\exp\\left[\\left(\\mu - \\tfrac{\\sigma^2}{2}\\right) t
      + \\sigma\\,W_t\\right],$$

    donde $W_t$ es un movimiento Browniano estándar acumulado
    $\\sum_i \\sqrt{\\Delta t}\\,Z_i$.

    Parameters
    ----------
    s0 : float
        Precio inicial $S_0$.
    mu : float
        Drift anual del activo.
    sigma : float
        Volatilidad anual del activo.
    T : float
        Horizonte total en años.
    dt : float
        Paso de discretización en años (1/252 para datos diarios).

    Returns
    -------
    np.ndarray
        Serie de precios simulados con ``T/dt`` puntos.
    """
    n = int(T / dt)
    t = np.linspace(0.0, T, n)
    w = np.cumsum(np.random.standard_normal(size=n)) * np.sqrt(dt)
    log_s = (mu - 0.5 * sigma ** 2) * t + sigma * w
    return s0 * np.exp(log_s)


# =============================================================================
# Información mutua y clustering
# =============================================================================


def gaussian_mi_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Matriz de Información Mutua bajo hipótesis gaussiana.

    Para distribuciones gaussianas la MI se reduce a una transformación
    monótona de la correlación de Pearson:

    $$I(X;Y) = -\\tfrac{1}{2}\\,\\log\\bigl(1-\\rho^2\\bigr).$$

    Por tanto es funcionalmente equivalente a la correlación y no
    captura dependencias no lineales.  Se incluye por completitud y por
    compatibilidad con versiones anteriores.
    """
    if returns.empty:
        raise ValueError("El DataFrame de retornos está vacío.")
    rho = returns.corr(method="pearson").clip(lower=-0.99999, upper=0.99999)
    mi_arr = -0.5 * np.log(1 - rho.to_numpy() ** 2)
    np.fill_diagonal(mi_arr, np.nan)
    return pd.DataFrame(mi_arr, index=rho.index, columns=rho.columns)


def ksg_mi_matrix(
    returns: pd.DataFrame, k_neighbors: int = 3, seed: int = SEED
) -> pd.DataFrame:
    """Matriz de Información Mutua con el estimador KSG (k-NN).

    Para cada par de activos $(X_i, X_j)$ se estiman dos valores
    $\\widehat{MI}(X_i;X_j)$ y $\\widehat{MI}(X_j;X_i)$ y se devuelve
    la media para mitigar asimetrías de la búsqueda k-NN.  Series con
    varianza cero se omiten y la diagonal queda ``NaN``.

    Coste: $\\mathcal{O}(N^2)$ estimaciones de pares.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos limpios (sin ``NaN``).
    k_neighbors : int
        Parámetro ``k`` del estimador (3 por defecto, balance sesgo-
        varianza).
    seed : int
        Semilla aleatoria del estimador (afecta al jittering).
    """
    if returns is None or returns.empty:
        raise ValueError("El DataFrame de retornos está vacío.")

    clean = returns.dropna(how="any")
    if clean.empty:
        raise ValueError("No quedan observaciones tras limpiar NaN.")

    assets = clean.columns
    data = clean.to_numpy()
    n = len(assets)
    mi = np.full((n, n), np.nan)

    print(f"Calculando matriz MI (KSG) para {n} activos...")
    for i in range(n):
        for j in range(i + 1, n):
            x, y = data[:, i], data[:, j]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            xi = x.reshape(-1, 1)
            yi = y.reshape(-1, 1)
            mi_xy = mutual_info_regression(
                xi, y, n_neighbors=k_neighbors, random_state=seed
            )[0]
            mi_yx = mutual_info_regression(
                yi, x, n_neighbors=k_neighbors, random_state=seed
            )[0]
            sym = (mi_xy + mi_yx) / 2.0
            mi[i, j] = sym
            mi[j, i] = sym
    return pd.DataFrame(mi, index=assets, columns=assets)


# =============================================================================
# Caché de matrices costosas
# =============================================================================


def _hash_returns(returns: pd.DataFrame) -> str:
    """Hash determinista (sha256) del contenido de los retornos."""
    arr = np.ascontiguousarray(returns.to_numpy(dtype=np.float64))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return digest[:16]


def _ksg_cache_key(
    returns: pd.DataFrame,
    k_neighbors: int,
    seed: int,
) -> dict:
    """Metadatos usados como clave de caché del KSG MI."""
    index = pd.DatetimeIndex(returns.index)
    return {
        "version": KSG_CACHE_VERSION,
        "estimator": "ksg",
        "k_neighbors": int(k_neighbors),
        "seed": int(seed),
        "start": str(index.min().date()) if len(index) else "",
        "end": str(index.max().date()) if len(index) else "",
        "n_obs": int(returns.shape[0]),
        "n_assets": int(returns.shape[1]),
        "assets": list(map(str, returns.columns)),
        "returns_hash": _hash_returns(returns),
    }


def _ksg_cache_filename(meta: dict) -> str:
    """Nombre determinista del archivo de caché."""
    h = hashlib.sha256(repr(sorted(meta.items())).encode()).hexdigest()
    return (
        f"ksg_k{meta['k_neighbors']}_seed{meta['seed']}"
        f"_{meta['n_assets']}x{meta['n_obs']}_{h[:12]}.pkl"
    )


def cached_ksg_mi_matrix(
    returns: pd.DataFrame,
    k_neighbors: int = 3,
    seed: int = SEED,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Matriz MI KSG con caché en disco bajo ``cache_dir``.

    Si ya existe una matriz calculada con los mismos parámetros y los mismos
    datos, se carga directamente sin volver a calcular el estimador KSG.  La
    caché se invalida automáticamente al cambiar:

    - las fechas o el número de observaciones;
    - los activos o el hash del contenido;
    - ``k_neighbors`` o ``seed``;
    - la versión del esquema de caché.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos limpios con índice temporal y un ticker por columna.
    k_neighbors : int
        Parámetro ``k`` del estimador KSG.
    seed : int
        Semilla aleatoria del estimador.
    cache_dir : Path, opcional
        Carpeta de caché.  Por defecto ``DATA_DIR / 'pair_trading' / 'mi' / 'ksg'``.
    force : bool
        Si es ``True`` recalcula y sobrescribe la entrada de caché existente.
    """
    if returns is None or returns.empty:
        raise ValueError("El DataFrame de retornos está vacío.")
    clean = returns.dropna(how="any")
    if clean.empty:
        raise ValueError("No quedan observaciones tras limpiar NaN.")

    if cache_dir is None:
        cache_dir = DATA_DIR / "pair_trading" / "mi" / "ksg"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = _ksg_cache_key(clean, k_neighbors, seed)
    matrix_path = cache_dir / _ksg_cache_filename(meta)
    meta_path = matrix_path.with_suffix(".json")

    if matrix_path.exists() and meta_path.exists() and not force:
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                stored = json.load(fh)
            if all(stored.get(k) == v for k, v in meta.items()):
                cached = pd.read_pickle(matrix_path)
                if (
                    list(cached.index) == meta["assets"]
                    and list(cached.columns) == meta["assets"]
                ):
                    print(
                        f"[caché] Matriz MI KSG cargada desde {matrix_path.name} "
                        f"({meta['n_assets']}×{meta['n_assets']})"
                    )
                    return cached
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    matrix = ksg_mi_matrix(clean, k_neighbors=k_neighbors, seed=seed)

    payload = {**meta, "created_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(
        delete=False, dir=cache_dir, suffix=".pkl.tmp",
    ) as tmp_matrix, tempfile.NamedTemporaryFile(
        delete=False, dir=cache_dir, suffix=".json.tmp", mode="w",
    ) as tmp_meta:
        pd.to_pickle(matrix, tmp_matrix.name)
        json.dump(payload, tmp_meta, indent=2, sort_keys=True)
        tmp_meta.flush()
        os.replace(tmp_matrix.name, str(matrix_path))
        os.replace(tmp_meta.name, str(meta_path))

    print(
        f"[caché] Matriz MI KSG guardada en {matrix_path.relative_to(cache_dir)}"
    )
    return matrix


def top_pairs(
    matrix: pd.DataFrame,
    n: int = 20,
    ascending: bool = False,
    issuer_map: pd.Series | None = None,
) -> list[tuple[str, str, float]]:
    """Devuelve las ``n`` parejas más (o menos) acopladas de una matriz.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matriz cuadrada con tickers en índice y columnas.
    n : int
        Número de parejas a devolver.
    ascending : bool
        Si ``True`` devuelve los pares con valores más bajos.
    issuer_map : pd.Series, opcional
        Mapa ``ticker → emisor``.  Cuando se proporciona, se descartan
        parejas formadas por el mismo emisor (ej. ``GOOGL``/``GOOG``).
    """
    pairs: list[tuple[str, str, float]] = []
    cols = matrix.columns
    issuers = issuer_map if issuer_map is None else issuer_map.reindex(cols)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            val = matrix.iloc[i, j]
            if np.isnan(val):
                continue
            if issuers is not None:
                a_issuer = issuers.iloc[i]
                b_issuer = issuers.iloc[j]
                if (pd.notna(a_issuer) and pd.notna(b_issuer)
                        and a_issuer == b_issuer):
                    continue
            pairs.append((cols[i], cols[j], float(val)))
    pairs.sort(key=lambda t: t[2], reverse=not ascending)
    return pairs[:n]


# =============================================================================
# Evaluación sectorial de clusters
# =============================================================================

#: Sectores GICS con su denominación en castellano.  Wikipedia publica la
#: tabla en inglés; la tesis está escrita en español, así que las etiquetas
#: se traducen antes de calcular métricas, dibujar figuras y exportar
#: tablas.  Traducir no altera ningún resultado: las métricas de
#: alineación sólo dependen de la partición, no de los nombres.
GICS_SECTOR_ES: dict[str, str] = {
    "Communication Services": "Servicios de Comunicación",
    "Consumer Discretionary": "Consumo Discrecional",
    "Consumer Staples": "Consumo Básico",
    "Energy": "Energía",
    # GICS renombró el sector; ambas denominaciones deben colapsar en la
    # misma etiqueta o la partición sectorial se fragmentaría en dos.
    "Financials": "Finanzas",
    "Financial Services": "Finanzas",
    "Health Care": "Salud",
    "Industrials": "Industriales",
    "Information Technology": "Tecnología de la Información",
    "Materials": "Materiales",
    "Real Estate": "Inmobiliario",
    "Utilities": "Servicios Públicos",
}


def translate_gics_sectors(sectors: pd.Series) -> pd.Series:
    """Traduce a castellano los nombres de sector GICS de una serie.

    Parameters
    ----------
    sectors : pd.Series
        Serie indexada por ticker con el sector GICS como valor.

    Returns
    -------
    pd.Series
        Misma serie con los sectores traducidos.  Los valores que no están
        en :data:`GICS_SECTOR_ES` se conservan tal cual, de modo que un
        sector nuevo no rompe el análisis y queda visible en las tablas.
    """
    traducidos = sectors.map(lambda s: GICS_SECTOR_ES.get(s, s))
    traducidos.name = sectors.name
    return traducidos


def unmapped_gics_sectors(sectors: pd.Series) -> list[str]:
    """Sectores presentes en la serie que no están en :data:`GICS_SECTOR_ES`.

    Sirve para detectar que GICS haya añadido un sector y la traducción se
    haya quedado obsoleta, en lugar de que la etiqueta en inglés aparezca
    sin avisar en la tesis.
    """
    return sorted(
        {
            str(s)
            for s in pd.unique(sectors.dropna())
            if str(s) not in GICS_SECTOR_ES
        }
    )


def align_cluster_sector_labels(
    cluster_labels: pd.Series,
    sector_labels: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Alinea series de cluster y sector por ticker y devuelve pares limpios.

    Parameters
    ----------
    cluster_labels : pd.Series
        Índice = ticker, valores = etiqueta de clúster.
    sector_labels : pd.Series
        Índice = ticker, valores = sector GICS.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Ambas series indexadas por los tickers comunes, excluyendo ``NaN``.
    """
    common = cluster_labels.index.intersection(sector_labels.index)
    cl = cluster_labels.reindex(common)
    se = sector_labels.reindex(common)
    valid = cl.notna() & se.notna()
    return cl[valid], se[valid]


def cluster_sector_purity(
    cluster_labels: pd.Series,
    sector_labels: pd.Series,
) -> pd.DataFrame:
    """Tabla de contingencia y pureza por clúster.

    La pureza de un cluster ``c`` se define como

    $$\\mathrm{purity}(c) = \\frac{\\max_s\\, n_{c,s}}{n_c},$$

    donde $n_{c,s}$ es el número de activos del sector ``s`` en el
    cluster ``c``.  Toma valores en $[0, 1]$; cuanto más alto, más
    homogéneo es el cluster en términos sectoriales.

    Returns
    -------
    pd.DataFrame
        Filas = sectores, columnas = clusters.  Incluye dos filas
        adicionales: ``cluster_size`` (número de activos del cluster) y
        ``cluster_purity`` (pureza del cluster).
    """
    cl, se = align_cluster_sector_labels(cluster_labels, sector_labels)
    contingency = pd.crosstab(se, cl)
    cluster_sizes = contingency.sum(axis=0)
    dominant_counts = contingency.max(axis=0)
    purity = (dominant_counts / cluster_sizes).fillna(0)
    contingency.loc["cluster_size"] = cluster_sizes
    contingency.loc["cluster_purity"] = purity
    return contingency


def normalized_entropy(series: pd.Series) -> float:
    """Entropía de Shannon normalizada por $\\log S$, donde $S$ es el
    número de categorías observadas.

    $$H_c = \\frac{-\\sum_s p(s|c)\\,\\log p(s|c)}{\\log S_c}.$$

    Toma valores en $[0, 1]$.  $H_c = 0$ indica que el cluster contiene
    un único sector; $H_c = 1$ indica que los sectores están
    distribuidos uniformemente dentro del cluster.
    """
    counts = series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    entropy = -np.sum(counts * np.log(counts + np.finfo(float).eps))
    return float(entropy / np.log(len(counts)))


def hhi(series: pd.Series) -> float:
    """Índice Herfindahl-Hirschman de concentración sectorial.

    $$\\mathrm{HHI}_c = \\sum_s p(s|c)^2.$$

    Toma valores en $[1/S_c, 1]$; cuanto más alto, más concentrado está
    el cluster en un único sector.  Equivale a una pureza de orden 2.
    """
    shares = series.value_counts(normalize=True)
    return float(np.sum(shares ** 2))


@dataclass(frozen=True)
class ClusterSectorReport:
    """Resultados de la evaluación de clusters contra sectores GICS.

    Attributes
    ----------
    contingency : pd.DataFrame
        Tabla sector × cluster con filas extra ``cluster_size`` y
        ``cluster_purity``.
    per_cluster : pd.DataFrame
        Métricas detalladas por cluster (tamaño, sector dominante, purity,
        entropía normalizada, HHI).
    weighted_purity : float
        Pureza global ponderada por el tamaño de cada cluster.
    normalized_entropy : float
        Entropía sectorial media (ponderada por tamaño).
    mean_hhi : float
        HHI medio (ponderado por tamaño).
    ari : float
        Adjusted Rand Index.
    ami : float
        Adjusted Mutual Information.
    homogeneity : float
        $\\in [0, 1]$.  Cada cluster contiene un único sector.
    completeness : float
        $\\in [0, 1]$.  Cada sector queda concentrado en pocos clusters.
    v_measure : float
        Media armónica de homogeneity y completeness.
    """

    contingency: pd.DataFrame
    per_cluster: pd.DataFrame
    weighted_purity: float
    normalized_entropy: float
    mean_hhi: float
    ari: float
    ami: float
    homogeneity: float
    completeness: float
    v_measure: float


def evaluate_cluster_sector_alignment(
    cluster_labels: pd.Series,
    sector_labels: pd.Series,
) -> ClusterSectorReport:
    """Evalúa qué tan bien los clusters reflejan la estructura sectorial.

    Combina métricas propias (purity, entropía normalizada, HHI) con
    métricas estándar de clustering (``sklearn.metrics``):

    - **Adjusted Rand Index (ARI)**: similitud entre asignaciones,
      corregida por azar.  Rango $[-1, 1]$, donde $1$ indica coincidencia
      perfecta.
    - **Adjusted Mutual Information (AMI)**: información mutua entre
      las dos particiones, normalizada por azar.  Rango $[0, 1]$.
    - **Homogeneity**: cada cluster contiene principalmente un sector.
    - **Completeness**: cada sector queda concentrado en pocos clusters.
    - **V-measure**: media armónica de homogeneity y completeness.

    Parameters
    ----------
    cluster_labels : pd.Series
        Índice = ticker, valores = etiqueta de clúster.
    sector_labels : pd.Series
        Índice = ticker, valores = sector GICS.

    Returns
    -------
    ClusterSectorReport
        Contingencia, métricas por clúster y métricas globales.
    """
    cl, se = align_cluster_sector_labels(cluster_labels, sector_labels)
    if cl.empty:
        raise ValueError("No hay tickers comunes entre clusters y sectores.")

    contingency = cluster_sector_purity(cl, se)

    cluster_ids = sorted(cl.unique())
    sizes = []
    purities = []
    entropies = []
    hhis = []
    dominant_sectors = []

    for cid in cluster_ids:
        cluster_mask = cl == cid
        sector_slice = se[cluster_mask]
        size = int(cluster_mask.sum())
        sizes.append(size)
        dominant = sector_slice.mode()
        dominant_sectors.append(dominant.iloc[0] if not dominant.empty else "N/A")
        purities.append(float(contingency.loc["cluster_purity", cid]))
        entropies.append(normalized_entropy(sector_slice))
        hhis.append(hhi(sector_slice))

    per_cluster = pd.DataFrame(
        {
            "cluster": cluster_ids,
            "size": sizes,
            "dominant_sector": dominant_sectors,
            "purity": purities,
            "norm_entropy": entropies,
            "hhi": hhis,
        }
    ).set_index("cluster")

    weighted_purity = float(np.average(purities, weights=sizes))
    mean_entropy = float(np.average(entropies, weights=sizes))
    mean_hhi_val = float(np.average(hhis, weights=sizes))

    ari = float(adjusted_rand_score(se, cl))
    ami = float(adjusted_mutual_info_score(se, cl))
    homogeneity = float(homogeneity_score(se, cl))
    completeness = float(completeness_score(se, cl))
    v_measure = float(v_measure_score(se, cl))

    return ClusterSectorReport(
        contingency=contingency,
        per_cluster=per_cluster,
        weighted_purity=weighted_purity,
        normalized_entropy=mean_entropy,
        mean_hhi=mean_hhi_val,
        ari=ari,
        ami=ami,
        homogeneity=homogeneity,
        completeness=completeness,
        v_measure=v_measure,
    )


def sector_average_matrix(
    matrix: pd.DataFrame,
    sector_labels: pd.Series,
    operation: str = "mean",
) -> pd.DataFrame:
    """Agrega una matriz de activos a una matriz sector × sector.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matriz cuadrada con tickers en índice y columnas.
    sector_labels : pd.Series
        Serie ticker → sector.
    operation : {"mean", "median", "min", "max"}
        Reducción aplicada dentro de cada bloque sectorial.
    """
    common = matrix.index.intersection(sector_labels.index).intersection(
        matrix.columns
    )
    if len(common) == 0:
        raise ValueError("No hay tickers comunes entre la matriz y los sectores.")

    sub_matrix = matrix.loc[common, common]
    sub_sectors = sector_labels.reindex(common)

    sectors = sorted(sub_sectors.unique())
    out = pd.DataFrame(np.nan, index=sectors, columns=sectors)

    func = getattr(np, operation)
    for s_i in sectors:
        idx_i = sub_sectors.index[sub_sectors == s_i]
        for s_j in sectors:
            idx_j = sub_sectors.index[sub_sectors == s_j]
            block = sub_matrix.loc[idx_i, idx_j].to_numpy().ravel()
            block = block[~np.isnan(block)]
            if s_i == s_j:
                # Excluir la diagonal dentro del bloque diagonal
                block = block[block < 1.0 - 1e-12]
            if len(block) > 0:
                out.loc[s_i, s_j] = float(func(block))
    return out


def hierarchical_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int,
    method: str = "ward",
) -> tuple[np.ndarray, list[int]]:
    """Clustering jerárquico de Ward sobre una matriz de distancias.

    El método de Ward minimiza el incremento de varianza intra-cluster
    al fusionar pares de clusters; es el más adecuado cuando la
    distancia cumple la desigualdad triangular (como ``sqrt(1-rho)``
    o las distancias euclidianas).

    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz cuadrada simétrica de distancias.
    n_clusters : int
        Número de clusters a extraer.
    method : str
        Método de linkage (por defecto ``"ward"``).

    Returns
    -------
    labels : np.ndarray
        Etiquetas de clúster 1-indexadas (``scipy.cluster.hierarchy.fcluster``).
    order : list[int]
        Orden óptimo de las hojas del dendrograma (``leaves_list``).
    """
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0, None)
    condensed = squareform(distance_matrix, checks=False)
    z = linkage(condensed, method=method)
    labels = fcluster(z, t=n_clusters, criterion="maxclust")
    order = leaves_list(z).tolist()
    return labels, order


def correlation_distance(
    corr: pd.DataFrame, use_sqrt: bool = True
) -> np.ndarray:
    """Convierte una matriz de correlación en distancia para clustering.

    Por defecto devuelve ``dist = sqrt(1 - rho)``.  Esta distancia
    cumple la desigualdad triangular, requisito del método de Ward.
    Con ``use_sqrt=False`` se devuelve simplemente ``1 - rho``.
    """
    distance = 1.0 - corr.to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0, None)
    if use_sqrt:
        distance = np.sqrt(distance)
    return distance


def mi_distance(
    mi_matrix: pd.DataFrame, fill_value: float | None = None,
) -> np.ndarray:
    """Convierte una matriz de Información Mutua en distancia para clustering.

    Para activos con MI alta la distancia debe ser pequeña.  Usamos la
    transformación

    $$d_{ij} = \\max(MI) - MI_{ij},$$

    donde el máximo se calcula sobre los valores válidos.  La diagonal
    queda a 0 (autodistancia nula).  Si ``fill_value`` es ``None`` se
    utiliza el máximo de la matriz.
    """
    if fill_value is None:
        fill_value = float(np.nanmax(mi_matrix.to_numpy()))
    filled = mi_matrix.copy()
    arr = filled.to_numpy().copy()
    np.fill_diagonal(arr, fill_value)
    filled = pd.DataFrame(arr, index=filled.index, columns=filled.columns)
    distance = fill_value - arr
    np.fill_diagonal(distance, 0.0)
    return np.clip(distance, 0, None)