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
from collections.abc import Sequence
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

from thesis.config import DATA_DIR, END, SEED, START
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

    Returns
    -------
    pd.DataFrame
        Columnas ``Symbol`` (normalizado para ``yfinance``), ``Security``
        (nombre del valor según la tabla oficial) y ``GICS Sector``.
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
    """Normaliza el nombre de un valor para identificar emisores duplicados."""
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
    """Filtra emisores con múltiples clases de acciones (ej. GOOGL/GOOG).

    Mantiene una sola fila por emisor utilizando la primera aparición en la
    tabla oficial (criterio determinista).  Excluye explícitamente pares de
    emisores conocidos que ``yfinance`` no resuelve con símbolos idénticos.

    Parameters
    ----------
    constituents : pd.DataFrame
        Salida de :func:`get_sp500_constituents`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``kept``: constituyentes filtrados con una fila por emisor y columna
        extra ``Issuer``.
        ``removed``: símbolos descartados y su símbolo representante
        (una fila por símbolo).
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
        Si ``True``, elimina símbolos duplicados del mismo emisor (ej.
        ``GOOGL``/``GOOG``).  Cuando se activa, devuelve además una tabla
        ``DataFrame`` con los símbolos descartados y su representante.

    Returns
    -------
    list[str] o tuple[list[str], pd.DataFrame]
        Tickers compatibles con ``yfinance``.  Si ``unique_issuers=True``
        devuelve también la tabla de símbolos descartados.
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


def load_log_returns(
    tickers: Sequence[str] | str,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
    min_valid_obs: int = 3,
) -> pd.DataFrame:
    """Descarga precios ``Adj Close`` y devuelve los log-retornos limpios.

    Reutiliza :func:`thesis.market_data.load_multiple_prices` para aprovechar
    la caché CSV.  Tras alinear fechas, descarta tickers con menos de
    ``min_valid_obs`` observaciones válidas y elimina filas con ``NaN``.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    prices = load_multiple_prices(list(tickers), start, end, cache_dir, refresh)
    if prices.empty:
        raise ValueError("No se descargaron datos para el rango solicitado.")

    valid_cols = [
        col for col in prices.columns
        if prices[col].notna().sum() > min_valid_obs - 1
    ]
    if len(valid_cols) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 tickers válidos. Encontrados: {valid_cols}"
        )
    prices = prices[valid_cols]

    log_ret = np.log(prices / prices.shift(1)).dropna(how="any")
    if log_ret.empty:
        raise ValueError("No quedan observaciones tras el alineamiento.")
    return log_ret


# =============================================================================
# Hedge ratio y spread
# =============================================================================


@dataclass(frozen=True)
class HedgeResult:
    """Resultado de la regresión OLS log-lineal."""

    alpha: float
    beta: float
    r_squared: float
    model: object


@dataclass(frozen=True)
class ADFResult:
    """Resumen del test ADF aplicado al spread."""

    statistic: float
    pvalue: float
    is_stationary: bool


def compute_hedge_ratio(
    log_a: pd.Series, log_b: pd.Series
) -> HedgeResult:
    """Estima ``log(A) = alpha + beta * log(B) + epsilon`` por OLS."""
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
    """Test Augmented Dickey-Fuller sobre el spread."""
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
    """Descarga dos activos, ajusta el hedge ratio y devuelve el spread.

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
    """Parámetros continuos del proceso de Ornstein-Uhlenbeck."""

    theta: float
    mu: float
    sigma: float
    half_life: float
    sigma_eq: float
    r_squared: float


def fit_ou_process(spread: pd.Series, dt: float = 1.0) -> OUParameters:
    """Ajusta un OU a un spread estacionario mediante AR(1) → OU.

    El modelo discreto ``Z_t = a + b * Z_{t-1} + eps_t`` se traduce a los
    parámetros continuos ``(theta, mu, sigma)`` asumiendo ``dt`` constante.
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
    """Simula una trayectoria del proceso de Ornstein-Uhlenbeck (Euler-Maruyama)."""
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
    """Simula un Movimiento Browniano Geométrico en escala lineal."""
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

    Para retornos gaussianos la MI se reduce a ``-1/2 * log(1 - rho^2)``.
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
    """Matriz de Información Mutua con el estimador KSG (k-NN)."""
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

    Returns
    -------
    pd.DataFrame
        Filas = sectores, columnas = clusters, con una fila extra
        ``cluster_purity`` que indica la proporción del sector dominante.
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
    """Entropía de una variable categórica normalizada por el log del número de categorías."""
    counts = series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    entropy = -np.sum(counts * np.log(counts + np.finfo(float).eps))
    return float(entropy / np.log(len(counts)))


def hhi(series: pd.Series) -> float:
    """Índice Herfindahl-Hirschman de concentración."""
    shares = series.value_counts(normalize=True)
    return float(np.sum(shares ** 2))


@dataclass(frozen=True)
class ClusterSectorReport:
    """Resultados de la evaluación de clusters contra sectores GICS."""

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

    Returns
    -------
    labels : np.ndarray
        Etiquetas de clúster (1-indexadas, ``fcluster``).
    order : list[int]
        Orden óptimo de las hojas (índices ``leaves_list``).
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
    """Convierte una matriz de correlación en distancia para clustering."""
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

    Si ``fill_value`` es ``None`` se utiliza el máximo valor de la matriz
    para la diagonal (autodistancia nula tras la transformación).
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