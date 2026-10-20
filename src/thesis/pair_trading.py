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

    El resultado es una clave canónica que agrupa ``GOOGL Class A`` y
    ``GOOG Class C`` bajo la misma raíz ``alphabet``.
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


def load_log_returns(
    tickers: Sequence[str] | str,
    start: str = START,
    end: str = END,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
    min_valid_obs: int = 3,
) -> pd.DataFrame:
    """Descarga precios ``Adj Close`` y devuelve los log-retornos limpios.

    Reutiliza :func:`thesis.market_data.load_multiple_prices` para
    aprovechar la caché CSV.  El log-retorno es
    $r_t = \\log(P_t / P_{t-1})$.  Tras alinear fechas descarta tickers
    con menos de ``min_valid_obs`` observaciones válidas y elimina filas
    con ``NaN``.

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

    Returns
    -------
    pd.DataFrame
        Retornos logarítmicos con índice temporal y un ticker por columna.
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