"""Funciones generales para el análisis de predictibilidad de características.

Este módulo contiene únicamente utilidades reutilizables: carga de precios,
construcción de series, estimadores de correlación y visualizaciones.  No
asume ninguna característica concreta (momentum, volatilidad, ...); los
análisis específicos viven en ``momentum_analysis.py`` y
``volatility_analysis.py``.

Convención de nomenclatura:

* ``feature``           : la característica cuya predictibilidad se mide.
* ``target``            : la variable que se intenta predecir (p. ej. retorno
                          futuro acumulado).
* ``feature_op``        : operación móvil aplicada a la característica
                          (``'sum'`` para momentum, ``'std'`` para
                          volatilidad, ``'mean'``, ``'var'`` o un callable).
* ``target_op``         : operación móvil aplicada al objetivo.  Por
                          defecto la suma de los retornos futuros.
* ``feature_windows``   : colección de ventanas de lookback (en sesiones).
* ``target_windows``    : colección de ventanas de lookahead (en sesiones).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import yfinance as yf


# =============================================================================
# Parámetros y estilo por defecto
# =============================================================================

TICKER: str = "SPY"
START: str = "1993-01-01"
END: str = "2026-01-01"
TRADING_DAYS: int = 252
SEED: int = 42

# Caché de precios en la carpeta ``data`` del directorio principal del proyecto,
# independientemente del directorio de trabajo desde el que se ejecute el script.
# Puede sobreescribirse con la variable de entorno ``THESIS_DATA_DIR``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.environ.get("THESIS_DATA_DIR", _PROJECT_ROOT / "data"))

# Reproducibilidad para consumidores del estado global de numpy.
np.random.seed(SEED)

FIGSIZE = (7.0, 3.5)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "legend.frameon": False,
})


# =============================================================================
# Carga de precios
# =============================================================================

def load_single_price(ticker: str = TICKER,
                      start: str = START, end: str = END,
                      cache_dir: Path = DATA_DIR,
                      refresh: bool = False) -> pd.Series:
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


def load_multiple_prices(tickers: Sequence[str],
                          start: str = START, end: str = END,
                          cache_dir: Path = DATA_DIR,
                          refresh: bool = False) -> pd.DataFrame:
    """Descarga cada ticker, persiste su CSV y devuelve un frame combinado."""
    series_list = [
        s for s in (load_single_price(t, start, end, cache_dir, refresh)
                    for t in tickers)
        if not s.empty
    ]
    return pd.concat(series_list, axis=1) if series_list else pd.DataFrame()


# =============================================================================
# Construcción de series
# =============================================================================

def log_returns(price: pd.Series) -> pd.Series:
    """Retorno logarítmico diario de ``price``."""
    return np.log(price / price.shift(1)).dropna()


def simple_returns(price: pd.Series) -> pd.Series:
    """Retorno simple diario de ``price``."""
    return (price / price.shift(1) - 1.0).dropna()


def rolling_forward_window(series: pd.Series, window: int) -> pd.Series:
    """Suma de las ``window`` observaciones futuras, anclada al día actual.

    ``series.rolling(window).sum().shift(-window)`` produce la suma de los
    próximos ``window`` valores terminando en ``t`` y desplazada hacia atrás
    para que el resultado esté alineado con el instante actual.
    """
    return series.rolling(window).sum().shift(-window)


RollingOp = Union[str, Callable[[pd.Series], float]]


def apply_window(
    series: pd.Series,
    window: float,
    operation: RollingOp,
    window_type: str = "fixed",
    ewm_adjust: bool = False,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Reduce ``series`` sobre una ventana móvil fija o exponencial.

    Parámetros
    ----------
    series : pd.Series
        Serie de entrada.
    window : float
        Tamaño de la ventana en sesiones.  Para ``window_type='fixed'``
        es el tamaño del rolling; para ``window_type='exponential'`` es
        el ``span`` pasado a :func:`pandas.Series.ewm`.
    operation : str o callable
        Reducción aplicada.  Con ``window_type='exponential'`` sólo se
        aceptan cadenas soportadas por ``Series.ewm`` (``'mean'``,
        ``'std'``, ``'var'``); con ``window_type='fixed'`` se aceptan
        además callables, que reciben la sub-serie móvil y devuelven un
        escalar.
    window_type : {'fixed', 'exponential'}
        Tipo de ventana.
    ewm_adjust : bool
        Si es ``True``, usa la forma con corrección de sesgo de
        ``Series.ewm`` (``adjust=True``).  Por defecto es ``False``, la
        forma recursiva equivalente a ``y_t = (1-α)·y_{t-1} + α·x_t``,
        que permite actualizar la serie de forma incremental sin
        recalcular toda la historia.
    min_periods : int, opcional
        Mínimo de observaciones requeridas.  Por defecto es ``window``
        para ventanas fijas y ``0`` para exponenciales.
    """
    if window_type == "fixed":
        window_i = int(window)
        mp = int(min_periods) if min_periods is not None else window_i
        rolling = series.rolling(window_i, min_periods=mp)
        if callable(operation):
            return rolling.agg(operation)
        method = getattr(rolling, operation, None)
        if method is None:
            raise ValueError(
                f"Operación móvil desconocida '{operation}'. Usa una cadena "
                "soportada por pd.Series.rolling ('sum', 'mean', 'std', "
                "'var', ...) o una función."
            )
        return method()

    if window_type == "exponential":
        if callable(operation):
            raise ValueError(
                "Las ventanas exponenciales no soportan operaciones callables; "
                "usa 'mean', 'std' o 'var'."
            )
        mp = int(min_periods) if min_periods is not None else 0
        ewm = series.ewm(span=float(window), adjust=ewm_adjust, min_periods=mp)
        method = getattr(ewm, operation, None)
        if method is None:
            raise ValueError(
                f"Operación '{operation}' no soportada por ewm. Usa 'mean', "
                "'std' o 'var'."
            )
        return method()

    raise ValueError(
        f"window_type debe ser 'fixed' o 'exponential', se recibió "
        f"'{window_type}'."
    )


def _apply_rolling(series: pd.Series, window: int, op: RollingOp) -> pd.Series:
    """Wrapper retro-compatible: ventana fija equivalente a ``apply_window``."""
    return apply_window(series, window, op, window_type="fixed")


# =============================================================================
# Estimador principal: matriz de correlación lookback × lookahead
# =============================================================================

def compute_correlation_matrix(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_windows: Optional[Sequence[int]] = None,
    target_windows: Optional[Sequence[int]] = None,
    max_lookback: int = 504,
    max_lookahead: int = 504,
    steps: int = 150,
    overlap: bool = True,
    feature_op: RollingOp = "sum",
    feature_window_type: str = "fixed",
    feature_transform: Optional[Callable[[pd.Series, pd.Series], pd.Series]] = None,
    target_op: RollingOp = "sum",
    correlation: str = "spearman",
    min_valid: int = 3,
    ewm_adjust: bool = False,
    verbose: bool = True,
    n_bootstrap: int = 1000,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matriz de correlación entre la característica y el objetivo futuro.

    La característica se reduce sobre cada ventana de lookback con
    ``feature_op`` (``'sum'`` para momentum, ``'std'`` para volatilidad,
    ``'mean'`` para drift promedio, etc.).  El objetivo siempre es el
    retorno futuro acumulado sobre el lookahead (``target_op='sum'`` por
    defecto, sin admitir ventanas exponenciales).  La métrica de
    correlación puede ser ``'spearman'`` o ``'pearson'``.

    Si ``feature_windows`` o ``target_windows`` se proporcionan, se usan
    directamente; si no, se genera una rejilla con
    ``(max_lookback + max_lookahead) // steps`` pasos.

    Parámetros de ventana
    ---------------------
    feature_window_type : {'fixed', 'exponential'}
        Tipo de ventana aplicada a la característica.  Con
        ``'exponential'`` el ``span`` se pasa a :func:`pandas.Series.ewm`
        y ``feature_op`` debe ser una cadena (``'mean'``, ``'std'``,
        ``'var'``).  Con ``'fixed'`` se admite además un callable.
    feature_transform : callable, opcional
        Función ``(windowed, base) -> feature`` aplicada tras el rolling
        o ewm.  Útil para combinar la serie original con su media móvil,
        p. ej. ``lambda w, s: s - w`` para ``s - MA(s)``.
    ewm_adjust : bool
        Si es ``True`` aplica la corrección de sesgo de EWM; por defecto
        es ``False`` para permitir actualizaciones incrementales.
    verbose : bool
        Si es ``True``, imprime al final el máximo ``|ρ|`` de la matriz,
        los ``τ`` correspondientes y un intervalo de confianza bootstrap
        (con y sin solapamiento).
    n_bootstrap : int
        Número de remuestras bootstrap para el intervalo.
    bootstrap_alpha : float
        Nivel de significancia bilateral para el IC (por defecto 0.05 →
        IC 95 %).
    bootstrap_seed : int, opcional
        Semilla del generador aleatorio del bootstrap; por defecto usa
        :data:`SEED` para resultados reproducibles.
    """
    if feature_windows is None or target_windows is None:
        step = max(1, (max_lookback + max_lookahead) // steps)
        lookbacks = np.arange(step, max_lookback + step, step)
        lookaheads = np.arange(step, max_lookahead + step, step)
    else:
        lookbacks = np.asarray(sorted(set(feature_windows)))
        lookaheads = np.asarray(sorted(set(target_windows)))

    corr_matrix = np.full((len(lookaheads), len(lookbacks)), np.nan)

    # Precomputación: la reducción móvil de la característica depende sólo
    # del lookback, no del lookahead, así que se cachea por columna.
    feature_cache: Dict[int, pd.Series] = {}
    for j, l in enumerate(lookbacks):
        windowed = apply_window(
            feature_series, float(l), feature_op,
            window_type=feature_window_type, ewm_adjust=ewm_adjust,
        )
        if feature_transform is None:
            feature_cache[j] = windowed
        else:
            feature_cache[j] = feature_transform(windowed, feature_series)

    if correlation == "spearman":
        corr_func = stats.spearmanr
    elif correlation == "pearson":
        corr_func = stats.pearsonr
    else:
        raise ValueError(
            f"Correlación desconocida '{correlation}'. Usa 'spearman' o 'pearson'."
        )

    for i, f in enumerate(lookaheads):
        # El objetivo es siempre el retorno futuro acumulado sobre el
        # lookahead; se mantiene como ventana fija.
        fut = target_series.rolling(int(f)).sum().shift(-int(f))

        for j in range(len(lookbacks)):
            feature = feature_cache[j]
            l = int(lookbacks[j])
            valid_idx = ~(feature.isna() | fut.isna())

            if valid_idx.sum() > min_valid - 1:
                shift_step = 1 if overlap else min(l, int(f))
                valid_positions = np.where(valid_idx.values)[0][::shift_step]
                x = feature.values[valid_positions]
                y = fut.values[valid_positions]
                corr, _ = corr_func(x, y)
                corr_matrix[i][j] = corr

    if verbose:
        _print_max_correlation_diagnostic(
            corr_matrix, lookbacks, lookaheads,
            feature_cache, target_series, corr_func,
            n_bootstrap=n_bootstrap,
            alpha=bootstrap_alpha,
            seed=SEED if bootstrap_seed is None else bootstrap_seed,
        )

    return corr_matrix, lookbacks, lookaheads


# =============================================================================
# Bootstrap del coeficiente máximo
# =============================================================================

def _corr_value(corr_func, x, y) -> float:
    """Devuelve el escalar de correlación tanto para tuplas como para
    objetos ``CorrelationResult`` (scipy ≥ 1.10)."""
    res = corr_func(x, y)
    return float(res.statistic if hasattr(res, "statistic") else res[0])


def _bootstrap_correlation(x: np.ndarray,
                            y: np.ndarray,
                            corr_func,
                            n_bootstrap: int = 1000,
                            alpha: float = 0.05,
                            seed: Optional[int] = None,
                            ) -> Tuple[float, Tuple[float, float]]:
    """Intervalo de confianza bootstrap (percentiles) para una correlación.

    Se remuestrea con reemplazo ``n_bootstrap`` veces y se calcula el
    percentil ``α/2`` y ``1 - α/2`` sobre la distribución bootstrap.
    Devuelve ``(point, (ci_low, ci_high))``.  Si los datos son
    insuficientes o todas las remuestras degeneran en varianza cero,
    devuelve ``(nan, (nan, nan))``.
    """
    n = len(x)
    if n < 3:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.full(n_bootstrap, np.nan)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0.0 or np.std(yb) == 0.0:
            continue
        try:
            boot[b] = _corr_value(corr_func, xb, yb)
        except Exception:
            continue
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return np.nan, (np.nan, np.nan)
    point = _corr_value(corr_func, x, y)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, (lo, hi)


def _print_max_correlation_diagnostic(
    corr_matrix: np.ndarray,
    lookbacks: np.ndarray,
    lookaheads: np.ndarray,
    feature_cache: Dict[int, pd.Series],
    target_series: pd.Series,
    corr_func,
    n_bootstrap: int,
    alpha: float,
    seed: Optional[int],
) -> None:
    """Imprime la celda con máximo |ρ|, sus τ y bootstrap con/sin overlap."""
    if np.all(np.isnan(corr_matrix)):
        print("\n[compute_correlation_matrix] No hay celdas válidas.")
        return

    abs_corr = np.abs(corr_matrix)
    i_max, j_max = np.unravel_index(np.nanargmax(abs_corr), abs_corr.shape)
    tau_p = int(lookbacks[j_max])
    tau_f = int(lookaheads[i_max])
    rho_matrix = float(corr_matrix[i_max, j_max])

    feature = feature_cache[j_max]
    fut = target_series.rolling(int(tau_f)).sum().shift(-int(tau_f))

    def _extract(shift_step: int) -> Tuple[np.ndarray, np.ndarray]:
        valid_idx = ~(feature.isna() | fut.isna())
        positions = np.where(valid_idx.values)[0][::shift_step]
        return feature.values[positions], fut.values[positions]

    x_with, y_with = _extract(1)
    rho_with, ci_with = _bootstrap_correlation(
        x_with, y_with, corr_func,
        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed,
    )

    shift_no = max(1, min(tau_p, tau_f))
    x_no, y_no = _extract(shift_no)
    rho_no, ci_no = _bootstrap_correlation(
        x_no, y_no, corr_func,
        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed,
    )

    pct = 100 * (1 - alpha)
    print(
        "\n[compute_correlation_matrix] Máximo |ρ| en la matriz:\n"
        f"  τ_p (lookback)  : {tau_p} sesiones\n"
        f"  τ_f (lookahead) : {tau_f} sesiones\n"
        f"  ρ matriz         : {rho_matrix:+.4f}\n"
        f"  ρ (con overlap)    : {rho_with:+.4f}   "
        f"IC {pct:.0f}%: [{ci_with[0]:+.4f}, {ci_with[1]:+.4f}]   "
        f"(n={len(x_with)}, bootstrap={n_bootstrap})\n"
        f"  ρ (sin overlap)    : {rho_no:+.4f}   "
        f"IC {pct:.0f}%: [{ci_no[0]:+.4f}, {ci_no[1]:+.4f}]   "
        f"(n={len(x_no)}, bootstrap={n_bootstrap})"
    )


# =============================================================================
# Estimador cross-sectional
# =============================================================================

def compute_cross_sectional_correlation(
    log_returns_df: pd.DataFrame,
    windows: Sequence[int],
    feature_df: Optional[pd.DataFrame] = None,
    feature_op: RollingOp = "sum",
    feature_window_type: str = "fixed",
    feature_transform: Optional[Callable[[pd.Series, pd.Series], pd.Series]] = None,
    target_op: RollingOp = "sum",
    correlation: str = "pearson",
    min_valid_n: int = 50,
    overlap: bool = False,
    shift_lookahead_by_lookback: bool = False,
    ewm_adjust: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Correlación cruzada entre activo y horizonte temporal.

    Para cada activo de ``log_returns_df`` se calcula la correlación entre
    la característica móvil (``feature_op`` aplicada al input de la
    característica) y el objetivo futuro (``target_op`` aplicada al input
    del objetivo) para cada ventana de ``windows``.  El resultado se
    devuelve como un ``DataFrame`` indexado por ventana con una columna
    por activo, junto con el promedio cross-sectional.

    Parámetros
    ----------
    log_returns_df : pd.DataFrame
        Series usadas como objetivo, con un activo por columna.  Por
        convención son retornos (logarítmicos) porque el objetivo por
        defecto es la suma de retornos futuros.
    windows : secuencia de int
        Ventanas (en sesiones) sobre las que se evalúa la predictibilidad.
    feature_df : pd.DataFrame, opcional
        Series usadas como input de la característica.  Si es ``None``
        (por defecto), se reutiliza ``log_returns_df``; útil cuando la
        característica se computa sobre precios (p. ej. distancia a la
        media móvil) y el objetivo sobre retornos.  Debe compartir
        columnas con ``log_returns_df``.
    feature_op, target_op : str o callable
        Reducciones móviles aplicadas a la característica y al objetivo.
    feature_window_type : {'fixed', 'exponential'}
        Tipo de ventana aplicada a la característica (análogo a
        :func:`compute_correlation_matrix`).
    feature_transform : callable, opcional
        Función ``(windowed, base) -> feature`` aplicada tras el rolling
        o ewm de la característica.
    correlation : {'pearson', 'spearman'}
        Métrica de correlación.
    min_valid_n : int
        Número mínimo de observaciones válidas para aceptar una estimación.
    overlap : bool
        Si ``False``, las observaciones se submuestrean con paso ``window``.
    shift_lookahead_by_lookback : bool
        Si ``True``, el objetivo comienza inmediatamente después de que
        termina el lookback (``shift(-lag).rolling(lag).sum()``); si no, el
        objetivo se calcula con la convención habitual
        (``rolling_forward_window``).
    ewm_adjust : bool
        Si es ``True`` aplica la corrección de sesgo de EWM; por defecto
        es ``False`` para permitir actualizaciones incrementales.
    """
    if correlation == "pearson":
        corr_func = stats.pearsonr
    elif correlation == "spearman":
        corr_func = stats.spearmanr
    else:
        raise ValueError(
            f"Correlación desconocida '{correlation}'. Usa 'pearson' o 'spearson'."
        )

    if feature_df is None:
        feature_df = log_returns_df

    corr_matrix = pd.DataFrame(index=list(windows),
                                columns=list(log_returns_df.columns),
                                dtype=float)

    for asset in log_returns_df.columns:
        target_series = log_returns_df[asset].dropna()
        if asset not in feature_df.columns:
            continue
        feature_series = feature_df[asset].dropna()
        for window in windows:
            windowed = apply_window(
                feature_series, float(window), feature_op,
                window_type=feature_window_type, ewm_adjust=ewm_adjust,
            )
            if feature_transform is None:
                feature = windowed
            else:
                feature = feature_transform(windowed, feature_series)
            if shift_lookahead_by_lookback:
                target = target_series.shift(-window).rolling(window).sum()
            else:
                target = _apply_rolling(target_series, window, target_op)
                if target_op == "sum":
                    target = target.shift(-window)
            valid_idx = ~(feature.isna() | target.isna())
            n_valid = int(valid_idx.sum())
            if n_valid >= min_valid_n:
                positions = np.where(valid_idx.values)[0]
                if not overlap:
                    positions = positions[::max(1, window)]
                x = feature.values[positions]
                y = target.values[positions]
                corr, _ = corr_func(x, y)
                corr_matrix.loc[window, asset] = corr

    avg_corr = corr_matrix.mean(axis=1, skipna=True)
    return corr_matrix, avg_corr


# =============================================================================
# Utilidades de ejes / ticks
# =============================================================================

def _year_ticks(windows: Sequence[int], trading_days: int = TRADING_DAYS,
                ticks_per_year: int = 4) -> Tuple[List[float], List[str]]:
    """Posiciones (centro de celda) y etiquetas en fracciones de año."""
    windows = np.asarray(windows)
    n_fractions = int(np.floor(windows[-1] / trading_days * ticks_per_year))
    fractions = np.arange(n_fractions + 1) / ticks_per_year

    positions, labels, used = [], [], set()
    for frac in fractions:
        j = 0 if frac == 0 else int(np.argmin(np.abs(windows - frac * trading_days)))
        if j in used:
            continue
        used.add(j)
        positions.append(j + 0.5)
        labels.append(f"{frac:.2f}")

    return positions, labels


# =============================================================================
# Visualización
# =============================================================================

def plot_correlation_matrix(corr_matrix: np.ndarray,
                            lookbacks: Sequence[int],
                            lookaheads: Sequence[int],
                            title: str = "Rank Correlation by Lag",
                            xlabel: str = r"$\tau_p$ (años)",
                            ylabel: str = r"$\tau_f$ (años)",
                            trading_days: int = TRADING_DAYS,
                            ticks_per_year: int = 4,
                            cbar_label: str = "Correlación",
                            cmap: str = "coolwarm") -> None:
    """Heatmap de la matriz de correlación con ticks en fracciones de año."""
    plt.figure(figsize=(10 / 2, 8 / 2), dpi=300)
    ax = sns.heatmap(corr_matrix, cmap=cmap, center=0,
                     cbar_kws={"label": cbar_label})

    xticks, xlabels = _year_ticks(lookbacks, trading_days, ticks_per_year)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45)

    yticks, ylabels = _year_ticks(lookaheads, trading_days, ticks_per_year)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, rotation=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_scatter_grid(feature_by_window: Dict[int, pd.Series],
                       target_by_window: Dict[int, pd.Series],
                       feature_label: str = "Feature",
                       target_label: str = "Target",
                       feature_format: Callable[[int], str] = str,
                       target_format: Callable[[int], str] = str,
                       figsize: Tuple[float, float] = (12.0, 10.0),
                       scatter_kwargs: Optional[dict] = None) -> None:
    """Grilla de scatter feature × target para todas las combinaciones de ventanas.

    ``feature_by_window`` y ``target_by_window`` deben compartir índices (o,
    al menos, solapar en sus fechas).  Las claves determinan los rótulos de
    cada eje.
    """
    scatter_kwargs = scatter_kwargs or {"s": 8, "alpha": 0.25}
    feature_windows = list(feature_by_window.keys())
    target_windows = list(target_by_window.keys())

    fig, axes = plt.subplots(len(target_windows), len(feature_windows),
                             figsize=figsize, sharex=False, sharey=False,
                             constrained_layout=True)

    for i, tw in enumerate(target_windows):
        for j, fw in enumerate(feature_windows):
            ax = axes[i, j]
            data = pd.concat(
                [feature_by_window[fw].rename(feature_label),
                 target_by_window[tw].rename(target_label)],
                axis=1,
            ).dropna()
            ax.scatter(data.iloc[:, 0], data.iloc[:, 1], **scatter_kwargs)
            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.axvline(0, color="black", lw=0.8, alpha=0.6)
            ax.set_title(
                f"{feature_label} {feature_format(fw)} vs "
                f"{target_label} {target_format(tw)}", fontsize=9,
            )
            ax.set_xlabel(f"{feature_label} {feature_format(fw)}")
            ax.set_ylabel(f"{target_label} {target_format(tw)}")
    plt.show()


def plot_feature_boxplots(feature: pd.Series,
                          target: pd.Series,
                          n_quantiles: int = 5,
                          x_axis: str = "quantiles",
                          feature_name: str = "feature",
                          target_name: str = "target",
                          feature_window: Optional[int] = None,
                          target_window: Optional[int] = None,
                          ax: Optional[plt.Axes] = None,
                          palette: str = "viridis",
                          return_correlation: bool = True) -> Optional[float]:
    """Boxplot del objetivo agrupado por la característica.

    El eje X puede construirse de dos formas equivalentes:

    * ``x_axis='quantiles'`` (por defecto): usa :func:`pandas.qcut` para
      dividir la característica en ``n_quantiles`` grupos con el mismo
      número de observaciones cada uno (cuantiles clásicos).
    * ``x_axis='bins'``: usa :func:`pandas.cut` para dividir el rango
      ``[min(feature), max(feature)]`` en ``n_quantiles`` intervalos de
      igual anchura en valor crudo.

    ``n_quantiles`` controla el número de grupos en ambos casos.

    Si ``return_correlation`` es ``True`` devuelve la correlación de
    Pearson entre ``feature`` y ``target`` sobre la muestra conjunta; si
    es ``False`` devuelve ``None``.
    """
    if x_axis not in ("quantiles", "bins"):
        raise ValueError(
            f"x_axis debe ser 'quantiles' o 'bins'; se recibió '{x_axis}'."
        )

    data = pd.concat(
        [feature.rename(feature_name), target.rename(target_name)],
        axis=1,
    ).dropna()

    if x_axis == "quantiles":
        labels = [f"Q{i + 1}" for i in range(n_quantiles)]
        if n_quantiles >= 2:
            labels[0] = "Q1\n(Lowest)"
            labels[-1] = f"Q{n_quantiles}\n(Highest)"
        data["quantile"] = pd.qcut(data[feature_name],
                                    q=n_quantiles, labels=labels)
        x_label_suffix = f"{feature_name.capitalize()} Quantiles"
    else:
        f_min = float(data[feature_name].min())
        f_max = float(data[feature_name].max())
        edges = np.linspace(f_min, f_max, n_quantiles + 1)
        edges = np.unique(edges)
        if len(edges) < 2:
            raise ValueError(
                "No se pueden construir bins: la característica es constante."
            )
        bin_labels = [f"{(lo + hi) / 2:.2g}"
                      for lo, hi in zip(edges[:-1], edges[1:])]
        data["quantile"] = pd.cut(data[feature_name],
                                   bins=edges, include_lowest=True,
                                   labels=bin_labels)
        x_label_suffix = f"{feature_name.capitalize()} Bins"

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(data=data, x="quantile", y=target_name, ax=ax,
                palette=palette, width=0.6, fliersize=3)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)

    fw_str = f"({feature_window}d) " if feature_window is not None else ""
    tw_str = f"({target_window}d) " if target_window is not None else ""
    ax.set_title(f"{feature_name.capitalize()} {fw_str}vs {target_name} {tw_str}",
                 fontsize=12)
    ax.set_xlabel(f"{x_label_suffix} {fw_str}".strip())
    ax.set_ylabel(f"{target_name} {tw_str}".strip())

    if created_fig:
        plt.tight_layout()
        plt.show()

    if return_correlation:
        return data[feature_name].corr(data[target_name])
    return None


def plot_cross_sectional_curves(corr_matrix: pd.DataFrame,
                                 avg_corr: pd.Series,
                                 asset_groups: Optional[Dict[str, Sequence[str]]] = None,
                                 group_colors: Optional[Dict[str, str]] = None,
                                 avg_label: str = "Promedio",
                                 xlabel: str = r"$\tau$ (años)",
                                 ylabel: str = "Correlación de Pearson",
                                 figsize: Tuple[float, float] = (6.0, 4.0),
                                 tau_to_yf: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
                                 trading_days: int = TRADING_DAYS) -> None:
    """Curvas de correlación por activo con promedio cross-sectional."""
    fig, ax = plt.subplots(figsize=figsize)

    if tau_to_yf is None:
        tau = 2 * corr_matrix.index.to_numpy() / trading_days
    else:
        tau = tau_to_yf(corr_matrix.index.to_numpy())

    if asset_groups is None:
        for asset in corr_matrix.columns:
            ax.plot(tau, corr_matrix[asset], linewidth=1, alpha=0.9,
                    marker="o", markersize=5, label=asset)
    else:
        for group_name, assets in asset_groups.items():
            color = (group_colors or {}).get(group_name)
            for asset in assets:
                if asset not in corr_matrix.columns:
                    continue
                kwargs = {"linewidth": 1, "alpha": 0.9,
                          "marker": "o", "markersize": 5,
                          "label": group_name}
                if color is not None:
                    kwargs["color"] = color
                ax.plot(tau, corr_matrix[asset], **kwargs)

    ax.plot(tau, avg_corr, color="black", linewidth=2.5,
            linestyle="--", marker="o", markersize=6, label=avg_label)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()