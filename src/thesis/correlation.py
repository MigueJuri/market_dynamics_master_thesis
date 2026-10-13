"""Estimadores de correlación lookback × lookahead.

Incluye la matriz temporal :func:`compute_correlation_matrix`, su versión
cross-sectional :func:`compute_cross_sectional_correlation` y las
utilidades de bootstrap asociadas.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from thesis.config import SEED, TRADING_DAYS
from thesis.timeseries import RollingOp, apply_window


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
    target_transform: Optional[Callable[..., pd.Series]] = None,
    target_window_type: str = "fixed",
    target_ewm_adjust: bool = False,
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
    ``'mean'`` para drift promedio, etc.).  El objetivo por defecto es el
    retorno futuro acumulado sobre el lookahead (``target_op='sum'``,
    ventana fija); cuando se pasa ``target_transform`` se sustituye por
    un callable ``(series, tau_p, tau_f, **kwargs) -> pd.Series`` que
    puede depender de ambas ventanas (p. ej. la distancia futura a la
    media móvil histórica).  La métrica de correlación puede ser
    ``'spearman'`` o ``'pearson'``.

    Si ``feature_windows`` o ``target_windows`` se proporcionan, se usan
    directamente; si no, se genera una rejilla con
    ``(max_lookback + max_lookahead) // steps`` pasos.

    Parámetros de ventana
    ---------------------
    feature_window_type : {'fixed', 'exponential'}
        Tipo de ventana aplicada a la característica.  Con
        ``'exponential'`` el ``span`` se pasa a
        :func:`pandas.Series.ewm` y ``feature_op`` debe ser una cadena
        (``'mean'``, ``'std'``, ``'var'``).  Con ``'fixed'`` se admite
        además un callable.
    feature_transform : callable, opcional
        Función ``(windowed, base) -> feature`` aplicada tras el rolling
        o ewm.  Útil para combinar la serie original con su media móvil,
        p. ej. ``lambda w, s: s - w`` para ``s - MA(s)``.
    target_transform : callable, opcional
        Si se indica, sustituye al objetivo por defecto por la salida de
        ``target_transform(target_series, tau_p, tau_f,
        window_type=target_window_type, ewm_adjust=target_ewm_adjust)``.
        Útil cuando el target depende de ambas ventanas (p. ej.
        :func:`thesis.features.future_distance_to_moving_average`).
    target_window_type : {'fixed', 'exponential'}
        Tipo de ventana pasado a ``target_transform`` si se usa; sólo
        relevante si el callable lo consume.
    target_ewm_adjust : bool
        Corrección de sesgo de EWM pasada a ``target_transform`` si se
        usa.
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
        :data:`thesis.config.SEED` para resultados reproducibles.
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

    # Caché del target por fila (target_op) o por celda (target_transform).
    fut_cache: Dict[Tuple[int, int], pd.Series] = {}

    for i, f in enumerate(lookaheads):
        for j in range(len(lookbacks)):
            l_j = int(lookbacks[j])
            f_i = int(f)
            key = (i, j) if target_transform is not None else i
            if key not in fut_cache:
                if target_transform is None:
                    fut_cache[key] = (
                        target_series.rolling(f_i).sum().shift(-f_i)
                    )
                else:
                    fut_cache[key] = target_transform(
                        target_series, l_j, f_i,
                        window_type=target_window_type,
                        ewm_adjust=target_ewm_adjust,
                    )
            fut = fut_cache[key]
            feature = feature_cache[j]

            valid_idx = ~(feature.isna() | fut.isna())

            if valid_idx.sum() > min_valid - 1:
                shift_step = 1 if overlap else min(l_j, f_i)
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
            target_transform=target_transform,
            target_window_type=target_window_type,
            target_ewm_adjust=target_ewm_adjust,
        )

    return corr_matrix, lookbacks, lookaheads


def compute_cross_sectional_correlation(
    log_returns_df: pd.DataFrame,
    windows: Sequence[int],
    feature_df: Optional[pd.DataFrame] = None,
    target_df: Optional[pd.DataFrame] = None,
    feature_op: RollingOp = "sum",
    feature_window_type: str = "fixed",
    feature_transform: Optional[Callable[[pd.Series, pd.Series], pd.Series]] = None,
    target_op: RollingOp = "sum",
    target_transform: Optional[Callable[..., pd.Series]] = None,
    target_window_type: str = "fixed",
    target_ewm_adjust: bool = False,
    correlation: str = "pearson",
    min_valid_n: int = 50,
    overlap: bool = False,
    shift_lookahead_by_lookback: bool = False,
    ewm_adjust: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Correlación cruzada entre activo y horizonte temporal.

    Para cada activo de ``log_returns_df`` se calcula la correlación entre
    la característica móvil (``feature_op`` aplicada al input de la
    característica) y el objetivo (``target_op`` por defecto, o
    ``target_transform`` si se pasa) para cada ventana de ``windows``.
    El resultado se devuelve como un ``DataFrame`` indexado por ventana
    con una columna por activo, junto con el promedio cross-sectional.

    Parámetros
    ----------
    log_returns_df : pd.DataFrame
        Series usadas como objetivo por defecto y como entrada de la
        característica si ``feature_df`` es ``None``.  Por convención son
        retornos (logarítmicos) porque el objetivo por defecto es la
        suma de retornos futuros.
    windows : secuencia de int
        Ventanas (en sesiones) sobre las que se evalúa la predictibilidad.
    feature_df : pd.DataFrame, opcional
        Series usadas como input de la característica.  Si es ``None``
        (por defecto), se reutiliza ``log_returns_df``; útil cuando la
        característica se computa sobre precios (p. ej. distancia a la
        media móvil) y el objetivo sobre retornos.  Debe compartir
        columnas con ``log_returns_df``.
    target_df : pd.DataFrame, opcional
        Series usadas como input del objetivo.  Si es ``None`` (por
        defecto) se reutiliza ``log_returns_df``; útil cuando el
        objetivo depende de precios (p. ej. distancia futura a la media
        móvil) y la característica también.  Debe compartir columnas con
        ``log_returns_df``.
    feature_op, target_op : str o callable
        Reducciones móviles aplicadas a la característica y al objetivo.
    feature_window_type : {'fixed', 'exponential'}
        Tipo de ventana aplicada a la característica (análogo a
        :func:`compute_correlation_matrix`).
    feature_transform : callable, opcional
        Función ``(windowed, base) -> feature`` aplicada tras el rolling
        o ewm de la característica.
    target_transform : callable, opcional
        Si se indica, sustituye al objetivo por defecto por la salida de
        ``target_transform(target_source, window, window,
        window_type=target_window_type, ewm_adjust=target_ewm_adjust)``.
        Útil cuando el target depende del propio lookback (p. ej.
        :func:`thesis.features.future_distance_to_moving_average`).
    target_window_type : {'fixed', 'exponential'}
        Tipo de ventana pasado a ``target_transform`` si se usa.
    target_ewm_adjust : bool
        Corrección de sesgo de EWM pasada a ``target_transform`` si se
        usa.
    correlation : {'pearson', 'spearman'}
        Métrica de correlación.
    min_valid_n : int
        Número mínimo de observaciones válidas para aceptar una estimación.
    overlap : bool
        Si ``False``, las observaciones se submuestrean con paso ``window``.
    shift_lookahead_by_lookback : bool
        Si ``True`` y no se usa ``target_transform``, el objetivo
        comienza inmediatamente después de que termina el lookback
        (``shift(-window).rolling(window).sum()``); si no, se calcula con
        la convención habitual (``rolling_forward_window``).
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
            f"Correlación desconocida '{correlation}'. Usa 'pearson' o 'spearman'."
        )

    if feature_df is None:
        feature_df = log_returns_df
    if target_df is None:
        target_df = log_returns_df

    corr_matrix = pd.DataFrame(index=list(windows),
                                columns=list(log_returns_df.columns),
                                dtype=float)

    for asset in log_returns_df.columns:
        if asset not in feature_df.columns or asset not in target_df.columns:
            continue
        feature_series = feature_df[asset].dropna()
        target_source = target_df[asset].dropna()
        for window in windows:
            windowed = apply_window(
                feature_series, float(window), feature_op,
                window_type=feature_window_type, ewm_adjust=ewm_adjust,
            )
            if feature_transform is None:
                feature = windowed
            else:
                feature = feature_transform(windowed, feature_series)
            if target_transform is None:
                if shift_lookahead_by_lookback:
                    target = target_source.shift(-window).rolling(window).sum()
                else:
                    target = _apply_rolling(target_source, window, target_op)
                    if target_op == "sum":
                        target = target.shift(-window)
            else:
                target = target_transform(
                    target_source, window, window,
                    window_type=target_window_type,
                    ewm_adjust=target_ewm_adjust,
                )
            valid_idx = ~(feature.isna() | target.isna())
            valid_idx = valid_idx.fillna(False)
            common_idx = feature.index.intersection(target.index)
            common_idx = common_idx[valid_idx.reindex(common_idx,
                                                      fill_value=False)]
            n_valid = len(common_idx)
            if n_valid >= min_valid_n:
                x = feature.reindex(common_idx).to_numpy()
                y = target.reindex(common_idx).to_numpy()
                if not overlap:
                    step = max(1, window)
                    x = x[::step]
                    y = y[::step]
                corr, _ = corr_func(x, y)
                corr_matrix.loc[window, asset] = corr

    avg_corr = corr_matrix.mean(axis=1, skipna=True)
    return corr_matrix, avg_corr


# =============================================================================
# Bootstrap del coeficiente máximo
# =============================================================================

def _corr_value(corr_func, x, y) -> float:
    """Devuelve el escalar de correlación tanto para tuplas como para
    objetos ``CorrelationResult`` (scipy ≥ 1.10)."""
    res = corr_func(x, y)
    return float(res.statistic if hasattr(res, "statistic") else res[0])


def _bootstrap_correlation(
    x: np.ndarray,
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
    target_transform: Optional[Callable[..., pd.Series]] = None,
    target_window_type: str = "fixed",
    target_ewm_adjust: bool = False,
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
    if target_transform is None:
        fut = target_series.rolling(int(tau_f)).sum().shift(-int(tau_f))
    else:
        fut = target_transform(
            target_series, tau_p, int(tau_f),
            window_type=target_window_type,
            ewm_adjust=target_ewm_adjust,
        )

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


def _apply_rolling(series: pd.Series, window: int, op: RollingOp) -> pd.Series:
    """Wrapper interno: ventana fija equivalente a ``apply_window``."""
    return apply_window(series, window, op, window_type="fixed")
