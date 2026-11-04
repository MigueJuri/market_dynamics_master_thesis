"""Evaluación y backtest de pronósticos de volatilidad.

Incluye métricas de error estándar, test de Diebold-Mariano, backtest de
vol-timing y análisis de adaptabilidad post-shock.  Las figuras se dibujan
en el cuaderno ``notebooks/volatility/Volatilidad_LSTM.py`` para que puedan
personalizarse sin tocar este módulo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Modelo",
) -> dict[str, Any]:
    """Evaluación comprensiva de pronósticos de volatilidad.

    Todas las métricas operan en espacio de sigma (volatilidad anualizada).

    Parameters
    ----------
    y_true : np.ndarray
        Volatilidad real.
    y_pred : np.ndarray
        Volatilidad predicha.
    label : str
        Etiqueta del modelo para el reporte.

    Returns
    -------
    dict
        Diccionario con R², RMSE, MAE, Spearman, precisión direccional,
        QLIKE, coeficientes Mincer-Zarnowitz y estadístico F conjunto.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    sp_r, sp_p = stats.spearmanr(y_pred, y_true)

    dir_acc = np.mean((y_pred > np.median(y_pred)) == (y_true > np.median(y_true)))
    dir_test = stats.binomtest(round(dir_acc * len(y_true)), len(y_true), 0.5)
    dir_p = float(getattr(dir_test, "pvalue", dir_test))

    eps = 1e-8
    ratio = y_true**2 / (y_pred**2 + eps)
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))

    X_mz = sm.add_constant(y_pred)
    mz_res = sm.OLS(y_true, X_mz).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    R = np.eye(2)
    r_v = np.array([0.0, 1.0])
    mz_f = mz_res.f_test((R, r_v))

    metrics = {
        "model": label,
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Spearman_r": float(sp_r),
        "Spearman_p": float(sp_p),
        "Dir_Acc": float(dir_acc),
        "Dir_p": dir_p,
        "QLIKE": qlike,
        "MZ_alpha": float(mz_res.params[0]),
        "MZ_beta": float(mz_res.params[1]),
        "MZ_F_pval": float(mz_f.pvalue),
        "N": len(y_true),
    }

    sep = "-" * 46
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}")
    print(f"  R²:            {r2:+.4f}   (referencia: > 0.08)")
    print(f"  RMSE:          {rmse * 100:.4f}%  (unidades de vol anualizada)")
    print(f"  MAE:           {mae * 100:.4f}%")
    print(f"  Spearman r:    {sp_r:.4f}   (p = {sp_p:.4f})")
    print(f"  Dir. Accuracy: {dir_acc:.4f}   (p = {dir_p:.4f})")
    print(f"  QLIKE:         {qlike:.6f}")
    print(f"  MZ alpha:      {mz_res.params[0]:.4f}   (H₀: = 0)")
    print(f"  MZ beta:       {mz_res.params[1]:.4f}   (H₀: = 1)")
    print(f"  MZ F p-valor:  {float(mz_f.pvalue):.4f}   (H₀: insesgado)")
    return metrics


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 5,
) -> dict[str, float]:
    """Test de Diebold-Mariano (1995) para igualdad de precisión predictiva.

    ``H₀``: los modelos 1 y 2 tienen el mismo MSE esperado.

    Parameters
    ----------
    e1 : np.ndarray
        Residuos del modelo 1 (LSTM).
    e2 : np.ndarray
        Residuos del modelo 2 (HAR-RV).
    h : int
        Horizonte de predicción, determina los rezagos HAC.

    Returns
    -------
    dict
        ``{DM_stat, DM_pvalue}``.
    """
    d = e1**2 - e2**2
    d_bar = np.mean(d)
    T = len(d)

    gamma0 = np.var(d, ddof=0)
    hac_var = gamma0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        hac_var += 2.0 * (1.0 - k / h) * gamma_k

    se = np.sqrt(max(hac_var, 1e-12) / T)
    dm_stat = d_bar / se
    dm_pval = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    direction = "LSTM más preciso" if dm_stat < 0 else "HAR-RV más preciso"
    sig = "SIGNIFICATIVO" if dm_pval < 0.05 else "no significativo"

    print(
        f"\n[TEST DM] LSTM vs HAR-RV: stat = {dm_stat:.4f}, p = {dm_pval:.4f}  [{sig}]"
    )
    print(f"  ({direction} bajo criterio MSE)")

    return {"DM_stat": float(dm_stat), "DM_pvalue": float(dm_pval)}


def sharpe_ratio(r: np.ndarray) -> float:
    """Sharpe anualizado sobre retornos simples (tasa libre de riesgo nula)."""
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return float("nan")
    return float(np.sqrt(252.0) * np.mean(r) / (np.std(r) + 1e-10))


def max_drawdown(r: np.ndarray) -> float:
    """Máximo drawdown sobre retornos simples.

    La riqueza se construye como ``cumprod(1 + r)`` **anteponiendo el capital
    inicial (1.0)**, de modo que una caída desde el primer día también cuenta
    como drawdown.  El valor devuelto es negativo (o cero).
    """
    r = np.asarray(r, dtype=float)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / (peak + 1e-10)))


def calmar_ratio(r: np.ndarray) -> float:
    """Ratio de Calmar sobre retornos simples: ``CAGR / |MDD|``.

    El CAGR anualizado se calcula como ``prod(1 + r) ** (252 / n) - 1``.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    if n == 0:
        return float("nan")
    mdd = abs(max_drawdown(r))
    total = float(np.prod(1.0 + r))
    if total <= 0.0:
        return float("nan")
    cagr = total ** (252.0 / n) - 1.0
    if mdd <= 0.0:
        if cagr > 0.0:
            return float("inf")
        return 0.0 if cagr == 0.0 else float("-inf")
    return float(cagr / mdd)


def summarize_returns(r: np.ndarray) -> dict[str, float]:
    """Resumen de una serie de retornos simples diarios.

    Parameters
    ----------
    r : np.ndarray
        Retornos **simples** (no logarítmicos) de la estrategia.

    Returns
    -------
    dict
        ``total_return`` (``prod(1 + r) - 1``), ``sharpe``, ``maxdd`` y
        ``calmar``.
    """
    r = np.asarray(r, dtype=float)
    total_return = float(np.prod(1.0 + r)) - 1.0
    return {
        "total_return": total_return,
        "sharpe": sharpe_ratio(r),
        "maxdd": max_drawdown(r),
        "calmar": calmar_ratio(r),
    }


def run_vol_timing_backtest(
    ret: np.ndarray,
    pred_vol: np.ndarray,
    sigma_bar: float,
    cost_bps: float = 5.0,
) -> dict[str, Any]:
    """Estrategia de vol-timing sobre una señal de momentum.

    La señal base es el signo del retorno medio de los últimos 63 días
    (ventana adaptativa ``min(252, n // 4)``, con un mínimo de 5
    observaciones).  El ``shift(1)`` hace que la posición de ``t`` use
    únicamente retornos hasta ``t-1`` (sin look-ahead).  La posición se escala
    por la volatilidad predicha inversa:

        ``pos_vol(t) = clip(signo(t) / (σ̂(t) / σ̄), -1, 1)``

    Parameters
    ----------
    ret : np.ndarray
        Retornos **logarítmicos** alineados con ``pred_vol``.  La señal de
        momentum se calcula sobre ellos; la aritmética de la cartera usa
        retornos simples (``expm1``).
    pred_vol : np.ndarray
        Volatilidad predicha.
    sigma_bar : float
        Volatilidad de referencia (mediana del train, en espacio sigma).
    cost_bps : float
        Costo de transacción unilateral en puntos base.

    Returns
    -------
    dict
        Series de retornos **simples** netos de costos y métricas de
        Vol-Timed, baseline y buy & hold.
    """
    n = len(pred_vol)
    ret_arr = np.asarray(ret, dtype=float)[:n]

    # Señal de momentum: media móvil de los retornos con desplazamiento de una
    # barra, de modo que la posición de ``t`` use sólo información de ``t-1``
    # o anterior.  Los ``NaN`` del warm-up se tratan explícitamente (flat).
    mom_win = min(252, max(1, n // 4))
    mom = (
        pd.Series(ret_arr).rolling(mom_win, min_periods=min(5, mom_win)).mean().shift(1)
    )
    sig = np.sign(mom.fillna(0.0).to_numpy())

    vol_scale = pred_vol / sigma_bar

    pos_vol = np.clip(sig / (vol_scale + 1e-8), -1.0, 1.0)
    pos_base = np.clip(sig, -1.0, 1.0)
    pos_bh = np.ones_like(ret_arr, dtype=float)

    cost = cost_bps * 1e-4
    tc_vol = cost * np.abs(np.diff(np.r_[0.0, pos_vol]))
    tc_base = cost * np.abs(np.diff(np.r_[0.0, pos_base]))
    tc_bh = np.zeros_like(ret_arr, dtype=float)

    # Retornos simples: la aritmética de la cartera y los costos viven en el
    # mismo espacio (no se mezclan retornos logarítmicos con costos).
    r_simple = np.expm1(ret_arr)
    ret_vol = pos_vol * r_simple - tc_vol
    ret_base = pos_base * r_simple - tc_base
    ret_bh = pos_bh * r_simple - tc_bh

    vol_stats = summarize_returns(ret_vol)
    base_stats = summarize_returns(ret_base)
    bh_stats = summarize_returns(ret_bh)

    return {
        "ret_vol": ret_vol,
        "ret_base": ret_base,
        "ret_bh": ret_bh,
        "total_vol": vol_stats["total_return"],
        "total_base": base_stats["total_return"],
        "total_bh": bh_stats["total_return"],
        "sharpe_vol": vol_stats["sharpe"],
        "sharpe_base": base_stats["sharpe"],
        "sharpe_bh": bh_stats["sharpe"],
        "maxdd_vol": vol_stats["maxdd"],
        "maxdd_base": base_stats["maxdd"],
        "maxdd_bh": bh_stats["maxdd"],
        "calmar_vol": vol_stats["calmar"],
        "calmar_base": base_stats["calmar"],
        "calmar_bh": bh_stats["calmar"],
        "sharpe_improvement": vol_stats["sharpe"] - base_stats["sharpe"],
        "sharpe_vs_bh": vol_stats["sharpe"] - bh_stats["sharpe"],
    }


def calculate_adaptability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 15,
    threshold: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Curva de decaimiento del Event-MAE tras shocks de volatilidad.

    Identifica shocks como valores por encima de ``threshold`` desviaciones
    estándar de una media móvil de 63 días, promedia el error absoluto
    post-shock y ajusta una exponencial para obtener la vida media.

    Parameters
    ----------
    y_true : np.ndarray
        Volatilidad real.
    y_pred : np.ndarray
        Volatilidad predicha.
    horizon : int
        Días post-shock a analizar.
    threshold : float
        Umbral de desviaciones estándar para definir un shock.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(mean_mae, half_life)``.  Si no hay shocks, ``mean_mae`` es un
        vector de ceros y ``half_life`` es ``NaN``.
    """
    error = np.abs(y_true - y_pred)
    y_s = pd.Series(y_true)

    roll_mean = y_s.rolling(63, min_periods=21).mean()
    roll_std = y_s.rolling(63, min_periods=21).std()

    is_shock = (y_s > (roll_mean + threshold * roll_std)).fillna(False).astype(bool)
    prev_shock = is_shock.shift(1).fillna(False).astype(bool)
    starts = is_shock & np.logical_not(prev_shock)
    shock_indices = np.where(starts)[0]

    matrices = []
    for idx in shock_indices:
        if idx + horizon < len(y_true):
            matrices.append(error[idx + 1 : idx + horizon + 1])

    if not matrices:
        return np.zeros(horizon), np.nan

    mean_mae = np.mean(matrices, axis=0)

    def decay_func(x: np.ndarray, A: float, lam: float, C: float) -> np.ndarray:
        return A * np.exp(-lam * x) + C

    x_data = np.arange(1, horizon + 1)
    try:
        p0 = [mean_mae.max() - mean_mae.min(), 0.1, mean_mae.min()]
        popt, _ = curve_fit(
            decay_func,
            x_data,
            mean_mae,
            p0=p0,
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        lam = popt[1]
        half_life = np.log(2) / lam if lam > 1e-6 else np.nan
    except (RuntimeError, ValueError):
        half_life = np.nan

    return mean_mae, float(half_life)
