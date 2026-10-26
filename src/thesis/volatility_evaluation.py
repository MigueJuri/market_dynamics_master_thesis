"""Evaluación, backtest y visualización de pronósticos de volatilidad.

Incluye métricas de error estándar, test de Diebold-Mariano, backtest de
vol-timing, análisis de adaptabilidad post-shock y figuras diagnósticas.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec
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
    """Sharpe anualizado asumiendo retornos logarítmicos diarios."""
    return float(np.sqrt(252) * np.mean(r) / (np.std(r) + 1e-10))


def max_drawdown(r: np.ndarray) -> float:
    """Máximo drawdown sobre retornos logarítmicos."""
    eq = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / (peak + 1e-10)))


def calmar_ratio(r: np.ndarray) -> float:
    """Ratio de Calmar anualizado."""
    mdd = abs(max_drawdown(r))
    return float(np.sqrt(252) * np.mean(r) / (mdd + 1e-10))


def summarize_log_returns(r: np.ndarray) -> dict[str, float]:
    """Resumen de una serie de retornos logarítmicos diarios."""
    equity = np.exp(np.cumsum(r))
    return {
        "total_return": float(equity[-1] - 1.0),
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

    La señal base es el signo del retorno medio de los últimos 63 días.
    La posición se escala por la volatilidad predicha inversa:

        ``pos_vol(t) = clip(signo(t) / (σ̂(t) / σ̄), -1, 1)``

    Parameters
    ----------
    ret : np.ndarray
        Retornos logarítmicos alineados con ``pred_vol``.
    pred_vol : np.ndarray
        Volatilidad predicha.
    sigma_bar : float
        Volatilidad de referencia (mediana del train, en espacio sigma).
    cost_bps : float
        Costo de transacción unilateral en puntos base.

    Returns
    -------
    dict
        Series de retornos y métricas de Vol-Timed, baseline y buy & hold.
    """
    n = len(pred_vol)
    ret_arr = ret[:n]

    mom_win = min(63, n // 4)
    mom = pd.Series(ret_arr).rolling(mom_win, min_periods=5).mean().shift(1)
    sig = np.sign(mom.fillna(0).values)
    sig[np.isnan(sig)] = 0.0

    vol_scale = pred_vol / sigma_bar

    pos_vol = np.clip(sig / (vol_scale + 1e-8), -1.0, 1.0)
    pos_base = np.clip(sig, -1.0, 1.0)
    pos_bh = np.ones_like(ret_arr, dtype=float)

    cost = cost_bps * 1e-4
    tc_vol = cost * np.abs(np.diff(np.r_[0.0, pos_vol]))
    tc_base = cost * np.abs(np.diff(np.r_[0.0, pos_base]))
    tc_bh = np.zeros_like(ret_arr, dtype=float)

    ret_vol = pos_vol * ret_arr - tc_vol
    ret_base = pos_base * ret_arr - tc_base
    ret_bh = pos_bh * ret_arr - tc_bh

    vol_stats = summarize_log_returns(ret_vol)
    base_stats = summarize_log_returns(ret_base)
    bh_stats = summarize_log_returns(ret_bh)

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


def generate_plots(
    y_true: np.ndarray,
    y_pred_lstm: np.ndarray,
    y_pred_har: np.ndarray,
    test_dates: pd.DatetimeIndex,
    history: dict[str, list[float]],
    bt: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Figura diagnóstica de 6 paneles para el experimento de volatilidad.

    Parameters
    ----------
    y_true : np.ndarray
        Volatilidad real en test.
    y_pred_lstm : np.ndarray
        Predicciones de la LSTM.
    y_pred_har : np.ndarray
        Predicciones del HAR-RV.
    test_dates : pd.DatetimeIndex
        Fechas del período de test.
    history : dict
        Historial de pérdida de entrenamiento.
    bt : dict
        Resultado del backtest de vol-timing.
    config : dict
        Configuración con ``plot_path``.
    """
    matplotlib.rcParams.update({"font.family": "monospace"})

    BG = "#0d1117"
    CELL = "#161b22"
    GRID = "#21262d"
    TEXT = "#c9d1d9"
    C = {
        "actual": "#58a6ff",
        "lstm": "#f78166",
        "har": "#3fb950",
        "volbt": "#d2a8ff",
        "basebt": "#ffa657",
        "white": "#ffffff",
    }

    fig = plt.figure(figsize=(20, 24), facecolor=BG)
    gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    def ax_style(ax: plt.Axes, title: str = "") -> None:
        ax.set_facecolor(CELL)
        for side in ax.spines.values():
            side.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
        if title:
            ax.set_title(title, color=TEXT, fontweight="bold", pad=8, fontsize=10)

    n = len(y_true)
    td = test_dates[-n:] if len(test_dates) >= n else pd.RangeIndex(n)

    ax = fig.add_subplot(gs[0, 0])
    ax_style(ax, "Pérdida LINEX por Época (Train / Val)")
    ax.plot(history["train_loss"], color=C["lstm"], lw=1.5, label="Train")
    ax.plot(history["val_loss"], color=C["har"], lw=1.5, label="Validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("LINEX")
    ax.legend(facecolor=CELL, labelcolor=TEXT)

    ax = fig.add_subplot(gs[0, 1])
    ax_style(ax, "LSTM: Predicho vs Real (Test)")
    ax.scatter(y_true, y_pred_lstm, alpha=0.35, s=6, color=C["lstm"])
    lo = min(y_true.min(), y_pred_lstm.min())
    hi = max(y_true.max(), y_pred_lstm.max())
    ax.plot([lo, hi], [lo, hi], "--", color=C["white"], lw=1.0, alpha=0.5)
    ax.text(
        0.05,
        0.93,
        f"R² = {r2_score(y_true, y_pred_lstm):.4f}",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=10,
    )
    ax.set_xlabel("Vol Real (anualizada)")
    ax.set_ylabel("Vol Predicha (anualizada)")

    ax = fig.add_subplot(gs[1, :])
    ax_style(ax, "Volatilidad Realizada Forward (5 días) — Período de Test")
    ax.plot(td, y_true, color=C["actual"], lw=1.2, label="Real", alpha=0.9)
    ax.plot(td, y_pred_lstm, color=C["lstm"], lw=1.0, label="LSTM", alpha=0.8)
    ax.plot(
        td,
        y_pred_har,
        color=C["har"],
        lw=1.0,
        label="HAR-RV",
        alpha=0.8,
        ls="--",
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Vol Anualizada")
    ax.legend(facecolor=CELL, labelcolor=TEXT)
    if hasattr(td[0], "strftime"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")

    ax = fig.add_subplot(gs[3, 0])
    ax_style(ax, "Adaptabilidad Post-Shock (Decaimiento Event-MAE)")
    horizon = 11
    mae_lstm, hl_lstm = calculate_adaptability(
        y_true, y_pred_lstm, horizon=horizon, threshold=2.5
    )
    mae_har, hl_har = calculate_adaptability(
        y_true, y_pred_har, horizon=horizon, threshold=2.5
    )
    x_k = np.arange(1, horizon + 1)
    lbl_lstm = f"LSTM (t_1/2 = {hl_lstm:.1f}d)" if not np.isnan(hl_lstm) else "LSTM"
    lbl_har = f"HAR-RV (t_1/2 = {hl_har:.1f}d)" if not np.isnan(hl_har) else "HAR-RV"
    ax.plot(
        x_k, mae_lstm, color=C["lstm"], marker="o", markersize=4, lw=1.5, label=lbl_lstm
    )
    ax.plot(
        x_k,
        mae_har,
        color=C["har"],
        marker="s",
        markersize=4,
        lw=1.5,
        ls="--",
        label=lbl_har,
    )
    ax.set_xlabel("Días Hábiles Post-Shock (k)")
    ax.set_ylabel("Error Medio Absoluto (MAE)")
    ax.set_xticks(x_k[::2])
    ax.legend(facecolor=CELL, labelcolor=TEXT)

    ax = fig.add_subplot(gs[3, 1])
    ax_style(ax, "Correlación de Rangos: LSTM vs HAR-RV")
    rank_true = stats.rankdata(y_true)
    ax.scatter(
        rank_true,
        stats.rankdata(y_pred_lstm),
        color=C["lstm"],
        alpha=0.3,
        s=5,
        label="LSTM",
    )
    ax.scatter(
        rank_true,
        stats.rankdata(y_pred_har),
        color=C["har"],
        alpha=0.3,
        s=5,
        label="HAR-RV",
    )
    ax.set_xlabel("Rango: Vol Real")
    ax.set_ylabel("Rango: Vol Predicha")
    ax.legend(facecolor=CELL, labelcolor=TEXT)

    ax = fig.add_subplot(gs[2, :])
    ax_style(ax, "Backtest Estratégico — Retorno Acumulado (Test)")
    eq_vol = np.cumprod(1.0 + bt["ret_vol"])
    eq_base = np.cumprod(1.0 + bt["ret_base"])
    ax.plot(
        td,
        eq_vol,
        color=C["volbt"],
        lw=1.5,
        label=f"Vol-Timed (SR={bt['sharpe_vol']:.2f})",
    )
    ax.plot(
        td,
        eq_base,
        color=C["basebt"],
        lw=1.5,
        ls="--",
        label=f"Baseline  (SR={bt['sharpe_base']:.2f})",
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno Acumulado")
    ax.legend(facecolor=CELL, labelcolor=TEXT)
    if hasattr(td[0], "strftime"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")

    fig.suptitle(
        "Sistema LSTM de Predicción de Volatilidad — Diagnóstico de Adaptación",
        color=TEXT,
        fontsize=16,
        fontweight="bold",
        y=0.92,
    )

    plt.savefig(config["plot_path"], dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"  Figura guardada → {config['plot_path']}")
