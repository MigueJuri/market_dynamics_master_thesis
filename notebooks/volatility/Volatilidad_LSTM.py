# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: tesis
#     language: python
#     name: python3
# ---

# %%
# # Predicción de Volatilidad con LSTM
#
# Este cuaderno entrena una red LSTM parsimoniosa para predecir la
# volatilidad realizada forward del ETF ``SPY`` y la compara con el
# benchmark HAR-RV (Corsi, 2009).
#
# La lógica reutilizable vive en ``src/thesis/volatility.py``,
# ``src/thesis/volatility_models.py`` y ``src/thesis/volatility_evaluation.py``.
#
# El pipeline completo vive en :func:`run_experiment`, de modo que importar
# este archivo no descarga datos ni entrena modelos.

# %%
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Permite ejecutar el archivo desde ``notebooks/`` o ``notebooks/volatility/``.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1] if _HERE.name == "volatility" else _HERE.parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.config import SEED
from thesis.volatility import (
    align_har_features_with_sequences,
    build_volatility_dataset,
    count_extreme_moves,
    download_ohlcv,
    make_sequences,
    scale_splits,
    temporal_split,
)
from thesis.volatility_evaluation import (
    diebold_mariano_test,
    evaluate_forecasts,
    generate_plots,
    run_vol_timing_backtest,
    summarize_log_returns,
)
from thesis.volatility_models import (
    HARRVModel,
    VolatilityLSTM,
    set_volatility_seed,
    train_lstm,
)

# %%
# ## Configuración del experimento
CONFIG: dict[str, Any] = {
    # Datos
    "ticker": "SPY",
    "start_date": "1995-01-01",
    # Divisiones temporales
    "train_end": "2016-01-01",
    "val_end": "2022-01-01",
    # Secuencias y features
    "lookback": 5,
    "forecast_horizon": 5,
    "smoothing_window": 15,
    "linex_b": 0.8,
    "features": [
        "rv_daily",
        "rv_weekly",
        "rv_monthly",
        "vol_ratio",
        "gap",
        "parkinson_vol",
        "gk_vol",
        "rsi_14",
        "rel_volume",
        "volume_trend",
        "intraday_range",
        "skew_5",
        "mean_ret_5",
    ],
    # Arquitectura
    "hidden": 12,
    "dropout": 0.2,
    # Entrenamiento
    "batch_size": 16,
    "lr": 5e-5,
    "weight_decay": 1e-3,
    "min_epochs": 50,
    "max_epochs": 1000,
    "patience": 15,
    "grad_clip": 1.0,
    # Backtest
    "cost_bps": 5,
    # Salidas
    "model_path": "lstm_vol_model.pt",
    "scaler_path": "scaler.pkl",
    "metrics_path": "metrics.csv",
    "report_path": "report.md",
    "config_path": "model_config.json",
    "plot_path": "lstm_vol_report.png",
    "target_scaler_path": "target_scaler.pkl",
}


# %%
def run_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo de predicción de volatilidad.

    Pasos: descarga de datos, construcción del dataset, división temporal,
    escalado, baseline HAR-RV, entrenamiento LSTM, evaluación comparativa,
    backtest de vol-timing, figura diagnóstica y persistencia de artefactos.

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.  Si es ``None`` se usa la
        configuración por defecto.

    Returns
    -------
    dict[str, Any]
        Resultados del experimento: dataset, predicciones, métricas,
        historial de entrenamiento y estadísticas del backtest.
    """
    cfg = {**CONFIG, **(config or {})}

    set_volatility_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DISPOSITIVO] Usando: {device}")

    with open(cfg["config_path"], "w") as f:
        json.dump(
            {
                k: v
                for k, v in cfg.items()
                if isinstance(v, (str, int, float, bool, list))
            },
            f,
            indent=2,
        )
    print("[CONFIG] Guardada en:", cfg["config_path"])

    # --- Descarga de datos ---
    print(f"\n[DATOS] Descargando {cfg['ticker']} desde {cfg['start_date']}...")
    raw = download_ohlcv(cfg["ticker"], start=cfg["start_date"])
    n_extreme = count_extreme_moves(raw)
    if n_extreme:
        print(f"  INFO: {n_extreme} movimientos diarios > 15% detectados (retenidos).")
    print(f"  Forma: {raw.shape} | {raw.index[0].date()} → {raw.index[-1].date()}")

    # --- Construcción del dataset ---
    data = build_volatility_dataset(raw, horizon=cfg["forecast_horizon"])
    print(f"\n[DATASET] {len(data)} observaciones limpias tras dropna.")
    print(f"  Período: {data.index[0].date()} → {data.index[-1].date()}")

    # --- División temporal ---
    train_df, val_df, test_df = temporal_split(data, cfg["train_end"], cfg["val_end"])
    fcols = cfg["features"]

    X_tr = train_df[fcols].values.astype(np.float32)
    X_vl = val_df[fcols].values.astype(np.float32)
    X_te = test_df[fcols].values.astype(np.float32)

    y_tr = train_df["target_log_vol"].values.astype(np.float32)
    y_vl = val_df["target_log_vol"].values.astype(np.float32)
    y_te = test_df["target_log_vol"].values.astype(np.float32)

    te_dates = test_df.index

    print(f"[DIVISIONES] Train: {len(X_tr)} | Val: {len(X_vl)} | Test: {len(X_te)}")
    print(f"  Train: {train_df.index[0].date()} → {train_df.index[-1].date()}")
    print(f"  Val:   {val_df.index[0].date()} → {val_df.index[-1].date()}")
    print(f"  Test:  {test_df.index[0].date()} → {test_df.index[-1].date()}")

    # --- Escalado ---
    scaled = scale_splits(X_tr, X_vl, X_te, y_tr, y_vl, y_te)
    X_tr_s, X_vl_s, X_te_s = scaled["X_train"], scaled["X_val"], scaled["X_test"]
    y_tr_s, y_vl_s, y_te_s = scaled["y_train"], scaled["y_val"], scaled["y_test"]
    X_scaler = scaled["X_scaler"]
    y_scaler = scaled["y_scaler"]

    # --- Secuencias para la LSTM ---
    LB = cfg["lookback"]
    Xtr_seq, ytr_seq = make_sequences(X_tr_s, y_tr_s, LB)
    Xvl_seq, yvl_seq = make_sequences(X_vl_s, y_vl_s, LB)
    Xte_seq, yte_seq = make_sequences(X_te_s, y_te_s, LB)

    print(
        f"[SECUENCIAS] Train: {Xtr_seq.shape} | Val: {Xvl_seq.shape} | Test: {Xte_seq.shape}"
    )

    train_ds = TensorDataset(torch.from_numpy(Xtr_seq), torch.from_numpy(ytr_seq))
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )

    X_vl_t = torch.from_numpy(Xvl_seq)
    y_vl_t = torch.from_numpy(yvl_seq)
    X_te_t = torch.from_numpy(Xte_seq)

    # --- Baseline HAR-RV ---
    har_X_tr = align_har_features_with_sequences(train_df[fcols], ytr_seq, LB)
    har_X_te = align_har_features_with_sequences(test_df[fcols], yte_seq, LB)
    y_tr_raw_aligned = y_tr[LB : LB + len(ytr_seq)]

    har = HARRVModel()
    har.fit(har_X_tr, y_tr_raw_aligned)
    print(har.summary())

    y_pred_har_te = har.predict(har_X_te)[: len(yte_seq)]

    # --- Entrenamiento de la LSTM ---
    n_features = Xtr_seq.shape[-1]
    model = VolatilityLSTM(n_features, cfg["hidden"], cfg["dropout"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_seq_train = len(ytr_seq)

    print(f"\n[LSTM] Parámetros entrenables: {n_params:,}")
    print(
        f"  n_secuencias_train / n_params = {n_seq_train / n_params:.4f}  "
        f"({'OK (>1)' if n_seq_train / n_params > 1 else 'ADVERTENCIA: sobredeterminado'})"
    )

    print("\n[LSTM] Entrenando...")
    history = train_lstm(model, train_loader, X_vl_t, y_vl_t, cfg, device)

    # --- Predicciones en test ---
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred_lstm = model(X_te_t.to(device)).cpu().numpy()

    n_eval = min(len(yte_seq), len(y_pred_lstm), len(y_pred_har_te))

    y_true_log = y_scaler.inverse_transform(
        y_te_s[LB : LB + n_eval].reshape(-1, 1)
    ).ravel()
    y_lstm_log = y_scaler.inverse_transform(
        y_pred_lstm[:n_eval].reshape(-1, 1)
    ).flatten()
    y_har_log = y_pred_har_te[:n_eval]

    # Suavizado EWMA opcional para reducir ruido en las series.
    span = cfg["smoothing_window"]
    y_true_log = pd.Series(y_true_log).ewm(span=span, adjust=False).mean().to_numpy()
    y_lstm_log = pd.Series(y_lstm_log).ewm(span=span, adjust=False).mean().to_numpy()
    y_har_log = pd.Series(y_har_log).ewm(span=span, adjust=False).mean().to_numpy()

    # Conversión a espacio sigma (volatilidad anualizada).
    y_true = np.exp(y_true_log)
    y_lstm = np.exp(y_lstm_log)
    y_har = np.exp(y_har_log)

    # --- Evaluación comparativa ---
    print("\n[EVALUACIÓN] Desempeño en el conjunto de test:")
    m_lstm = evaluate_forecasts(y_true, y_lstm, label="LSTM")
    m_har = evaluate_forecasts(y_true, y_har, label="HAR-RV (Baseline)")

    e_lstm = y_true - y_lstm
    e_har = y_true - y_har
    dm = diebold_mariano_test(e_lstm, e_har, h=cfg["forecast_horizon"])

    metrics_df = pd.DataFrame([m_lstm, m_har])
    metrics_df.to_csv(cfg["metrics_path"], index=False)
    print(f"\n  Métricas guardadas → {cfg['metrics_path']}")

    print("\n[CHECKLIST DE VALIDACIÓN]")
    flags = {
        "R² > 0.08": m_lstm["R2"] > 0.08,
        "Dir. Acc. > 0.55": m_lstm["Dir_Acc"] > 0.55,
        "Spearman r > 0.15": m_lstm["Spearman_r"] > 0.15,
        "Spearman p < 0.05": m_lstm["Spearman_p"] < 0.05,
        "MZ F p > 0.05 (insesgado)": m_lstm["MZ_F_pval"] > 0.05,
    }
    for check, passed in flags.items():
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {check}")

    # --- Backtest de vol-timing ---
    # sigma_bar se calcula sobre el conjunto de entrenamiento para evitar look-ahead.
    sigma_bar = np.median(np.exp(y_tr[LB : LB + len(ytr_seq)])) + 1e-8

    te_log_ret = (
        np.log(raw["Close"] / raw["Close"].shift(1))
        .reindex(te_dates)
        .fillna(0.0)
        .values
    )
    ret_te_aligned = te_log_ret[LB : LB + n_eval]

    print("\n[BACKTEST] Estrategia de vol-timing (período de test)...")
    bt = run_vol_timing_backtest(ret_te_aligned, y_lstm, sigma_bar, cfg["cost_bps"])
    asset_stats = summarize_log_returns(ret_te_aligned)

    summary = pd.DataFrame(
        [
            {
                "strategy": "Vol-Timed",
                "total_return": bt["total_vol"],
                "sharpe": bt["sharpe_vol"],
                "maxdd": bt["maxdd_vol"],
                "calmar": bt["calmar_vol"],
            },
            {
                "strategy": "Baseline",
                "total_return": bt["total_base"],
                "sharpe": bt["sharpe_base"],
                "maxdd": bt["maxdd_base"],
                "calmar": bt["calmar_base"],
            },
            {
                "strategy": "Buy & Hold",
                "total_return": asset_stats["total_return"],
                "sharpe": asset_stats["sharpe"],
                "maxdd": asset_stats["maxdd"],
                "calmar": asset_stats["calmar"],
            },
        ]
    )

    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n  Mejora Sharpe vs baseline: {bt['sharpe_improvement']:+.4f}")
    print(f"  Mejora Sharpe vs buy & hold: {bt['sharpe_vs_bh']:+.4f}")

    if bt["sharpe_improvement"] > 0.2:
        print("  ✓ Mejora > 0.2 Sharpe: la predicción de vol agrega valor.")
    else:
        print("  ✗ Mejora ≤ 0.2: la predicción de vol es marginal para el sizing.")

    # --- Figura diagnóstica ---
    print("\n[GRÁFICO] Generando figura diagnóstica...")
    generate_plots(y_true, y_lstm, y_har, te_dates, history, bt, cfg)

    # --- Persistencia de artefactos ---
    with open(cfg["scaler_path"], "wb") as f:
        pickle.dump(X_scaler, f)
    with open(cfg["target_scaler_path"], "wb") as f:
        pickle.dump(y_scaler, f)
    print(
        f"  Escaladores guardados → {cfg['scaler_path']}, {cfg['target_scaler_path']}"
    )

    return {
        "raw": raw,
        "data": data,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "model": model,
        "har": har,
        "history": history,
        "metrics": {"lstm": m_lstm, "har": m_har},
        "dm": dm,
        "backtest": bt,
        "summary": summary,
        "y_true": y_true,
        "y_lstm": y_lstm,
        "y_har": y_har,
    }


# %%
if __name__ == "__main__":
    results = run_experiment()
