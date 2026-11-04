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
# El pipeline está partido en etapas re-ejecutables, cada una en su propia
# celda: :func:`prepare_data` (datos) → :func:`train_and_infer` (entrenamiento
# e inferencia) → :func:`smooth_predictions` (suavizado EWMA) →
# :func:`compute_metrics` (métricas) → :func:`run_backtest_and_strategies`
# (backtest).  :func:`run_experiment` las encadena todas para uso programático.
#
# Importar este archivo no descarga datos ni entrena modelos: las etapas son
# funciones puras y sólo se ejecutan desde las celdas bajo
# ``if __name__ == "__main__":``.  Las figuras se dibujan al final del
# cuaderno, en celdas independientes, para poder personalizarlas sin tocar las
# funciones generales ni repetir el entrenamiento.

# %%
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import r2_score
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
    purge_boundary_labels,
    scale_splits,
    temporal_split,
)
from thesis.volatility_evaluation import (
    calculate_adaptability,
    diebold_mariano_test,
    evaluate_forecasts,
    run_vol_timing_backtest,
    summarize_returns,
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
    "train_end": "2017-01-01",
    "val_end": "2022-01-01",
    # "end_date": "2027-01-0",
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
    "cost_bps": 0.0,
    # Salidas
    "model_path": "lstm_vol_model.pt",
    "scaler_path": "scaler.pkl",
    "metrics_path": "metrics.csv",
    "report_path": "report.md",
    "config_path": "model_config.json",
    "target_scaler_path": "target_scaler.pkl",
    # Predicciones crudas (log, sin suavizar) para re-ejecutar el post-proceso
    "inference_path": "inference.pkl",
}


# %%
def _merge_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combina :data:`CONFIG` con las sobreescrituras del usuario.

    ``inference_path`` se deriva de ``model_path`` (mismo directorio, nombre
    ``inference.pkl``) cuando no se indica explícitamente, de modo que el
    artefacto de inferencia viaja siempre junto al modelo entrenado.
    """
    cfg = {**CONFIG, **(config or {})}
    if "inference_path" not in (config or {}):
        cfg["inference_path"] = str(Path(cfg["model_path"]).with_name("inference.pkl"))
    return cfg


# %%
def prepare_data(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Etapa 1: descarga y preparación de los datos del experimento.

    Descarga el OHLCV, construye el dataset de volatilidad, hace la división
    temporal train/val/test, purga las filas de train y validación cuyos
    labels cruzan la frontera del split, escala los splits (el escalador se
    ajusta **solo** con el train), arma las secuencias para la LSTM y ajusta
    el baseline HAR-RV.  No entrena la LSTM ni produce predicciones.

    Unidades
    --------
    ``target_log_vol`` es el log de la volatilidad realizada *forward*
    anualizada; los arrays ``X_*_s`` / ``y_*_s`` están estandarizados con la
    media y la desviación estándar del **train**.

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.  Si es ``None`` se usa la
        configuración por defecto.

    Returns
    -------
    dict[str, Any]
        Datos crudos, dataset, particiones, arrays (en natural y escalados),
        escaladores, secuencias, ``DataLoader`` de train, tensores de val/test,
        matrices HAR alineadas y el vector de targets de train alineado con
        las secuencias.
    """
    cfg = _merge_config(config)

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

    # --- Purga de frontera (fuga de labels) ---
    # El target forward de las últimas ``horizon`` filas de train usa retornos
    # de validación, y el de las últimas de validación usa retornos de test.
    # Se eliminan esas filas antes de construir X/y y de escalar.
    train_df = purge_boundary_labels(train_df, cfg["forecast_horizon"])
    val_df = purge_boundary_labels(val_df, cfg["forecast_horizon"])
    fcols = cfg["features"]

    X_tr = train_df[fcols].values.astype(np.float32)
    X_vl = val_df[fcols].values.astype(np.float32)
    X_te = test_df[fcols].values.astype(np.float32)

    y_tr = train_df["target_log_vol"].values.astype(np.float32)
    y_vl = val_df["target_log_vol"].values.astype(np.float32)
    y_te = test_df["target_log_vol"].values.astype(np.float32)

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

    return {
        "raw": raw,
        "data": data,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_tr": X_tr,
        "X_vl": X_vl,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_vl": y_vl,
        "y_te": y_te,
        "X_tr_s": X_tr_s,
        "X_vl_s": X_vl_s,
        "X_te_s": X_te_s,
        "y_tr_s": y_tr_s,
        "y_vl_s": y_vl_s,
        "y_te_s": y_te_s,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "Xtr_seq": Xtr_seq,
        "ytr_seq": ytr_seq,
        "Xvl_seq": Xvl_seq,
        "yvl_seq": yvl_seq,
        "Xte_seq": Xte_seq,
        "yte_seq": yte_seq,
        "train_loader": train_loader,
        "X_vl_t": X_vl_t,
        "y_vl_t": y_vl_t,
        "X_te_t": X_te_t,
        "har_X_tr": har_X_tr,
        "har_X_te": har_X_te,
        "har": har,
        "y_tr_raw_aligned": y_tr_raw_aligned,
    }


# %%
def train_and_infer(
    config: dict[str, Any] | None = None,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Etapa 2: entrenamiento de la LSTM e inferencia sobre el test.

    Entrena la LSTM con early stopping, carga el mejor checkpoint, predice
    sobre el test y persiste las predicciones **en espacio log y sin
    suavizar** en ``inference.pkl``.

    El suavizado EWMA no se aplica aquí: vive en :func:`smooth_predictions`
    para poder cambiar ``smoothing_window`` sin volver a entrenar.  La
    semilla se fija en esta etapa (y no en el post-proceso) para que el
    entrenamiento sea reproducible.

    Unidades
    --------
    ``y_*_log``: log de la volatilidad anualizada (predicciones crudas).
    ``sigma_bar``: volatilidad anualizada (sigma), mediana del **train**.
    ``ret_te_aligned``: retornos logarítmicos diarios del test alineados con
    el forecast operable (``dates[j+1]``), de longitud ``n_eval - 1``.

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.
    prepared : dict[str, Any] | None
        Salida de :func:`prepare_data`.

    Returns
    -------
    dict[str, Any]
        ``model``, ``history`` y ``har``.
    """
    cfg = _merge_config(config)
    if prepared is None:
        raise ValueError("train_and_infer requiere el diccionario 'prepared'.")

    set_volatility_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DISPOSITIVO] Usando: {device}")

    LB = cfg["lookback"]
    Xtr_seq = prepared["Xtr_seq"]
    ytr_seq = prepared["ytr_seq"]
    yte_seq = prepared["yte_seq"]
    har = prepared["har"]

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
    history = train_lstm(
        model,
        prepared["train_loader"],
        prepared["X_vl_t"],
        prepared["y_vl_t"],
        cfg,
        device,
    )

    # --- Predicciones en test ---
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred_lstm = model(prepared["X_te_t"].to(device)).cpu().numpy()

    y_pred_har_te = har.predict(prepared["har_X_te"])[: len(yte_seq)]

    n_eval = min(len(yte_seq), len(y_pred_lstm), len(y_pred_har_te))

    y_scaler = prepared["y_scaler"]
    y_true_log = y_scaler.inverse_transform(
        prepared["y_te_s"][LB : LB + n_eval].reshape(-1, 1)
    ).ravel()
    y_lstm_log = y_scaler.inverse_transform(
        y_pred_lstm[:n_eval].reshape(-1, 1)
    ).flatten()
    y_har_log = y_pred_har_te[:n_eval]

    # sigma_bar: mediana del TRAIN en espacio sigma (evita look-ahead).
    sigma_bar = float(np.median(np.exp(prepared["y_tr_raw_aligned"])) + 1e-8)

    te_dates = prepared["test_df"].index
    te_log_ret = (
        np.log(prepared["raw"]["Close"] / prepared["raw"]["Close"].shift(1))
        .reindex(te_dates)
        .fillna(0.0)
        .values
    )
    # Alineación forecast ↔ retorno operado: el forecast ``j`` se origina al
    # cierre de ``dates[j-1]`` y predice los retornos de ``dates[j+1 ... j+H]``;
    # por tanto el retorno operable es el de ``dates[j+1]`` (el último forecast
    # no es operable porque su ventana cae fuera del test).
    ret_te_aligned = te_log_ret[LB + 1 : LB + n_eval]
    dates = te_dates[LB : LB + n_eval]
    bt_dates = te_dates[LB + 1 : LB + n_eval]

    # --- Persistencia de artefactos ---
    with open(cfg["inference_path"], "wb") as f:
        pickle.dump(
            {
                "dates": dates,
                "bt_dates": bt_dates,
                "y_true_log": y_true_log,
                "y_lstm_log": y_lstm_log,
                "y_har_log": y_har_log,
                "sigma_bar": sigma_bar,
                "ret_te_aligned": ret_te_aligned,
                "history": history,
            },
            f,
        )
    print(f"  Inferencia (log, sin suavizar) guardada → {cfg['inference_path']}")

    with open(cfg["scaler_path"], "wb") as f:
        pickle.dump(prepared["X_scaler"], f)
    with open(cfg["target_scaler_path"], "wb") as f:
        pickle.dump(prepared["y_scaler"], f)
    print(
        f"  Escaladores guardados → {cfg['scaler_path']}, {cfg['target_scaler_path']}"
    )

    return {"model": model, "history": history, "har": har}


# %%
def smooth_predictions(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Etapa 3: suavizado EWMA de las predicciones (re-ejecutable).

    Lee ``inference.pkl`` (predicciones crudas en espacio log) y aplica un
    suavizado exponencial **causal** ``ewm(span=smoothing_window,
    adjust=False)`` sobre el período de test.  No requiere volver a
    entrenar: cambiar ``smoothing_window`` y re-ejecutar esta celda actualiza
    métricas y figuras sin tocar el modelo ni los datos.

    La **verdad no se suaviza**: ``y_true`` es el target crudo, de modo que la
    evaluación mide al modelo contra el objetivo declarado.  El EWMA se aplica
    sólo a los pronósticos (``y_lstm``/``y_har``) y sirve como vista
    secundaria para trading; las versiones sin suavizar se exponen como
    ``y_*_raw``.

    Unidades
    --------
    Entrada: log de la volatilidad anualizada.  Salida: volatilidad
    anualizada (sigma).

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.

    Returns
    -------
    dict[str, Any]
        ``y_true``, ``y_lstm``, ``y_har`` (sigma), ``y_true_raw``,
        ``y_lstm_raw``, ``y_har_raw``, ``sigma_bar``, ``ret_te_aligned``,
        ``dates`` y ``bt_dates``.
    """
    cfg = _merge_config(config)
    if not Path(cfg["inference_path"]).exists():
        raise FileNotFoundError(
            f"No existe {cfg['inference_path']}: ejecuta antes la etapa de "
            "entrenamiento (train_and_infer)."
        )
    with open(cfg["inference_path"], "rb") as f:
        art = pickle.load(f)

    if "bt_dates" not in art:
        raise KeyError(
            f"{cfg['inference_path']} no contiene 'bt_dates': vuelve a ejecutar "
            "la etapa de entrenamiento (train_and_infer) para regenerarlo."
        )

    span = cfg["smoothing_window"]
    print(f"\n[SUAVIZADO] EWMA causal (span={span}) sobre el período de test.")
    print(f"  Origen: {cfg['inference_path']}")

    def _ewm(x: np.ndarray) -> np.ndarray:
        return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()

    # La verdad se conserva cruda: el suavizado sólo afecta a los pronósticos.
    y_true_log = _ewm(art["y_true_log"])
    y_lstm_log = _ewm(art["y_lstm_log"])
    y_har_log = _ewm(art["y_har_log"])

    # Conversión a espacio sigma (volatilidad anualizada).
    return {
        "y_true": np.exp(y_true_log),
        "y_lstm": np.exp(y_lstm_log),
        "y_har": np.exp(y_har_log),
        "y_true_raw": np.exp(y_true_log),
        "y_lstm_raw": np.exp(art["y_lstm_log"]),
        "y_har_raw": np.exp(art["y_har_log"]),
        "sigma_bar": art["sigma_bar"],
        "ret_te_aligned": art["ret_te_aligned"],
        "dates": art["dates"],
        "bt_dates": art["bt_dates"],
    }


# %%
def compute_metrics(
    config: dict[str, Any] | None = None,
    smoothed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Etapa 4: métricas comparativas LSTM vs HAR-RV.

    Evalúa los pronósticos **crudos** contra la verdad cruda (métricas
    primarias, las que se escriben en ``metrics.csv``), aplica el test de
    Diebold-Mariano, escribe ``metrics.csv`` e imprime el checklist de
    validación.  Es re-ejecutable sin reentrenar: basta con re-ejecutar tras
    cambiar ``smoothing_window``.

    Las métricas calculadas sobre los pronósticos **suavizados** se imprimen
    como vista secundaria para trading; no son el estimando declarado (la
    verdad nunca se suaviza).

    Unidades
    --------
    Todas las métricas se calculan en espacio sigma (volatilidad anualizada).

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.
    smoothed : dict[str, Any] | None
        Salida de :func:`smooth_predictions`.  Si es ``None`` se calcula.

    Returns
    -------
    dict[str, Any]
        ``lstm`` y ``har`` (diccionarios de métricas primarias raw-vs-raw),
        ``lstm_smoothed`` y ``har_smoothed`` (vista secundaria), ``dm`` (test
        Diebold-Mariano) y ``metrics_df`` (tabla escrita en disco).
    """
    cfg = _merge_config(config)
    if smoothed is None:
        smoothed = smooth_predictions(cfg)

    y_true = smoothed["y_true"]
    y_lstm_raw = smoothed["y_lstm_raw"]
    y_har_raw = smoothed["y_har_raw"]
    y_lstm = smoothed["y_lstm"]
    y_har = smoothed["y_har"]

    # --- Evaluación comparativa (métricas primarias: raw vs raw) ---
    print("\n[EVALUACIÓN] Desempeño en el conjunto de test (raw vs raw):")
    m_lstm = evaluate_forecasts(y_true, y_lstm_raw, label="LSTM")
    m_har = evaluate_forecasts(y_true, y_har_raw, label="HAR-RV (Baseline)")

    e_lstm = y_true - y_lstm_raw
    e_har = y_true - y_har_raw
    dm = diebold_mariano_test(e_lstm, e_har, h=cfg["forecast_horizon"])

    metrics_df = pd.DataFrame([m_lstm, m_har])
    metrics_df.to_csv(cfg["metrics_path"], index=False)
    print(f"\n  Métricas guardadas → {cfg['metrics_path']}")

    # --- Vista secundaria: pronósticos suavizados (sólo para trading) ---
    print(
        "\n[NOTA] Las métricas suavizadas siguientes son una vista secundaria "
        "para trading; el estimando declarado es el raw-vs-raw de arriba."
    )
    m_lstm_smooth = evaluate_forecasts(y_true, y_lstm, label="LSTM (suavizado)")
    m_har_smooth = evaluate_forecasts(y_true, y_har, label="HAR-RV (suavizado)")

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

    return {
        "lstm": m_lstm,
        "har": m_har,
        "lstm_smoothed": m_lstm_smooth,
        "har_smoothed": m_har_smooth,
        "dm": dm,
        "metrics_df": metrics_df,
    }


# %%
def run_backtest_and_strategies(
    config: dict[str, Any] | None = None,
    smoothed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Etapa 5: backtest de vol-timing y comparación de estrategias.

    Re-ejecutable sin reentrenar ni re-metricar: cambiar ``cost_bps`` y
    re-ejecutar esta celda actualiza el backtest.

    Unidades
    --------
    ``ret_te_aligned``: retornos logarítmicos diarios, alineados con el
    forecast operable (``bt_dates``).  ``y_lstm``: volatilidad anualizada
    predicha.  ``sigma_bar``: volatilidad anualizada de referencia (mediana
    del train).  La señal base es de momentum (signo de la media móvil de los
    retornos) y se calcula dentro de ``run_vol_timing_backtest``, con un
    desplazamiento causal de una barra.

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.
    smoothed : dict[str, Any] | None
        Salida de :func:`smooth_predictions`.  Si es ``None`` se calcula.

    Returns
    -------
    dict[str, Any]
        ``backtest`` (series y métricas de la estrategia), ``summary``
        (DataFrame con las tres estrategias), ``asset_stats`` (buy & hold) y
        ``bt_dates`` (fechas de los retornos operados).
    """
    cfg = _merge_config(config)
    if smoothed is None:
        smoothed = smooth_predictions(cfg)

    y_lstm = smoothed["y_lstm"]
    sigma_bar = smoothed["sigma_bar"]
    ret_te_aligned = smoothed["ret_te_aligned"]
    bt_dates = smoothed["bt_dates"]

    print("\n[BACKTEST] Estrategia de vol-timing (período de test)...")
    # El último forecast no es operable (su ventana cae fuera del test): se
    # descarta para que ``ret_te_aligned``, ``y_lstm[:-1]`` y ``bt_dates``
    # tengan la misma longitud.
    bt = run_vol_timing_backtest(
        ret_te_aligned, y_lstm[:-1], sigma_bar, cfg["cost_bps"]
    )
    asset_stats = summarize_returns(np.expm1(ret_te_aligned))

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

    return {
        "backtest": bt,
        "summary": summary,
        "asset_stats": asset_stats,
        "bt_dates": bt_dates,
    }


# %%
def run_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo encadenando las etapas.

    Equivale a ejecutar, en orden, :func:`prepare_data`,
    :func:`train_and_infer`, :func:`smooth_predictions`,
    :func:`compute_metrics` y :func:`run_backtest_and_strategies`.  Se usa
    desde código (tests, scripts); en el cuaderno cada etapa tiene su propia
    celda para poder re-ejecutar el post-proceso sin reentrenar.

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
    cfg = _merge_config(config)

    prepared = prepare_data(cfg)
    trained = train_and_infer(cfg, prepared)
    smoothed = smooth_predictions(cfg)
    metrics = compute_metrics(cfg, smoothed)
    backtest = run_backtest_and_strategies(cfg, smoothed)

    return {
        "raw": prepared["raw"],
        "data": prepared["data"],
        "train_df": prepared["train_df"],
        "val_df": prepared["val_df"],
        "test_df": prepared["test_df"],
        "model": trained["model"],
        "har": trained["har"],
        "history": trained["history"],
        "metrics": {"lstm": metrics["lstm"], "har": metrics["har"]},
        "dm": metrics["dm"],
        "backtest": backtest["backtest"],
        "summary": backtest["summary"],
        "y_true": smoothed["y_true"],
        "y_lstm": smoothed["y_lstm"],
        "y_har": smoothed["y_har"],
    }


# %%
# ## Ejecución por etapas
#
# Cada etapa tiene su propia celda para poder re-ejecutar el post-proceso
# (suavizado, métricas y backtest) sin volver a entrenar ni volver a
# descargar los datos.

# %%
# ### Etapas 1-2: datos y entrenamiento
#
# Descarga los datos, entrena la LSTM una sola vez y escribe
# ``inference.pkl`` con las predicciones crudas (log, sin suavizar).

if __name__ == "__main__":
    prepared = prepare_data(CONFIG)
    trained = train_and_infer(CONFIG, prepared)


# %%
# ### Etapa 3: suavizado
#
# Cambiar ``CONFIG["smoothing_window"]`` y re-ejecutar **esta celda** (y las
# siguientes) actualiza métricas y figuras sin reentrenar.

if __name__ == "__main__":
    smoothed = smooth_predictions(CONFIG)


# %%
# ### Etapa 4: métricas

if __name__ == "__main__":
    metrics = compute_metrics(CONFIG, smoothed)


# %%
# ### Etapa 5: backtest y estrategias
#
# Cambiar ``CONFIG["cost_bps"]`` y re-ejecutar **esta celda** actualiza el
# backtest sin reentrenar ni re-metricar.

if __name__ == "__main__":
    backtest = run_backtest_and_strategies(CONFIG, smoothed)


# %%
# ## Gráficas
#
# Las figuras se dibujan *in situ* en este cuaderno, y no dentro de
# ``src/thesis/volatility_evaluation.py``, para poder personalizarlas sin
# tocar las funciones generales y sin reiniciar el kernel: basta con
# re-ejecutar estas celdas, que sólo dependen del diccionario ``smoothed``
# que devuelve :func:`smooth_predictions`.
#
# Hay una figura por celda, de modo que cada una se puede modificar y volver
# a dibujar de forma independiente.

# %%
# ### Estilo
#
# Fondo blanco, tipografía estándar de matplotlib y cuadrícula muy tenue.
# Todos los ajustes de estilo viven en esta celda: cambiar aquí los aplica a
# todas las figuras.

if __name__ == "__main__":
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
        }
    )

    # Nombres de los archivos PNG donde se guarda cada figura.
    PLOT_PATHS = {
        "loss": "lstm_vol_loss.png",
        "predictions": "lstm_vol_predictions.png",
        "timeseries": "lstm_vol_timeseries.png",
        "adaptability": "lstm_vol_adaptability.png",
        "backtest": "lstm_vol_backtest.png",
        "ranks": "lstm_vol_ranks.png",
        "april2025": "lstm_vol_april2025.png",
    }

# %%
# ### Datos comunes
#
# Todo lo que necesitan las figuras viene de ``smoothed`` (etapa 3), de modo
# que estas celdas se pueden re-ejecutar sin volver a entrenar la LSTM.

if __name__ == "__main__":
    # ``y_true`` es el target **crudo** (la verdad no se suaviza) y
    # ``y_lstm``/``y_har`` son los pronósticos suavizados (EWMA), que son los
    # que se usan para operar.
    y_true = smoothed["y_true"]
    y_lstm = smoothed["y_lstm"]
    y_har = smoothed["y_har"]

    # ``dates`` viene alineado con las predicciones (mismo ``n_eval`` que al
    # entrenar), así que no hay que volver a indexar el DataFrame de test.
    dates = smoothed["dates"]

    # ``bt_dates`` son las fechas de los retornos operados por el backtest:
    # el forecast ``j`` opera el retorno de ``bt_dates[j]`` (longitud
    # ``n_eval - 1``, porque el último forecast no es operable).
    bt_dates = smoothed["bt_dates"]

# %%
# ### 1. Pérdida de entrenamiento

if __name__ == "__main__":
    history = trained["history"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], color="C0", lw=1.2, label="Train")
    ax.plot(history["val_loss"], color="C1", lw=1.2, label="Validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida LINEX")
    ax.set_title("Pérdida de entrenamiento")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["loss"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 2. Predicción frente a realidad

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_lstm, s=6, alpha=0.35, color="C0")
    lo = min(y_true.min(), y_lstm.min())
    hi = max(y_true.max(), y_lstm.max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.0, alpha=0.6)
    ax.set_xlabel("Volatilidad real (anualizada)")
    ax.set_ylabel("Volatilidad predicha (anualizada)")
    ax.set_title("LSTM: predicho vs real")
    ax.text(
        0.05,
        0.93,
        f"R² = {r2_score(y_true, y_lstm):.4f}",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["predictions"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 3. Evolución de la volatilidad

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, y_true, color="black", lw=1.2, label="Real")
    ax.plot(dates, y_lstm, color="C0", lw=1.0, label="LSTM")
    ax.plot(dates, y_har, color="C1", lw=1.0, ls="--", label="HAR-RV")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Volatilidad anualizada")
    ax.set_title("Volatilidad realizada forward — test")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["timeseries"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 4. Adaptabilidad post-shock

if __name__ == "__main__":
    horizon = 11
    mae_lstm, hl_lstm = calculate_adaptability(
        y_true, y_lstm, horizon=horizon, threshold=2.5
    )
    mae_har, hl_har = calculate_adaptability(
        y_true, y_har, horizon=horizon, threshold=2.5
    )
    k = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        k,
        mae_lstm,
        color="C0",
        marker="o",
        ms=4,
        lw=1.2,
        label=f"LSTM (t½ = {hl_lstm:.1f} d)" if not np.isnan(hl_lstm) else "LSTM",
    )
    ax.plot(
        k,
        mae_har,
        color="C1",
        marker="s",
        ms=4,
        lw=1.2,
        ls="--",
        label=f"HAR-RV (t½ = {hl_har:.1f} d)" if not np.isnan(hl_har) else "HAR-RV",
    )
    ax.set_xlabel("Días hábiles post-shock (k)")
    ax.set_ylabel("Error medio absoluto (MAE)")
    ax.set_title("Adaptabilidad post-shock")
    ax.set_xticks(k[::2])
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["adaptability"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 5. Backtest de vol-timing

if __name__ == "__main__":
    # Se lee del resultado de la etapa 5 para que un cambio de ``cost_bps``
    # se refleje al re-dibujar la figura.
    bt = backtest["backtest"]

    # Los retornos del backtest son simples: la riqueza es ``cumprod(1 + r)``
    # y las fechas son las del retorno operado (``bt_dates``).
    equity_vol = np.cumprod(1.0 + bt["ret_vol"])
    equity_base = np.cumprod(1.0 + bt["ret_base"])
    equity_bh = np.cumprod(1.0 + bt["ret_bh"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        bt_dates,
        equity_vol,
        color="C0",
        lw=1.3,
        label=f"Con Volatilidad (SR = {bt['sharpe_vol']:.2f})",
    )
    ax.plot(
        bt_dates,
        equity_base,
        color="C1",
        lw=1.3,
        ls="--",
        label=f"Base (SR = {bt['sharpe_base']:.2f})",
    )

    ax.plot(
        bt_dates,
        equity_bh,
        color="black",
        lw=1.3,
        ls=":",
        label=f"Buy & Hold (SR = {bt['sharpe_bh']:.2f})",
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno acumulado")
    ax.set_title("Backtest de vol-timing — test")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["backtest"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 6. Correlación de rangos

if __name__ == "__main__":
    rank_true = stats.rankdata(y_true)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(
        rank_true,
        stats.rankdata(y_lstm),
        s=5,
        alpha=0.3,
        color="C0",
        label="LSTM",
    )
    ax.scatter(
        rank_true,
        stats.rankdata(y_har),
        s=5,
        alpha=0.3,
        color="C1",
        label="HAR-RV",
    )
    ax.set_xlabel("Rango: volatilidad real")
    ax.set_ylabel("Rango: volatilidad predicha")
    ax.set_title("Correlación de rangos: LSTM vs HAR-RV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["ranks"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 7. Abril 2025: tarifas
#
# Volatilidad realizada forward frente a la predicción de la LSTM durante
# abril de 2025, con los anuncios de las tarifas recíprocas marcados:
# "Liberation Day" (2 de abril) y la pausa de 90 días acordada con China
# (9 de abril).
#
# El filtrado es exclusivamente temporal: no se descartan observaciones por
# magnitud de volatilidad, porque eso introduciría un sesgo de selección.

if __name__ == "__main__":
    mask = (dates >= "2025-01-01") & (dates <= "2025-08-01")

    eventos = [
        (pd.Timestamp("2025-04-02"), "2 abr: Liberation Day", "2 abr"),
        (pd.Timestamp("2025-04-09"), "9 abr: pausa de 90 días", "9 abr"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        dates[mask],
        y_true[mask],
        color="black",
        lw=1.2,
        label="Realizada (forward)",
    )
    ax.plot(
        dates[mask],
        y_lstm[mask],
        color="C0",
        lw=1.0,
        label="Predicción LSTM",
    )
    ax.plot(
        dates[mask],
        y_har[mask],
        color="C1",
        lw=1.0,
        ls="--",
        label="HAR-RV",
    )
    for fecha, etiqueta, texto in eventos:
        ax.axvline(fecha, color="C3", ls="--", lw=1.0, alpha=0.9, label=etiqueta)
        ax.annotate(
            texto,
            xy=(fecha, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=8,
            color="C3",
        )

    # Holgura arriba para que anotaciones y leyenda no tapen las series.
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.25 * (y_hi - y_lo))

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Volatilidad anualizada")
    ax.set_title("Volatilidad realizada vs predicción LSTM — abril de 2025")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["april2025"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
