"""Tests de integración con datos reales descargados de Yahoo Finance.

Estos tests requieren conexión a red, por lo que están marcados con
``integration`` y se omiten por defecto.  Para ejecutarlos:

```bash
uv run pytest -m integration
```

El fixture de sesión descarga ``SPY`` una sola vez y lo comparte entre
todos los tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

pytestmark = pytest.mark.integration

TICKER = "SPY"
START = "2015-01-01"
TRAIN_END = "2020-01-01"
VAL_END = "2022-01-01"
HORIZON = 5
LOOKBACK = 5


@pytest.fixture(scope="session")
def real_ohlcv() -> pd.DataFrame:
    """OHLCV real de ``SPY`` descargado de Yahoo Finance."""
    return download_ohlcv(TICKER, start=START)


@pytest.fixture(scope="session")
def real_dataset(real_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Dataset de volatilidad construido sobre los datos reales."""
    return build_volatility_dataset(real_ohlcv, horizon=HORIZON)


@pytest.fixture(scope="session")
def real_splits(
    real_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """División temporal del dataset real."""
    return temporal_split(real_dataset, TRAIN_END, VAL_END)


def test_download_real_ohlcv_shape(real_ohlcv: pd.DataFrame) -> None:
    assert isinstance(real_ohlcv, pd.DataFrame)
    assert set(real_ohlcv.columns) == {"Open", "High", "Low", "Close", "Volume"}
    assert len(real_ohlcv) > 1000
    assert real_ohlcv.index.is_monotonic_increasing
    assert not real_ohlcv.index.has_duplicates


def test_download_real_ohlcv_no_nan_and_positive(real_ohlcv: pd.DataFrame) -> None:
    assert not real_ohlcv.isna().any().any()
    assert (real_ohlcv[["Open", "High", "Low", "Close"]] > 0).all().all()
    # Invariantes OHLC de mercado.
    assert (real_ohlcv["High"] >= real_ohlcv["Low"]).all()
    assert (real_ohlcv["High"] >= real_ohlcv[["Open", "Close"]].max(axis=1)).all()
    assert (real_ohlcv["Low"] <= real_ohlcv[["Open", "Close"]].min(axis=1)).all()
    assert (real_ohlcv["Volume"] > 0).all()


def test_real_ohlcv_covers_requested_period(real_ohlcv: pd.DataFrame) -> None:
    assert real_ohlcv.index[0] >= pd.Timestamp(START)
    assert real_ohlcv.index[0] >= pd.Timestamp(START)


def test_count_extreme_moves_real(real_ohlcv: pd.DataFrame) -> None:
    n = count_extreme_moves(real_ohlcv)
    assert n >= 0
    assert n < len(real_ohlcv) * 0.2


def test_real_dataset_no_nan(real_dataset: pd.DataFrame) -> None:
    assert not real_dataset.isna().any().any()
    assert "target_log_vol" in real_dataset.columns
    assert len(real_dataset) > 1000


def test_real_dataset_target_is_log_volatility(real_dataset: pd.DataFrame) -> None:
    target = real_dataset["target_log_vol"]
    # El target es log de volatilidad anualizada, acotado por construcción.
    assert target.max() < 5.0
    assert target.min() > -20.0


def test_real_features_are_causal(
    real_ohlcv: pd.DataFrame, real_dataset: pd.DataFrame
) -> None:
    """Truncar la serie no debe alterar las features ya calculadas."""
    from thesis.volatility import compute_volatility_features

    t = len(real_ohlcv) // 2
    truncated = compute_volatility_features(real_ohlcv.iloc[: t + 1])
    full = compute_volatility_features(real_ohlcv)
    for col in full.columns:
        value_full = full[col].iloc[t]
        if pd.isna(value_full):
            continue
        assert truncated[col].iloc[-1] == pytest.approx(value_full, rel=1e-12)


def test_real_split_boundaries(
    real_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train, val, test = real_splits
    assert train.index.max() <= pd.Timestamp(TRAIN_END)
    assert val.index.min() > pd.Timestamp(TRAIN_END)
    assert val.index.max() <= pd.Timestamp(VAL_END)
    assert test.index.min() > pd.Timestamp(VAL_END)
    assert len(train) + len(val) + len(test) == sum(len(x) for x in real_splits)


def test_real_scaler_does_not_leak_test_data(
    real_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """El escalador debe ajustarse solo con el train."""
    train, val, test = real_splits
    cols = ["rv_daily", "rv_weekly", "rv_monthly"]
    X_tr = train[cols].values
    X_te = test[cols].values
    scaled = scale_splits(
        X_tr,
        val[cols].values,
        X_te,
        train["target_log_vol"].values,
        val["target_log_vol"].values,
        test["target_log_vol"].values,
    )
    np.testing.assert_allclose(scaled["X_scaler"].mean_, X_tr.mean(axis=0), rtol=1e-10)
    # La media del test (con información futura) difiere de la del train.
    assert not np.allclose(scaled["X_scaler"].mean_, X_te.mean(axis=0))


def test_real_sequences_align_with_har(
    real_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """La alineación HAR debe apuntar al último día de cada secuencia."""
    train, _, _ = real_splits
    cols = ["rv_daily", "rv_weekly", "rv_monthly"]
    scaled = scale_splits(
        train[cols].values,
        train[cols].values,
        train[cols].values,
        train["target_log_vol"].values,
        train["target_log_vol"].values,
        train["target_log_vol"].values,
    )
    _, ytr_seq = make_sequences(scaled["X_train"], scaled["y_train"], LOOKBACK)
    har = align_har_features_with_sequences(train[cols], ytr_seq, LOOKBACK)
    assert len(har) == len(ytr_seq)
    # La fila i del HAR usa features RAW del día i + lookback - 1 del train,
    # que es el último día observado de la secuencia i.
    for i in [0, 5, len(har) // 2]:
        expected = train["rv_daily"].values[i + LOOKBACK - 1]
        assert har["rv_daily"].iloc[i] == pytest.approx(expected)


def test_real_har_rv_fits_and_predicts(
    real_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    from thesis.volatility_models import HARRVModel

    train, _, test = real_splits
    cols = ["rv_daily", "rv_weekly", "rv_monthly"]
    X_train = train[cols].copy()
    y_train = train["target_log_vol"].values[LOOKBACK:]
    X_train = X_train.iloc[LOOKBACK - 1 : LOOKBACK - 1 + len(y_train)]

    model = HARRVModel()
    model.fit(X_train, y_train)
    preds = model.predict(test[cols].iloc[LOOKBACK - 1 :].reset_index(drop=True))
    assert len(preds) == len(test) - LOOKBACK + 1
    assert np.isfinite(preds).all()
    # Las predicciones deben estar en el rango lógico de log-volatilidad.
    assert preds.min() > -10.0
    assert preds.max() < 5.0


def test_real_lstm_trains_and_predicts(
    real_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Entrenamiento corto de la LSTM sobre features reales."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from thesis.volatility_models import VolatilityLSTM, train_lstm

    train, val, test = real_splits
    cols = [
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
    ]
    scaled = scale_splits(
        train[cols].values,
        val[cols].values,
        test[cols].values,
        train["target_log_vol"].values,
        val["target_log_vol"].values,
        test["target_log_vol"].values,
    )
    Xtr, ytr = make_sequences(scaled["X_train"], scaled["y_train"], LOOKBACK)
    Xvl, yvl = make_sequences(scaled["X_val"], scaled["y_val"], LOOKBACK)
    Xte, _ = make_sequences(scaled["X_test"], scaled["y_test"], LOOKBACK)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )
    model = VolatilityLSTM(n_features=Xtr.shape[-1], hidden=8, dropout=0.1)
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 3,
        "patience": 2,
        "min_epochs": 1,
        "grad_clip": 1.0,
        "linex_b": 0.5,
        "model_path": str(tmp_path / "m.pt"),
    }
    history = train_lstm(
        model,
        loader,
        torch.from_numpy(Xvl),
        torch.from_numpy(yvl),
        config,
        torch.device("cpu"),
    )
    assert len(history["train_loss"]) == 3
    assert all(np.isfinite(v) for v in history["val_loss"])

    model.load_state_dict(torch.load(config["model_path"], map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(Xte)).numpy()
    assert preds.shape == (len(Xte),)
    assert np.isfinite(preds).all()


def test_real_purged_labels_do_not_cross_split_boundaries(
    real_ohlcv: pd.DataFrame,
) -> None:
    """Los labels del train purgado no dependen de los retornos de validación.

    Se corrompe la cola de la serie (validación y test) y se comprueba que los
    targets del train purgado no cambian; sin purga, los últimos ``HORIZON``
    labels del train sí cambian.
    """
    data = build_volatility_dataset(real_ohlcv, horizon=HORIZON)
    train, _, _ = temporal_split(data, TRAIN_END, VAL_END)
    purged = purge_boundary_labels(train, HORIZON)

    corrupta = real_ohlcv.copy()
    futura = corrupta.index > train.index[-1]
    corrupta.loc[futura, ["Open", "High", "Low", "Close"]] *= 3.0
    data_corrupta = build_volatility_dataset(corrupta, horizon=HORIZON)
    train_corrupto, _, _ = temporal_split(data_corrupta, TRAIN_END, VAL_END)
    purged_corrupto = purge_boundary_labels(train_corrupto, HORIZON)

    assert len(purged) == len(purged_corrupto)
    np.testing.assert_allclose(
        purged["target_log_vol"].to_numpy(),
        purged_corrupto["target_log_vol"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # Sin purga, los últimos labels del train sí dependen de la validación.
    assert not np.allclose(
        train["target_log_vol"].to_numpy()[-HORIZON:],
        train_corrupto["target_log_vol"].to_numpy()[-HORIZON:],
    )


def test_real_backtest_and_metrics(
    real_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    real_ohlcv: pd.DataFrame,
) -> None:
    """Métricas y backtest sobre predicciones sintéticas ancladas a datos reales."""
    from thesis.volatility_evaluation import (
        calculate_adaptability,
        diebold_mariano_test,
        evaluate_forecasts,
        run_vol_timing_backtest,
        summarize_returns,
    )

    _, _, test = real_splits
    y_true = np.exp(test["target_log_vol"].values[:500])
    y_pred = y_true * (1 + np.random.default_rng(0).normal(0, 0.05, 500))
    metrics = evaluate_forecasts(y_true, y_pred, label="Real")
    assert metrics["N"] == 500
    assert np.isfinite(metrics["RMSE"])

    dm = diebold_mariano_test(y_true - y_pred, y_true - y_true * 1.02, h=HORIZON)
    assert np.isfinite(dm["DM_stat"])

    ret = (
        np.log(real_ohlcv["Close"]).diff().reindex(test.index).fillna(0.0).values[:500]
    )
    sigma_bar = float(np.median(y_true))
    # Señal base de momentum: se calcula dentro del backtest (signo de la
    # media móvil de los retornos, con desplazamiento de una barra).
    bt = run_vol_timing_backtest(ret, y_pred, sigma_bar, cost_bps=5)
    assert len(bt["ret_vol"]) == len(ret)
    assert np.isfinite(bt["sharpe_vol"])
    # El backtest devuelve retornos simples: buy & hold es ``expm1`` de los
    # log-retornos.
    np.testing.assert_allclose(bt["ret_bh"], np.expm1(ret), rtol=1e-12)

    asset = summarize_returns(np.expm1(ret))
    assert np.isfinite(asset["total_return"])
    assert asset["total_return"] == pytest.approx(float(np.prod(1.0 + np.expm1(ret))) - 1.0)

    mae, _ = calculate_adaptability(y_true, y_pred, horizon=10)
    assert len(mae) == 10


def test_real_predictions_invariant_to_future_data() -> None:
    """Las predicciones no deben depender de datos posteriores a cada observación.

    Se entrena con la serie real, se corrompe la cola de la serie y se
    comprueba que las features y las predicciones del tramo anterior a la
    corrupción son idénticas.  Detecta cualquier look-ahead en la
    construcción de features, el escalado o el modelo.
    """
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from thesis.volatility_models import VolatilityLSTM, train_lstm

    raw = download_ohlcv(TICKER, start=START)
    data_a = build_volatility_dataset(raw, horizon=HORIZON)
    train_a, _, test_a = temporal_split(data_a, TRAIN_END, VAL_END)

    # Corte a mitad del período de test: a partir de aquí los datos se
    # destruyen en la serie corrupta.
    cut = test_a.index[len(test_a) // 2]

    raw_b = raw.copy()
    rng = np.random.default_rng(0)
    future = raw_b.index > cut
    n_fut = int(future.sum())
    permuted = raw_b.loc[future, "Close"].to_numpy()[rng.permutation(n_fut)]
    jitter = rng.uniform(0.9, 1.1, n_fut)
    raw_b.loc[future, "Close"] = permuted * jitter
    raw_b.loc[future, "Open"] = permuted * jitter
    raw_b.loc[future, "High"] = permuted * jitter * 1.01
    raw_b.loc[future, "Low"] = permuted * jitter * 0.99

    data_b = build_volatility_dataset(raw_b, horizon=HORIZON)
    train_b, _, test_b = temporal_split(data_b, TRAIN_END, VAL_END)

    # El train no cambia: la corrupción es posterior al train.
    pd.testing.assert_frame_equal(train_a, train_b)

    # Las features hasta el corte son idénticas.
    cols = [c for c in data_a.columns if c != "target_log_vol"]
    np.testing.assert_allclose(
        data_a.loc[:cut, cols].to_numpy(),
        data_b.loc[:cut, cols].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # El escalador se ajusta solo con el train (idéntico en ambas series).
    scaled = scale_splits(
        train_a[cols].values,
        train_a[cols].values,
        train_a[cols].values,
        train_a["target_log_vol"].values,
        train_a["target_log_vol"].values,
        train_a["target_log_vol"].values,
    )
    Xtr, ytr = make_sequences(scaled["X_train"], scaled["y_train"], LOOKBACK)

    model = VolatilityLSTM(n_features=Xtr.shape[-1], hidden=8, dropout=0.0)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=64,
        shuffle=False,
        drop_last=True,
    )
    config = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 2,
        "patience": 1,
        "min_epochs": 1,
        "grad_clip": 1.0,
        "linex_b": 0.5,
        "model_path": "leak_probe.pt",
    }
    try:
        train_lstm(
            model,
            loader,
            torch.from_numpy(Xtr[:8]),
            torch.from_numpy(ytr[:8]),
            config,
            torch.device("cpu"),
        )
    finally:
        Path(config["model_path"]).unlink(missing_ok=True)

    model.eval()
    with torch.no_grad():
        preds = []
        for frame in (test_a, test_b):
            seq_X, _ = make_sequences(
                scaled["X_scaler"].transform(frame[cols].values),
                frame["target_log_vol"].values,
                LOOKBACK,
            )
            n_prefix = int((frame.index[LOOKBACK:] <= cut).sum())
            preds.append(model(torch.from_numpy(seq_X[:n_prefix])).numpy())

    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-12, atol=1e-12)


def test_notebook_import_does_not_run_pipeline() -> None:
    """Importar el notebook no debe descargar datos ni entrenar."""
    notebook = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "volatility"
        / "Volatilidad_LSTM.py"
    )
    spec = importlib.util.spec_from_file_location("volatility_notebook", notebook)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["volatility_notebook"] = module
    spec.loader.exec_module(module)
    assert callable(module.run_experiment)
    assert "features" in module.CONFIG


def test_descriptive_notebook_import_does_not_download() -> None:
    """Importar el cuaderno descriptivo no debe descargar datos ni graficar."""
    notebook = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "volatility"
        / "Volatilidad_LSTM_Descripcion.py"
    )
    spec = importlib.util.spec_from_file_location("volatility_desc_notebook", notebook)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["volatility_desc_notebook"] = module
    spec.loader.exec_module(module)
    assert module.CONFIG["ticker"] == "SPY"
    assert module.CONFIG["train_end"] < module.CONFIG["val_end"]


def test_notebook_end_to_end_with_real_data(tmp_path: Path) -> None:
    """Pipeline completo del notebook con datos reales y entrenamiento corto."""
    pytest.importorskip("torch")
    notebook = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "volatility"
        / "Volatilidad_LSTM.py"
    )
    spec = importlib.util.spec_from_file_location("volatility_notebook_e2e", notebook)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["volatility_notebook_e2e"] = module
    spec.loader.exec_module(module)

    config = {
        "start_date": START,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "max_epochs": 2,
        "min_epochs": 1,
        "patience": 1,
        "batch_size": 64,
        "lr": 1e-3,
        "model_path": str(tmp_path / "model.pt"),
        "scaler_path": str(tmp_path / "scaler.pkl"),
        "target_scaler_path": str(tmp_path / "target_scaler.pkl"),
        "metrics_path": str(tmp_path / "metrics.csv"),
        "config_path": str(tmp_path / "config.json"),
    }

    import matplotlib

    matplotlib.use("Agg")
    results = module.run_experiment(config)

    assert results["metrics"]["lstm"]["N"] > 100
    assert np.isfinite(results["metrics"]["lstm"]["RMSE"])
    assert len(results["history"]["train_loss"]) == 2
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "scaler.pkl").exists()
    assert (tmp_path / "target_scaler.pkl").exists()

    metrics_csv = pd.read_csv(tmp_path / "metrics.csv")
    assert list(metrics_csv["model"]) == ["LSTM", "HAR-RV (Baseline)"]

    # --- Alineación del backtest (forecast ↔ retorno operado) ---
    smoothed = module.smooth_predictions(config)
    n_eval = len(smoothed["y_lstm"])
    assert len(smoothed["dates"]) == n_eval
    assert len(smoothed["ret_te_aligned"]) == n_eval - 1
    assert len(smoothed["bt_dates"]) == n_eval - 1

    # ``bt_dates[j]`` es el día en que se realiza el retorno operado por el
    # forecast ``j``.
    log_ret = np.log(results["raw"]["Close"] / results["raw"]["Close"].shift(1))
    expected_ret = log_ret.reindex(smoothed["bt_dates"]).fillna(0.0).to_numpy()
    np.testing.assert_allclose(
        smoothed["ret_te_aligned"], expected_ret, rtol=1e-12, atol=1e-12
    )

    # --- Métricas primarias raw-vs-raw (la verdad no recibe EWMA) ---
    from sklearn.metrics import r2_score

    np.testing.assert_allclose(smoothed["y_true"], smoothed["y_true_raw"])
    assert results["metrics"]["lstm"]["R2"] == pytest.approx(
        r2_score(smoothed["y_true"], smoothed["y_lstm_raw"])
    )
