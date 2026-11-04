"""Tests para ``thesis.volatility``.

Usan datos sintéticos, no requieren red ni descarga de datos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.timeseries import log_returns
from thesis.volatility import (
    align_har_features_with_sequences,
    build_volatility_dataset,
    compute_log_vol_target,
    compute_volatility_features,
    count_extreme_moves,
    download_ohlcv,
    make_sequences,
    purge_boundary_labels,
    scale_splits,
    temporal_split,
    validate_ohlcv,
)


def _make_ohlcv(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Precio que crece con ruido diario del 1%.
    returns = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.015, n))
    low = close / (1 + rng.uniform(0.001, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_validate_ohlcv_ok() -> None:
    df = _make_ohlcv(50)
    cleaned = validate_ohlcv(df)
    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert cleaned.index.is_monotonic_increasing
    assert (cleaned[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_validate_ohlcv_reconstruye_consistencia() -> None:
    """``High``/``Low`` se ajustan para contener ``Open``/``Close``."""
    df = _make_ohlcv(20)
    df.loc[df.index[5], "Close"] = df.loc[df.index[5], "High"] + 10.0
    df.loc[df.index[7], "Open"] = df.loc[df.index[7], "Low"] - 5.0

    cleaned = validate_ohlcv(df)
    assert (cleaned["High"] >= cleaned[["Open", "Close"]].max(axis=1)).all()
    assert (cleaned["Low"] <= cleaned[["Open", "Close"]].min(axis=1)).all()
    assert (cleaned["High"] >= cleaned["Low"]).all()


def test_validate_ohlcv_rejects_negative_price() -> None:
    df = _make_ohlcv(50)
    df.loc[df.index[10], "Close"] = -1.0
    with pytest.raises(ValueError, match="Precio no positivo"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_missing_columns() -> None:
    df = _make_ohlcv(50).drop(columns=["Volume"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_ohlcv(df)


def test_compute_volatility_features_is_causal() -> None:
    df = _make_ohlcv(80)
    # Las features dependen del pasado; ningún valor en t debe depender de
    # precios futuros.  Si truncamos el precio, las features conocidas en t
    # deben coincidir con las calculadas sobre la serie completa.
    full_feat = compute_volatility_features(df)
    for t in [40, 50, 60]:
        truncated = df.iloc[: t + 1]
        partial_feat = compute_volatility_features(truncated)
        # Comparamos fila a fila para características que ya tienen datos suficientes.
        common_cols = [
            c for c in full_feat.columns if not pd.isna(full_feat.iloc[t][c])
        ]
        for col in common_cols:
            assert np.isclose(full_feat.iloc[t][col], partial_feat.iloc[-1][col])


def test_compute_log_vol_target_uses_future() -> None:
    df = _make_ohlcv(50)
    target = compute_log_vol_target(df, horizon=5)
    # Las últimas ``horizon`` observaciones son NaN.
    assert target.iloc[-5:].isna().all()
    assert target.iloc[:-5].notna().all()


def test_build_volatility_dataset_dropna() -> None:
    df = _make_ohlcv(50)
    dataset = build_volatility_dataset(df, horizon=5)
    assert not dataset.isna().any().any()
    assert "target_log_vol" in dataset.columns
    assert "rv_daily" in dataset.columns


def test_temporal_split_disjoint() -> None:
    dates = pd.date_range("2000-01-01", periods=100, freq="B")
    data = pd.DataFrame({"x": np.arange(100)}, index=dates)
    train, val, test = temporal_split(data, "2000-03-15", "2000-05-15")
    assert train.index.max() <= pd.Timestamp("2000-03-15")
    assert val.index.min() > pd.Timestamp("2000-03-15")
    assert val.index.max() <= pd.Timestamp("2000-05-15")
    assert test.index.min() > pd.Timestamp("2000-05-15")
    assert len(train) + len(val) + len(test) == len(data)


def test_purge_boundary_labels_removes_last_rows() -> None:
    """La purga elimina las últimas ``horizon`` filas de la partición."""
    data = build_volatility_dataset(_make_ohlcv(120), horizon=5)
    train, _, _ = temporal_split(data, "2020-03-15", "2020-04-15")

    purged = purge_boundary_labels(train, 5)

    assert len(purged) == len(train) - 5
    assert purged.index.equals(train.index[:-5])
    # No se toca el DataFrame original.
    assert len(train) == len(purged) + 5


def test_purge_boundary_labels_labels_independent_of_next_split() -> None:
    """Los labels del train purgado no dependen de los retornos de validación.

    Se corrompe la cola de la serie (validación y test) y se comprueba que los
    targets del train purgado son idénticos; sin purga, las últimas ``horizon``
    filas del train sí cambian.
    """
    horizon = 5
    df = _make_ohlcv(200)
    data = build_volatility_dataset(df, horizon=horizon)
    train, _, _ = temporal_split(data, "2020-03-15", "2020-04-15")

    # Serie corrupta: se multiplica el precio de toda la cola posterior al
    # train, lo que altera los retornos que usan sus labels de frontera.
    df_corrupta = df.copy()
    futura = df_corrupta.index > train.index[-1]
    df_corrupta.loc[futura, ["Open", "High", "Low", "Close"]] *= 3.0
    data_corrupta = build_volatility_dataset(df_corrupta, horizon=horizon)
    train_corrupto, _, _ = temporal_split(data_corrupta, "2020-03-15", "2020-04-15")

    purgado = purge_boundary_labels(train, horizon)
    purgado_corrupto = purge_boundary_labels(train_corrupto, horizon)

    assert len(purgado) == len(purgado_corrupto)
    np.testing.assert_allclose(
        purgado["target_log_vol"].to_numpy(),
        purgado_corrupto["target_log_vol"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # Sin purga, los últimos labels del train sí dependen de la validación.
    assert not np.allclose(
        train["target_log_vol"].to_numpy()[-horizon:],
        train_corrupto["target_log_vol"].to_numpy()[-horizon:],
    )


def test_make_sequences_shape() -> None:
    X = np.arange(60).reshape(-1, 2).astype(np.float32)
    y = np.arange(30).astype(np.float32)
    lookback = 5
    Xs, ys = make_sequences(X, y, lookback)
    assert Xs.shape == (len(y) - lookback, lookback, 2)
    assert ys.shape == (len(y) - lookback,)
    # El target de la primera secuencia debe ser y[lookback].
    assert ys[0] == pytest.approx(y[lookback])
    # La última fila de la primera secuencia debe ser X[lookback-1].
    assert np.allclose(Xs[0, -1], X[lookback - 1])


def test_align_har_features_with_sequences() -> None:
    features = pd.DataFrame({"a": np.arange(20), "b": np.arange(20) * 2})
    y_seq = np.zeros(15)
    aligned = align_har_features_with_sequences(features, y_seq, lookback=5)
    assert len(aligned) == len(y_seq)
    assert aligned.iloc[0]["a"] == 4  # lookback - 1
    assert aligned.iloc[-1]["a"] == 18  # start + len(y_seq) - 1


def test_scale_splits_zero_mean_unit_variance() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(10, 2, (50, 3)).astype(np.float32)
    X_val = rng.normal(10, 2, (20, 3)).astype(np.float32)
    X_test = rng.normal(10, 2, (20, 3)).astype(np.float32)
    y_train = rng.normal(5, 1, 50).astype(np.float32)
    y_val = rng.normal(5, 1, 20).astype(np.float32)
    y_test = rng.normal(5, 1, 20).astype(np.float32)

    scaled = scale_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    np.testing.assert_allclose(scaled["X_train"].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(scaled["X_train"].std(axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(scaled["y_train"].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(scaled["y_train"].std(), 1, atol=1e-6)


def test_count_extreme_moves() -> None:
    df = _make_ohlcv(100)
    log_ret = log_returns(df["Close"])
    threshold = 0.05
    expected = int((np.abs(log_ret) > threshold).sum())
    assert count_extreme_moves(df, threshold=threshold) == expected


def test_download_ohlcv_mock(monkeypatch) -> None:
    """``download_ohlcv`` delega en ``yfinance``; validamos el wrapper."""
    df = _make_ohlcv(30)

    def fake_download(*_, **__):
        return df.copy()

    monkeypatch.setattr("yfinance.download", fake_download)
    result = download_ohlcv("FAKE", start="2020-01-01")
    pd.testing.assert_frame_equal(result, validate_ohlcv(df))
