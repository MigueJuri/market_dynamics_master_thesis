"""Tests para ``thesis.volatility_intraday``.

Usan datos sintéticos, no requieren red ni credenciales de Alpaca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.volatility import compute_volatility_features
from thesis.volatility_intraday import (
    aggregate_intraday_to_daily,
    build_volatility_dataset_from_intraday,
    compute_log_vol_target_from_intraday_daily,
    compute_volatility_features_from_intraday_daily,
    download_ohlcv_alpaca,
)


def _make_intraday(n_bars: int = 26 * 40, seed: int = 0) -> pd.DataFrame:
    """Serie OHLCV sintética de barras de 15 minutos."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.002, n_bars)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.0005, 0.004, n_bars))
    low = close / (1 + rng.uniform(0.0005, 0.004, n_bars))
    open_ = close * (1 + rng.normal(0, 0.001, n_bars))
    volume = rng.integers(100_000, 1_000_000, n_bars)
    index = pd.date_range("2020-01-01 09:30", periods=n_bars, freq="15min")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def test_aggregate_intraday_to_daily_ohlcv() -> None:
    """La agregación diaria respeta OHLCV por fecha."""
    df = _make_intraday(26 * 30)
    daily = aggregate_intraday_to_daily(df)

    assert list(daily.columns) == ["Open", "High", "Low", "Close", "Volume", "rv_intraday"]
    assert len(daily) == df.index.normalize().nunique()

    for day, group in df.groupby(pd.Grouper(freq="D")):
        if day not in daily.index:
            continue
        row = daily.loc[day]
        assert row["Open"] == group["Open"].iloc[0]
        assert row["High"] == group["High"].max()
        assert row["Low"] == group["Low"].min()
        assert row["Close"] == group["Close"].iloc[-1]
        assert row["Volume"] == group["Volume"].sum()


def test_aggregate_intraday_to_daily_excludes_overnight() -> None:
    """``rv_intraday`` no incluye el salto overnight de la primera barra."""
    # Dos días con un único retorno dentro de cada sesión para facilitar
    # el cálculo manual.
    idx = pd.to_datetime(
        ["2020-01-01 09:30", "2020-01-01 09:45", "2020-01-02 09:30", "2020-01-02 09:45"]
    )
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 110.0, 111.0],
            "High": [101.0, 102.0, 111.0, 112.0],
            "Low": [99.0, 100.0, 109.0, 110.0],
            "Close": [101.0, 102.0, 111.0, 112.0],
            "Volume": [100, 200, 300, 400],
        },
        index=idx,
    )
    daily = aggregate_intraday_to_daily(df)

    r1 = np.log(102.0 / 101.0)
    r2 = np.log(112.0 / 111.0)
    assert daily.loc["2020-01-01", "rv_intraday"] == pytest.approx(r1**2, rel=1e-12)
    assert daily.loc["2020-01-02", "rv_intraday"] == pytest.approx(r2**2, rel=1e-12)


def test_features_from_intraday_daily_same_columns_as_daily() -> None:
    """El juego de features es el mismo que el diario."""
    df = _make_intraday(26 * 30)
    daily = aggregate_intraday_to_daily(df)
    daily_features = compute_volatility_features_from_intraday_daily(daily)
    plain_features = compute_volatility_features(daily[["Open", "High", "Low", "Close", "Volume"]])
    assert list(daily_features.columns) == list(plain_features.columns)


def test_features_from_intraday_daily_use_intraday_rv() -> None:
    """Los componentes HAR usan ``rv_intraday`` (comprimido en log)."""
    df = _make_intraday(26 * 30)
    daily = aggregate_intraday_to_daily(df)
    feat = compute_volatility_features_from_intraday_daily(daily)

    np.testing.assert_allclose(
        np.exp(feat["rv_daily"].dropna()),
        daily["rv_intraday"].reindex(feat["rv_daily"].dropna().index),
        rtol=1e-10,
    )


def test_features_from_intraday_daily_are_causal() -> None:
    """Las features de un día no dependen de barras de días posteriores."""
    df = _make_intraday(26 * 100)
    daily = aggregate_intraday_to_daily(df)
    full = compute_volatility_features_from_intraday_daily(daily)

    for date in full.index[[10, 15, 20]]:
        end_of_day = date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        partial_df = df[df.index <= end_of_day]
        partial_daily = aggregate_intraday_to_daily(partial_df)
        partial = compute_volatility_features_from_intraday_daily(partial_daily)
        for col in full.columns:
            value_full = full.loc[date, col]
            if pd.isna(value_full):
                continue
            assert partial.loc[date, col] == pytest.approx(value_full, rel=1e-12)


def test_log_vol_target_from_intraday_daily_forward_and_nan_tail() -> None:
    """El target es forward y deja ``horizon`` NaN al final."""
    df = _make_intraday(26 * 30)
    daily = aggregate_intraday_to_daily(df)
    horizon = 5
    target = compute_log_vol_target_from_intraday_daily(daily, horizon=horizon)

    assert target.name == "target_log_vol"
    assert target.iloc[-horizon:].isna().all()
    assert target.iloc[:-horizon].notna().all()


def test_log_vol_target_from_intraday_daily_formula() -> None:
    """El target anualiza correctamente la suma forward de ``rv_intraday``."""
    df = _make_intraday(26 * 30)
    daily = aggregate_intraday_to_daily(df)
    horizon = 5
    target = compute_log_vol_target_from_intraday_daily(daily, horizon=horizon)

    expected = np.sqrt(
        (252 / horizon)
        * daily["rv_intraday"].rolling(horizon).sum().shift(-horizon)
    )
    expected = pd.Series(np.log(expected), index=daily.index)
    np.testing.assert_allclose(
        target.to_numpy()[:-horizon],
        expected.to_numpy()[:-horizon],
        rtol=1e-10,
    )


def test_build_volatility_dataset_from_intraday_dropna() -> None:
    """El dataset completo se construye sin NaN."""
    df = _make_intraday(26 * 40)
    dataset = build_volatility_dataset_from_intraday(df, horizon=5)
    assert not dataset.isna().any().any()
    assert "target_log_vol" in dataset.columns
    assert "rv_daily" in dataset.columns


def test_download_ohlcv_alpaca_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credenciales"):
        download_ohlcv_alpaca("SPY", start="2020-01-01")


def test_download_ohlcv_alpaca_normalizes_bars(monkeypatch) -> None:
    """El wrapper renombra columnas, quita el nivel del símbolo y pasa a ET."""
    pytest.importorskip("alpaca")
    import alpaca.data.historical as hist

    index = pd.MultiIndex.from_product(
        [["SPY"], pd.date_range("2020-01-01 14:30", periods=3, freq="15min", tz="UTC")],
        names=["symbol", "timestamp"],
    )
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 20.0, 30.0],
            "trade_count": [1.0, 2.0, 3.0],
            "vwap": [100.2, 101.2, 102.2],
        },
        index=index,
    )

    class FakeClient:
        def __init__(self, api_key: str, secret_key: str) -> None:
            self.api_key = api_key

        def get_stock_bars(self, request):
            class Result:
                df = bars

            return Result()

    monkeypatch.setattr(hist, "StockHistoricalDataClient", FakeClient)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    df = download_ohlcv_alpaca("SPY", start="2020-01-01")

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.tz is None
    # 14:30 UTC son las 09:30 ET (invierno).
    assert df.index[0] == pd.Timestamp("2020-01-01 09:30")
    assert len(df) == 3


def test_download_ohlcv_alpaca_caches_to_disk(monkeypatch, tmp_path) -> None:
    """La segunda llamada con el mismo cache no vuelve a descargar."""
    pytest.importorskip("alpaca")
    import alpaca.data.historical as hist

    index = pd.MultiIndex.from_product(
        [["SPY"], pd.date_range("2020-01-01 14:30", periods=2, freq="15min", tz="UTC")],
        names=["symbol", "timestamp"],
    )
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 20.0],
        },
        index=index,
    )

    calls = []

    class FakeClient:
        def __init__(self, api_key: str, secret_key: str) -> None:
            pass

        def get_stock_bars(self, request):
            calls.append(1)

            class Result:
                df = bars

            return Result()

    monkeypatch.setattr(hist, "StockHistoricalDataClient", FakeClient)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")

    df1 = download_ohlcv_alpaca("SPY", start="2020-01-01", cache_dir=tmp_path)
    assert len(calls) == 1
    assert len(df1) == 2

    df2 = download_ohlcv_alpaca("SPY", start="2020-01-01", cache_dir=tmp_path)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(df1, df2)
