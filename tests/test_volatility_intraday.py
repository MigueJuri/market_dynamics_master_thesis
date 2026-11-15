"""Tests para ``thesis.volatility_intraday``.

Usan datos sintéticos, no requieren red ni credenciales de Alpaca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.volatility_intraday import (
    build_volatility_dataset_intraday,
    compute_log_vol_target_intraday,
    compute_volatility_features_intraday,
    download_ohlcv_alpaca,
    infer_frequency_params,
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


def test_features_have_same_columns_as_daily() -> None:
    """El juego de features intradía es el mismo que el diario."""
    from thesis.volatility import compute_volatility_features

    daily = compute_volatility_features(_make_intraday(26 * 30))
    intra = compute_volatility_features_intraday(_make_intraday(26 * 30))
    assert list(intra.columns) == list(daily.columns)


def test_features_are_causal() -> None:
    """Las features en ``t`` no dependen de los precios futuros."""
    df = _make_intraday(26 * 30)
    full = compute_volatility_features_intraday(df)
    for t in (500, 600, 700):
        partial = compute_volatility_features_intraday(df.iloc[: t + 1])
        common = [c for c in full.columns if not pd.isna(full.iloc[t][c])]
        for col in common:
            assert np.isclose(full.iloc[t][col], partial.iloc[-1][col])


def test_rv_daily_sums_the_bars_of_a_day() -> None:
    """``rv_daily`` es la suma de ``r²`` de las 26 barras del día."""
    df = _make_intraday(26 * 5)
    feat = compute_volatility_features_intraday(df)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    esperado = (log_ret**2).rolling(26, min_periods=26).sum()
    # La columna comprimida es el log: se compara en ese espacio.
    np.testing.assert_allclose(
        feat["rv_daily"].to_numpy()[26:],
        np.log(esperado.to_numpy()[26:]),
        rtol=1e-10,
    )


def test_target_is_forward_and_has_nan_tail() -> None:
    df = _make_intraday(26 * 20)
    horizon = 130
    target = compute_log_vol_target_intraday(df, horizon_bars=horizon)
    assert target.iloc[-horizon:].isna().all()
    assert target.iloc[:-horizon].notna().all()


def test_target_depends_only_on_future_returns() -> None:
    """El target en ``t`` usa los retornos de ``t+1 ... t+H`` (sin look-ahead)."""
    df = _make_intraday(26 * 20)
    horizon = 130
    target = compute_log_vol_target_intraday(df, horizon_bars=horizon)

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    esperado = np.sqrt(
        (6552 / horizon) * (log_ret**2).rolling(horizon).sum().shift(-horizon)
    )
    esperado = pd.Series(np.log(esperado.to_numpy()), index=df.index)
    # El target descarta la primera fila (``log_returns`` elimina el primer NaN).
    esperado = esperado.iloc[1:]
    np.testing.assert_allclose(
        target.to_numpy()[:-horizon],
        esperado.to_numpy()[:-horizon],
        rtol=1e-10,
    )


def test_build_dataset_dropna() -> None:
    df = _make_intraday(26 * 30)
    dataset = build_volatility_dataset_intraday(df, horizon_bars=130)
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


@pytest.mark.parametrize(
    "timeframe,expected_bpd,expected_bpy",
    [
        ("1Min", 390, 98280),
        ("5Min", 78, 19656),
        ("15Min", 26, 6552),
        ("30Min", 13, 3276),
    ],
)
def test_infer_frequency_params(timeframe, expected_bpd, expected_bpy) -> None:
    """La inferencia de frecuencia escala correctamente con el timeframe."""
    freq = infer_frequency_params(timeframe)
    assert freq["bars_per_day"] == expected_bpd
    assert freq["bars_per_year"] == expected_bpy
    assert freq["minutes_per_bar"] * freq["bars_per_day"] == pytest.approx(390)


def test_infer_frequency_params_rejects_unsupported_timeframe() -> None:
    with pytest.raises(ValueError, match="Timeframe no soportado"):
        infer_frequency_params("1Day")


def _make_intraday_freq(n_days: int = 5, freq: str = "15min", seed: int = 0) -> pd.DataFrame:
    """Serie OHLCV sintética para una frecuencia arbitraria."""
    bars_per_day = {"1min": 390, "5min": 78, "15min": 26}[freq]
    n_bars = bars_per_day * n_days
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.002, n_bars)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.0005, 0.004, n_bars))
    low = close / (1 + rng.uniform(0.0005, 0.004, n_bars))
    open_ = close * (1 + rng.normal(0, 0.001, n_bars))
    volume = rng.integers(100_000, 1_000_000, n_bars)
    index = pd.date_range("2020-01-01 09:30", periods=n_bars, freq=freq)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


@pytest.mark.parametrize("freq,bpd", [("1min", 390), ("5min", 78), ("15min", 26)])
def test_rv_daily_sums_the_bars_of_a_day_frequency_agnostic(freq, bpd) -> None:
    """``rv_daily`` suma exactamente las barras de una sesión para 1/5/15 min."""
    df = _make_intraday_freq(n_days=5, freq=freq)
    feat = compute_volatility_features_intraday(df, bars_per_day=bpd)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    esperado = (log_ret**2).rolling(bpd, min_periods=bpd).sum()
    np.testing.assert_allclose(
        feat["rv_daily"].to_numpy()[bpd:],
        np.log(esperado.to_numpy()[bpd:]),
        rtol=1e-10,
    )


@pytest.mark.parametrize("freq,bpd", [("1min", 390), ("5min", 78), ("15min", 26)])
def test_target_annualization_scales_with_frequency(freq, bpd) -> None:
    """El target usa ``bars_per_year`` correctamente para cada frecuencia."""
    df = _make_intraday_freq(n_days=10, freq=freq)
    horizon = 5 * bpd
    bars_per_year = 252 * bpd
    target = compute_log_vol_target_intraday(
        df, horizon_bars=horizon, bars_per_year=bars_per_year
    )

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    esperado = np.sqrt(
        (bars_per_year / horizon) * (log_ret**2).rolling(horizon).sum().shift(-horizon)
    )
    esperado = pd.Series(np.log(esperado.to_numpy()), index=df.index).iloc[1:]
    np.testing.assert_allclose(
        target.to_numpy()[:-horizon],
        esperado.to_numpy()[:-horizon],
        rtol=1e-10,
    )


@pytest.mark.parametrize("freq,bpd", [("1min", 390), ("5min", 78), ("15min", 26)])
def test_build_dataset_frequency_agnostic(freq, bpd) -> None:
    """El dataset completo se construye sin NaN para 1/5/15 minutos."""
    df = _make_intraday_freq(n_days=30, freq=freq)
    dataset = build_volatility_dataset_intraday(df, horizon_bars=5 * bpd, bars_per_day=bpd)
    assert not dataset.isna().any().any()
    assert "target_log_vol" in dataset.columns
    assert "rv_daily" in dataset.columns
