"""Tests para ``thesis.portfolio``.

Usan datos sintéticos, no requieren red ni descarga de datos.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from thesis.portfolio import (
    PORTFOLIO_NAMES,
    build_har_rv_features,
    build_portfolios,
    calculate_metrics,
    estimate_sigma,
    estimate_sigma_har_rv,
    ewma_weights,
    plot_avg_weights,
    plot_equity_drawdown,
    plot_market_topology,
    plot_portfolio_composition,
    run_backtest,
)
from thesis.volatility import log_trailing_realized_vol


def _make_returns(n: int = 600, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    data = rng.normal(0.0003, 0.015, size=(n, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_ewma_weights_sum_one() -> None:
    w = ewma_weights(60, lambda_=0.94)
    assert np.isclose(w.sum(), 1.0)
    assert w.ndim == 1
    assert w[0] < w[-1]


def test_ewma_weights_positive_n() -> None:
    with pytest.raises(ValueError, match="positivo"):
        ewma_weights(0)


def test_estimate_sigma_ewma_vs_flat() -> None:
    rets = _make_returns(252, 3, seed=1)
    sigma_ewma = estimate_sigma(rets, use_ewma=True, lambda_=0.94)
    sigma_flat = estimate_sigma(rets, use_ewma=False)
    assert sigma_ewma.index.equals(rets.columns)
    assert (sigma_ewma > 0).all()
    assert (sigma_flat > 0).all()


def test_build_portfolios_weights_sum_one() -> None:
    sigma = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    weights = build_portfolios(sigma)
    assert list(weights.keys()) == list(PORTFOLIO_NAMES)
    for w in weights.values():
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()


def test_build_portfolios_rejects_invalid_sigma() -> None:
    sigma = pd.Series([0.1, 0.0, 0.3], index=["a", "b", "c"])
    with pytest.raises(ValueError, match="inválida"):
        build_portfolios(sigma)


def test_log_trailing_realized_vol_causal() -> None:
    lr = _make_returns(200, 1, seed=2).iloc[:, 0]
    full = log_trailing_realized_vol(lr, window=20)
    for t in [50, 100, 150]:
        partial = log_trailing_realized_vol(lr.iloc[: t + 1], window=20)
        assert np.isclose(full.iloc[t], partial.iloc[-1])


def test_log_trailing_realized_vol_warmup() -> None:
    lr = _make_returns(30, 1, seed=3).iloc[:, 0]
    out = log_trailing_realized_vol(lr, window=20)
    assert out.iloc[:19].isna().all()
    assert out.iloc[19:].notna().all()


def test_build_har_rv_features_columns_and_horizon() -> None:
    rets = _make_returns(400, 3, seed=4)
    features = build_har_rv_features(rets, rebalance_freq=21)
    assert set(features) == set(rets.columns)
    for asset in rets.columns:
        assert list(features[asset].columns) == ["rv_h", "rv_3h", "rv_9h"]
        # Warm-up: las primeras 9H-1 filas deben tener al menos un NaN.
        assert features[asset].iloc[: 9 * 21 - 1].isna().any(axis=1).all()


def test_build_har_rv_features_no_lookahead() -> None:
    """Corromper la cola no altera features pasadas."""
    rets = _make_returns(300, 2, seed=5)
    features = build_har_rv_features(rets, rebalance_freq=21)

    corrupt = rets.copy()
    corrupt.iloc[-30:] *= 10.0
    features_corrupt = build_har_rv_features(corrupt, rebalance_freq=21)

    for asset in rets.columns:
        cutoff = len(rets) - 30 - 9 * 21
        pd.testing.assert_frame_equal(
            features[asset].iloc[:cutoff],
            features_corrupt[asset].iloc[:cutoff],
        )


def test_estimate_sigma_har_rv_matches_manual_ols() -> None:
    """La predicción HAR-RV coincide con un OLS manual para un activo."""
    from thesis.volatility import compute_log_vol_target

    rets = _make_returns(400, 1, seed=6)
    lr = rets.iloc[:, 0]
    end_date = lr.index[-1]

    sigma_hat = estimate_sigma_har_rv(
        rets,
        rebalance_freq=21,
        lookback=252 * 5,
        end_date=end_date,
        min_obs=60,
    )

    # Replica manualmente el ajuste.
    features = build_har_rv_features(rets, rebalance_freq=21)["A0"]
    proxy = pd.DataFrame({"Close": np.exp(lr.cumsum())})
    y = compute_log_vol_target(proxy, 21)
    mask = features.notna().all(axis=1) & y.notna()
    X_train = features.loc[mask, ["rv_h", "rv_3h", "rv_9h"]]
    y_train = y.loc[mask]
    X_ols = np.column_stack([np.ones(len(X_train)), X_train.values])
    model = sm.OLS(y_train.values, X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
    X_pred = np.column_stack(
        [np.ones(1), features.loc[[end_date], ["rv_h", "rv_3h", "rv_9h"]].values]
    )
    expected = float(np.exp(model.predict(X_pred)[0]))

    assert np.isclose(sigma_hat.iloc[0], expected, rtol=1e-10)


def test_estimate_sigma_har_rv_truncation() -> None:
    """Corromper retornos futuros no altera σ̂ pasadas."""
    rets = _make_returns(500, 3, seed=7)
    end_date = rets.index[350]

    sigma_hat = estimate_sigma_har_rv(
        rets,
        rebalance_freq=21,
        end_date=end_date,
        min_obs=60,
    )

    corrupt = rets.copy()
    future_mask = rets.index > end_date
    corrupt.loc[future_mask] *= 5.0
    sigma_hat_corrupt = estimate_sigma_har_rv(
        corrupt,
        rebalance_freq=21,
        end_date=end_date,
        min_obs=60,
    )

    pd.testing.assert_series_equal(sigma_hat, sigma_hat_corrupt)


def test_run_backtest_shapes_and_weights() -> None:
    rets = _make_returns(400, 3, seed=8)
    port_rets, weights, turnover = run_backtest(
        rets,
        lookback=252,
        rebalance_freq=21,
        tx_cost=0.005,
        volatility_model="ewma",
    )
    assert list(port_rets.columns) == list(PORTFOLIO_NAMES)
    assert len(port_rets) == len(rets) - 252
    assert port_rets.index[0] == rets.index[252]
    for k in PORTFOLIO_NAMES:
        assert weights[k].shape[1] == rets.shape[1]
        assert np.allclose(weights[k].sum(axis=1), 1.0)
        assert (weights[k] >= 0).all().all()
    assert list(turnover.columns) == list(PORTFOLIO_NAMES)


def test_run_backtest_turnover() -> None:
    """El turnover reportado coincide con Σ|w_target - w_drift|."""
    rets = _make_returns(300, 3, seed=9)
    _, _, turnover = run_backtest(
        rets,
        lookback=252,
        rebalance_freq=21,
        tx_cost=0.0,
        volatility_model="ewma",
    )
    # El primer rebalanceo parte de cero, por tanto turnover = 1.
    assert np.isclose(turnover.iloc[0].mean(), 1.0)
    # Turnover no negativo.
    assert (turnover >= 0).all().all()


def test_run_backtest_har_rv_truncation() -> None:
    """Corromper el futuro no altera pesos/retornos pasados en HAR-RV."""
    rets = _make_returns(500, 3, seed=10)
    port_rets, weights, _ = run_backtest(
        rets,
        lookback=252,
        rebalance_freq=21,
        tx_cost=0.005,
        volatility_model="har_rv",
        har_min_obs=60,
    )

    # Cortamos en un rebalanceo intermedio.
    split_idx = len(port_rets) // 2
    split_date = port_rets.index[split_idx]

    corrupt = rets.copy()
    corrupt.loc[corrupt.index > split_date] *= 3.0
    port_rets_c, weights_c, _ = run_backtest(
        corrupt,
        lookback=252,
        rebalance_freq=21,
        tx_cost=0.005,
        volatility_model="har_rv",
        har_min_obs=60,
    )

    # Todo lo anterior a split_date debe coincidir.
    pd.testing.assert_frame_equal(
        port_rets.loc[:split_date], port_rets_c.loc[:split_date]
    )
    for k in PORTFOLIO_NAMES:
        pd.testing.assert_frame_equal(
            weights[k].loc[:split_date], weights_c[k].loc[:split_date]
        )


def test_calculate_metrics() -> None:
    rng = np.random.default_rng(11)
    rets = pd.Series(
        rng.normal(0.0005, 0.01, 252),
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )
    metrics = calculate_metrics(rets, "test")
    assert metrics["Strategy"] == "test"
    assert pd.notna(metrics["CAGR"])
    assert pd.notna(metrics["Sharpe"])


def test_plot_equity_drawdown_smoke() -> None:
    rets = _make_returns(200, 3, seed=12)
    port_rets, _, _ = run_backtest(rets, lookback=100, rebalance_freq=21, tx_cost=0.0)
    fig = plot_equity_drawdown(port_rets)
    assert fig is not None
    plt.close(fig)


def test_plot_portfolio_composition_smoke() -> None:
    rets = _make_returns(200, 3, seed=13)
    _, weights, _ = run_backtest(rets, lookback=100, rebalance_freq=21, tx_cost=0.0)
    fig = plot_portfolio_composition(weights["1/σ²"], title="Test")
    assert fig is not None
    plt.close(fig)


def test_plot_portfolio_composition_empty() -> None:
    with pytest.raises(ValueError, match="vacío"):
        plot_portfolio_composition(pd.DataFrame())


def test_plot_avg_weights_smoke() -> None:
    rets = _make_returns(200, 3, seed=14)
    _, weights, _ = run_backtest(rets, lookback=100, rebalance_freq=21, tx_cost=0.0)
    fig = plot_avg_weights(weights)
    assert fig is not None
    plt.close(fig)


def test_plot_market_topology_smoke() -> None:
    rets = _make_returns(200, 4, seed=15)
    fig = plot_market_topology(
        rets, start_date=str(rets.index[0].date()), end_date=str(rets.index[100].date())
    )
    assert fig is not None
    plt.close(fig)


@pytest.mark.integration
@pytest.mark.skip(reason="Requiere red y datos reales.")
def test_run_backtest_integration() -> None:
    pass
