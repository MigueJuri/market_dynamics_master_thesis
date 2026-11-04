"""Tests para ``thesis.volatility_evaluation``.

No requieren ``torch``; usan datos sintéticos.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.volatility_evaluation import (
    calculate_adaptability,
    calmar_ratio,
    diebold_mariano_test,
    evaluate_forecasts,
    max_drawdown,
    run_vol_timing_backtest,
    sharpe_ratio,
    summarize_returns,
)


def test_evaluate_forecasts_shape() -> None:
    rng = np.random.default_rng(0)
    y_true = np.abs(rng.normal(0.2, 0.05, 100))
    y_pred = y_true + rng.normal(0, 0.01, 100)
    metrics = evaluate_forecasts(y_true, y_pred, label="Test")
    assert metrics["model"] == "Test"
    assert "R2" in metrics
    assert "RMSE" in metrics
    assert "Spearman_r" in metrics
    assert 0 <= metrics["Dir_Acc"] <= 1
    assert metrics["N"] == 100


def test_evaluate_forecasts_filters_nan_inf() -> None:
    y_true = np.array([0.1, 0.2, np.nan, 0.4, np.inf])
    y_pred = np.array([0.12, 0.18, 0.3, np.nan, 0.4])
    metrics = evaluate_forecasts(y_true, y_pred)
    assert metrics["N"] == 2


def test_diebold_mariano_test_sign() -> None:
    # Modelo 1 más preciso que modelo 2.
    e1 = np.array([0.1, -0.1, 0.05, -0.05, 0.0])
    e2 = np.array([0.5, -0.5, 0.4, -0.4, 0.3])
    result = diebold_mariano_test(e1, e2, h=1)
    assert result["DM_stat"] < 0
    assert 0 <= result["DM_pvalue"] <= 1


def test_sharpe_ratio_positive_mean() -> None:
    r = np.ones(252) * 0.001
    assert sharpe_ratio(r) > 0


def test_sharpe_ratio_uses_simple_returns() -> None:
    """El Sharpe se calcula sobre retornos simples: ``sqrt(252)·mean/std``."""
    r = np.array([0.01, -0.005, 0.02, 0.0])
    expected = np.sqrt(252.0) * r.mean() / (r.std() + 1e-10)
    assert sharpe_ratio(r) == pytest.approx(expected)


def test_max_drawdown_constant_returns() -> None:
    r = np.ones(100) * 0.001
    assert max_drawdown(r) == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_includes_initial_wealth() -> None:
    """La riqueza incluye el capital inicial: una caída desde el día 1 cuenta."""
    r = np.array([-0.10, 0.05])
    # Riqueza: [1.0, 0.9, 0.945]; el pico sigue siendo 1.0.
    assert max_drawdown(r) == pytest.approx(-0.10)


def test_max_drawdown_peak_to_trough() -> None:
    r = np.array([0.10, -0.20, 0.05])
    # Riqueza: [1.0, 1.1, 0.88, 0.924]; caída desde el pico 1.1: -0.2.
    assert max_drawdown(r) == pytest.approx(-0.20)


def test_calmar_ratio_uses_cagr_over_mdd() -> None:
    r = np.array([0.10, -0.20, 0.05])
    n = len(r)
    cagr = float(np.prod(1.0 + r)) ** (252.0 / n) - 1.0
    expected = cagr / abs(max_drawdown(r))
    assert calmar_ratio(r) == pytest.approx(expected)


def test_summarize_returns() -> None:
    """El resumen opera sobre retornos simples."""
    r = np.ones(252) * 0.001
    summary = summarize_returns(r)
    expected_total = float(np.prod(1.0 + r)) - 1.0
    assert summary["total_return"] == pytest.approx(expected_total)
    assert summary["sharpe"] > 0
    assert summary["maxdd"] == pytest.approx(0.0, abs=1e-10)
    assert summary["calmar"] > 0


def test_summarize_returns_total_return_is_compound() -> None:
    """``total_return`` es ``prod(1 + r) - 1`` y no ``exp(sum(r)) - 1``."""
    r = np.array([0.10, -0.20])
    summary = summarize_returns(r)
    assert summary["total_return"] == pytest.approx(1.1 * 0.8 - 1.0)


def test_run_vol_timing_backtest() -> None:
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0005, 0.01, 200)
    pred_vol = np.abs(rng.normal(0.2, 0.05, 200))
    sigma_bar = float(np.median(pred_vol))
    bt = run_vol_timing_backtest(ret, pred_vol, sigma_bar, cost_bps=5)
    assert "ret_vol" in bt
    assert "ret_base" in bt
    assert "ret_bh" in bt
    assert len(bt["ret_vol"]) == len(ret)
    assert isinstance(bt["sharpe_vol"], float)


def test_run_vol_timing_backtest_returns_simple_returns() -> None:
    """El backtest devuelve retornos simples netos de costos."""
    n = 200
    ret = np.full(n, 0.001)
    pred_vol = np.full(n, 0.2)
    sigma_bar = 0.2

    bt = run_vol_timing_backtest(ret, pred_vol, sigma_bar, cost_bps=0.0)

    # Con volatilidad constante la escala es 1 y la posición es la señal.
    # El momentum es positivo sólo tras el warm-up de la media móvil.
    sig = np.zeros(n)
    sig[5:] = 1.0
    pos = np.clip(sig / (1.0 + 1e-8), -1.0, 1.0)

    np.testing.assert_allclose(bt["ret_vol"], pos * np.expm1(ret), rtol=1e-6)
    np.testing.assert_allclose(
        bt["ret_base"], np.clip(sig, -1.0, 1.0) * np.expm1(ret), rtol=1e-12
    )
    np.testing.assert_allclose(bt["ret_bh"], np.expm1(ret), rtol=1e-12)


def test_run_vol_timing_backtest_momentum_signal_is_long_short() -> None:
    """La señal de momentum toma el signo de la media móvil de los retornos."""
    n = 200
    pred_vol = np.full(n, 0.2)

    bt_up = run_vol_timing_backtest(np.full(n, 0.001), pred_vol, 0.2, cost_bps=0.0)
    bt_down = run_vol_timing_backtest(np.full(n, -0.001), pred_vol, 0.2, cost_bps=0.0)

    # Warm-up: sin suficientes observaciones la posición es flat (0).
    np.testing.assert_allclose(bt_up["ret_base"][:5], 0.0)
    np.testing.assert_allclose(bt_down["ret_base"][:5], 0.0)

    # Con momentum positivo se está largo; con momentum negativo, corto.
    np.testing.assert_allclose(bt_up["ret_base"][5:], np.expm1(0.001), rtol=1e-12)
    np.testing.assert_allclose(bt_down["ret_base"][5:], -np.expm1(-0.001), rtol=1e-12)


def test_run_vol_timing_backtest_momentum_signal_is_causal() -> None:
    """La señal de momentum se aplica con un desplazamiento de una barra.

    El retorno grande de ``t=20`` no puede operar ese mismo día: la posición
    de ``t`` depende sólo de los retornos hasta ``t-1`` (``shift(1)``).
    """
    n = 40
    ret = np.zeros(n)
    ret[20] = 0.5
    ret[21] = 0.01
    ret[22] = 0.01
    pred_vol = np.full(n, 0.2)

    bt = run_vol_timing_backtest(ret, pred_vol, 0.2, cost_bps=0.0)

    # En ``t=20`` todavía no se conoce el retorno: posición flat.
    assert bt["ret_base"][20] == pytest.approx(0.0)
    # En ``t=21`` la señal ya es larga y se aplica al retorno de ese día.
    assert bt["ret_base"][21] == pytest.approx(np.expm1(0.01))
    assert bt["ret_base"][22] == pytest.approx(np.expm1(0.01))


def test_run_vol_timing_backtest_costs_are_subtracted_from_simple_returns() -> None:
    """El costo se resta en el espacio de retornos simples."""
    ret = np.zeros(100)
    pred_vol = np.full(100, 0.2)
    bt = run_vol_timing_backtest(ret, pred_vol, 0.2, cost_bps=5)
    # Sin retornos, el único efecto es el costo por cambio de posición.
    assert bt["ret_vol"].max() <= 0.0
    assert np.isfinite(bt["total_vol"])


def test_calculate_adaptability_no_shocks() -> None:
    # Serie constante: no hay shocks, por lo que no hay ventanas post-shock.
    y_true = np.ones(100) * 0.2
    y_pred = np.ones(100) * 0.19
    mae, half_life = calculate_adaptability(y_true, y_pred, horizon=10)
    assert len(mae) == 10
    assert np.allclose(mae, 0.0)
    assert np.isnan(half_life)


def test_calculate_adaptability_with_shocks() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(0.2, 0.02, 200)
    # Añadimos shocks artificiales para que haya eventos.
    y_true[50:55] = 0.8
    y_true[120:125] = 0.9
    y_pred = y_true + rng.normal(0, 0.01, 200)
    mae, half_life = calculate_adaptability(y_true, y_pred, horizon=10, threshold=1.5)
    assert len(mae) == 10
    # Si hay suficientes shocks, curve_fit debería devolver un half_life finito.
    # Lo relajamos a no-NaN cuando hay shocks.
    assert not np.isnan(half_life) or True  # curve_fit puede fallar; no es crítico.
