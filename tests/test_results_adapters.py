"""Tests para ``thesis.results_adapters``.

Usan payloads sintéticos; no requieren red ni datos reales.
"""

from __future__ import annotations

import pandas as pd
import pytest

from thesis.results import ResultRecord
from thesis.results_adapters import ADAPTERS


def _lstm_payload() -> dict:
    metrics_df = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "R2": 0.40,
                "RMSE": 0.07,
                "MAE": 0.04,
                "Spearman_r": 0.64,
                "Spearman_p": 1e-10,
                "Dir_Acc": 0.77,
                "Dir_p": 1e-5,
                "QLIKE": 0.36,
                "MZ_alpha": 0.01,
                "MZ_beta": 0.93,
                "MZ_F_pval": 0.19,
                "N": 1212,
            },
            {
                "model": "HAR-RV (Baseline)",
                "R2": 0.26,
                "RMSE": 0.08,
                "MAE": 0.05,
                "Spearman_r": 0.55,
                "Spearman_p": 1e-8,
                "Dir_Acc": 0.74,
                "Dir_p": 1e-4,
                "QLIKE": 0.49,
                "MZ_alpha": 0.02,
                "MZ_beta": 0.97,
                "MZ_F_pval": 0.01,
                "N": 1212,
            },
        ]
    )
    return {
        "config": {
            "ticker": "SPY",
            "start_date": "1995-01-01",
            "train_end": "2017-01-01",
            "val_end": "2022-01-01",
            "end_date": "2026-01-01",
            "forecast_horizon": 5,
            "lookback": 5,
            "cost_bps": 0.0,
        },
        "metrics": {
            "LSTM": metrics_df[metrics_df.model == "LSTM"].iloc[0].to_dict(),
            "HAR-RV (Baseline)": metrics_df[metrics_df.model == "HAR-RV (Baseline)"].iloc[0].to_dict(),
        },
        "backtest": {
            "sharpe_improvement": 0.25,
            "sharpe_vs_bh": 0.10,
        },
        "summary": pd.DataFrame(
            [
                {"strategy": "Vol-Timed", "total_return": 0.5, "sharpe": 1.2, "maxdd": -0.2, "calmar": 3.0},
                {"strategy": "Baseline", "total_return": 0.3, "sharpe": 0.9, "maxdd": -0.25, "calmar": 2.0},
                {"strategy": "Buy & Hold", "total_return": 0.4, "sharpe": 1.0, "maxdd": -0.3, "calmar": 2.5},
            ]
        ),
    }


def _generic_payload() -> dict:
    return {
        "config": {"ticker": "SPY", "start_date": "1993-01-01", "end_date": "2026-01-01"},
        "metrics_df": pd.DataFrame(
            [
                {"model": "A", "R2": 0.1, "sharpe": 1.5},
                {"model": "B", "R2": 0.2, "sharpe": 1.6},
            ]
        ),
    }


def test_lstm_daily_adapter():
    adapter = ADAPTERS["volatility/lstm_daily"]
    records = adapter("volatility/lstm_daily", _lstm_payload())
    assert records
    assert all(isinstance(r, ResultRecord) for r in records)
    assert any(r.metric_name == "R²" and r.model == "LSTM" for r in records)
    assert any(r.metric_name == "Vol-Timed sharpe" for r in records)
    assert all(r.chapter == "Volatilidad" for r in records)


def test_lstm_alpaca_adapter():
    adapter = ADAPTERS["volatility/lstm_alpaca"]
    records = adapter("volatility/lstm_alpaca", _lstm_payload())
    assert records
    assert all(isinstance(r, ResultRecord) for r in records)
    assert any(r.metric_name == "R²" and r.model == "LSTM" for r in records)
    assert any(r.experiment == "LSTM Alpaca" for r in records)
    assert all(r.chapter == "Volatilidad" for r in records)


def test_lstm_daily_adapter_candidate_on_bad_payload():
    adapter = ADAPTERS["volatility/lstm_daily"]
    records = adapter("volatility/lstm_daily", "no es dict")
    assert len(records) == 1
    assert records[0].status == "candidate"
    assert "adapter pendiente" in records[0].notes


#: Los 18 experimentos canónicos de los notebooks del repositorio.
CANONICAL_EXPERIMENTS = (
    "feature_analysis/momentum",
    "feature_analysis/volatility",
    "feature_analysis/momentum_volatility",
    "feature_analysis/moving_average_distance",
    "feature_analysis/per",
    "trend_following/strategy_v2",
    "pair_trading/selection_mi",
    "pair_trading/selection_pearson",
    "pair_trading/practical",
    "pair_trading/spread",
    "pair_trading/ou",
    "btc/btc",
    "spy/characterization",
    "spy/sectors",
    "spy/orderbook",
    "volatility/lstm_daily",
    "volatility/lstm_alpaca",
    "volatility/lstm_description",
)


def test_all_canonical_experiments_have_adapter():
    missing = [exp for exp in CANONICAL_EXPERIMENTS if exp not in ADAPTERS]
    assert not missing


@pytest.mark.parametrize(
    "experiment_id",
    [
        "feature_analysis/momentum",
        "feature_analysis/volatility",
        "feature_analysis/momentum_volatility",
        "feature_analysis/moving_average_distance",
        "feature_analysis/per",
        "trend_following/strategy_v2",
        "pair_trading/selection_mi",
        "pair_trading/selection_pearson",
        "pair_trading/practical",
        "pair_trading/spread",
        "pair_trading/ou",
        "btc/btc",
        "spy/characterization",
        "spy/sectors",
        "spy/orderbook",
        "volatility/lstm_description",
    ],
)
def test_generic_adapters_accept_payload(experiment_id: str):
    adapter = ADAPTERS[experiment_id]
    records = adapter(experiment_id, _generic_payload())
    assert records
    assert all(isinstance(r, ResultRecord) for r in records)
    assert all(r.status == "candidate" for r in records)
    assert all(r.metric_value != "adapter_pending" for r in records)


def test_generic_adapter_skips_nan_metrics():
    """Las métricas que no aplican a una fila (``nan``) no se publican."""
    payload = {
        "config": {"ticker": "SPY", "start_date": "1993-01-01", "end_date": "2026-01-01"},
        "metrics_df": pd.DataFrame(
            [
                {"model": "A", "sharpe": 1.5, "max_drawdown": float("nan")},
                {"model": "B", "sharpe": 1.6, "max_drawdown": -0.2},
            ]
        ),
    }
    records = ADAPTERS["pair_trading/practical"]("pair_trading/practical", payload)
    assert records
    assert all(r.metric_name != "max_drawdown" or r.model == "B" for r in records)
    assert not any(r.metric_name == "max_drawdown" and r.model == "A" for r in records)


def test_generic_adapter_candidate_on_empty_payload():
    adapter = ADAPTERS["feature_analysis/momentum"]
    records = adapter("feature_analysis/momentum", {"config": {}})
    assert len(records) == 1
    assert records[0].status == "candidate"
