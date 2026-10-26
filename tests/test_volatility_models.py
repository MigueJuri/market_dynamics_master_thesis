"""Tests para ``thesis.volatility_models``.

Se saltan si ``torch`` no está instalado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch
from torch.utils.data import DataLoader, TensorDataset

from thesis.volatility_models import (
    ExponentialAsymmetricLoss,
    HARRVModel,
    VolatilityLSTM,
    set_volatility_seed,
    train_lstm,
)


def _make_har_data(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "rv_daily": rng.exponential(0.001, n),
            "rv_weekly": rng.exponential(0.001, n),
            "rv_monthly": rng.exponential(0.001, n),
        }
    )
    y = 0.001 + 0.3 * X["rv_daily"] + 0.4 * X["rv_weekly"] + 0.2 * X["rv_monthly"]
    y = y + rng.normal(0, 0.0001, n)
    return X, y.astype(np.float32)


def test_har_rv_model_smoke() -> None:
    X, y = _make_har_data(80)
    model = HARRVModel()
    model.fit(X.iloc[:60], y[:60])
    preds = model.predict(X.iloc[60:])
    assert len(preds) == 20
    assert isinstance(model.summary(), str)


def test_har_rv_model_not_fitted() -> None:
    model = HARRVModel()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict(
            pd.DataFrame(
                {"rv_daily": [0.001], "rv_weekly": [0.001], "rv_monthly": [0.001]}
            )
        )


def test_lstm_forward_shape() -> None:
    set_volatility_seed(0)
    model = VolatilityLSTM(n_features=4, hidden=8, dropout=0.2)
    x = torch.randn(16, 5, 4)
    out = model(x)
    assert out.shape == (16,)


def test_lstm_no_nan_output() -> None:
    set_volatility_seed(0)
    model = VolatilityLSTM(n_features=3, hidden=4, dropout=0.0)
    x = torch.randn(8, 10, 3)
    out = model(x)
    assert torch.isfinite(out).all()


def test_exponential_asymmetric_loss() -> None:
    loss_fn = ExponentialAsymmetricLoss(b=1.0)
    y = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    # Subestimación (y > y_hat) debe costar más que sobreestimación simétrica.
    underest = loss_fn(torch.tensor([-0.5, 0.5, -1.5], dtype=torch.float32), y)
    overest = loss_fn(torch.tensor([0.5, 1.5, -0.5], dtype=torch.float32), y)
    assert underest.item() > overest.item()


def test_train_lstm_smoke(tmp_path) -> None:
    set_volatility_seed(0)
    device = torch.device("cpu")

    # Datos sintéticos pequeños.
    X_train = torch.randn(40, 5, 3)
    y_train = torch.randn(40)
    X_val = torch.randn(10, 5, 3)
    y_val = torch.randn(10)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)

    model = VolatilityLSTM(n_features=3, hidden=4, dropout=0.0)
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 10,
        "patience": 5,
        "min_epochs": 2,
        "grad_clip": 1.0,
        "linex_b": 0.5,
        "model_path": str(tmp_path / "model.pt"),
    }

    history = train_lstm(model, train_loader, X_val, y_val, config, device)
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) <= config["max_epochs"]
    assert len(history["train_loss"]) >= config["min_epochs"]
    assert (tmp_path / "model.pt").exists()
