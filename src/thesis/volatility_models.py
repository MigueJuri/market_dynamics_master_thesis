"""Modelos y utilidades de entrenamiento para predicción de volatilidad.

Incluye el benchmark HAR-RV (Corsi, 2009), una red LSTM parsimoniosa,
una función de pérdida LINEX asimétrica y el bucle de entrenamiento.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_volatility_seed(seed: int = 42) -> None:
    """Fija semillas de ``numpy`` y ``torch`` para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HARRVModel:
    """Modelo HAR-RV (Corsi, 2009) estimado por MCO con covarianza HAC.

    Especificación:
        ``y_t = β₀ + β_d·RV_t^(d) + β_w·RV_t^(w) + β_m·RV_t^(m) + ε_t``

    El modelo se entrena y predice en espacio de log-volatilidad crudo,
    por lo que las predicciones son comparables con el target sin invertir
    ninguna estandarización.
    """

    HAR_COLS: ClassVar[list[str]] = ["rv_daily", "rv_weekly", "rv_monthly"]

    def __init__(self) -> None:
        self.ols_res_: sm.regression.linear_model.RegressionResultsWrapper | None = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> HARRVModel:
        """Ajusta el modelo HAR-RV por MCO con errores HAC (Newey-West)."""
        X_ols = sm.add_constant(X_train[self.HAR_COLS].values)
        self.ols_res_ = sm.OLS(y_train, X_ols).fit(
            cov_type="HAC", cov_kwds={"maxlags": 10}
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones en espacio de log-vol crudo."""
        if self.ols_res_ is None:
            raise RuntimeError("El modelo HAR-RV no ha sido entrenado.")
        X_ols = sm.add_constant(X[self.HAR_COLS].values)
        return self.ols_res_.predict(X_ols)

    def summary(self) -> str:
        """Resumen de la regresión ajustada."""
        if self.ols_res_ is None:
            return "HAR-RV no entrenado"
        return str(self.ols_res_.summary())


class VolatilityLSTM(nn.Module):
    """Red LSTM de capacidad reducida para predicción de volatilidad.

    Arquitectura:
        ``Pre-LayerNorm → LSTM → Dropout → Residual → FC-ELU → Dropout → Escalar``

    Diseño conservador pensado para series financieras con baja relación
    señal/ruido.
    """

    def __init__(self, n_features: int, hidden: int = 16, dropout: float = 0.4):
        super().__init__()
        self.hidden = hidden

        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.norm1 = nn.LayerNorm(n_features)
        self.project_input = (
            nn.Linear(n_features, hidden) if n_features != hidden else nn.Identity()
        )
        self.drop1 = nn.Dropout(p=dropout)
        self.drop_fc = nn.Dropout(p=dropout / 2)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden // 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialización ortogonal y forget gate primeado a 1.0."""
        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
                if "bias_hh" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de forma ``(batch, lookback, n_features)``.

        Returns
        -------
        torch.Tensor
            Predicciones de forma ``(batch,)``.
        """
        identity = self.project_input(x)
        out_norm = self.norm1(x)
        out, _ = self.lstm1(out_norm)
        out = self.drop1(out)
        out = out + identity
        last = out[:, -1, :]
        dense = self.elu(self.fc1(last))
        dense = self.drop_fc(dense)
        return self.fc2(dense).squeeze(-1)


class ExponentialAsymmetricLoss(nn.Module):
    """Pérdida LINEX asimétrica para predicción de volatilidad.

    Formulación:
        ``L(e) = exp(b·e) - b·e - 1``,
        donde ``e = y - ŷ`` (error positivo = subestimación).

    Penaliza las subestimaciones exponencialmente y las sobreestimaciones
    de forma lineal.
    """

    def __init__(self, b: float = 1.5):
        super().__init__()
        self.b = b

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = y - y_hat
        return torch.mean(torch.exp(self.b * e) - (self.b * e) - 1.0)


def train_lstm(
    model: VolatilityLSTM,
    train_loader: DataLoader,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, list[float]]:
    """Entrena la LSTM con early stopping y checkpointing.

    Parameters
    ----------
    model : VolatilityLSTM
        Instancia del modelo a entrenar.
    train_loader : DataLoader
        Batches de entrenamiento.
    X_val_t, y_val_t : torch.Tensor
        Tensores de validación.
    config : dict
        Diccionario con al menos:
        ``lr, weight_decay, max_epochs, patience, min_epochs, grad_clip,
        linex_b, model_path``.
    device : torch.device
        Dispositivo de entrenamiento.
    criterion : nn.Module | None
        Función de pérdida.  Si es ``None`` se usa ``ExponentialAsymmetricLoss``.

    Returns
    -------
    dict
        Historial de ``train_loss`` y ``val_loss`` por época.
    """
    if criterion is None:
        criterion = ExponentialAsymmetricLoss(b=config["linex_b"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    model.to(device)
    X_val_t = X_val_t.to(device)
    y_val_t = y_val_t.to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_ctr = 0
    clip_count = 0
    total_batches = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            if torch.isnan(Xb).any() or torch.isnan(yb).any():
                continue

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()

            norm = nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            if norm > config["grad_clip"]:
                clip_count += 1
            total_batches += 1

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), config["model_path"])
        else:
            patience_ctr += 1

        if (epoch + 1) % 20 == 0:
            print(
                f"  Época {epoch + 1:3d} | "
                f"train: {avg_train:.6f} | val: {val_loss:.6f} (LINEX)"
            )

        if patience_ctr >= config["patience"] and epoch >= config["min_epochs"]:
            print(
                f"  Detención temprana en época {epoch + 1}. "
                f"Mejor val LINEX loss: {best_val:.6f}"
            )
            break

    clip_pct = 100.0 * clip_count / max(total_batches, 1)
    print(f"  Gradient clipping activado: {clip_pct:.1f}% de los batches.")
    if clip_pct > 5.0:
        print("  *** ALERTA: clipping > 5%. Considerar reducir lr. ***")

    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()
    return history
