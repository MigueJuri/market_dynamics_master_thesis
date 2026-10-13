"""Construcción de características financieras y targets específicos.

Incluye transformaciones que combinan retornos y volatilidad, así como
objetivos relativos a la media móvil histórica.  Todas las funciones
preservan el índice temporal de la serie original y no introducen
lookahead bias.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from thesis.timeseries import apply_window


def future_distance_to_moving_average(
    source_series: pd.Series,
    tau_p: float,
    tau_f: int,
    window_type: str = "fixed",
    ewm_adjust: bool = False,
) -> pd.Series:
    """Distancia futura del log-precio a su media móvil histórica.

    Devuelve la serie ``log P_{t + τ_f} - MA_{τ_p}(log P)_t``.

    La media móvil se calcula exclusivamente con información disponible
    en ``t`` (incluye ``log P_t``, nunca usa precios futuros), por lo
    que no introduce lookahead bias.  El resultado queda alineado al
    índice de ``source_series`` (anclado en ``t``).  ``window_type``
    selecciona entre media fija y exponencial igual que
    :func:`thesis.timeseries.apply_window`.
    """
    future = source_series.shift(-int(tau_f))
    moving = apply_window(
        source_series, float(tau_p), "mean",
        window_type=window_type, ewm_adjust=ewm_adjust,
    )
    return future - moving


def volatility_scaled_return(
    returns: pd.Series,
    window: int,
    sigma_floor: float = 1e-12,
    ddof: int = 1,
) -> pd.Series:
    r"""Retorno acumulado pasado escalado por la volatilidad pasada.

    Para cada fecha ``t``:

    .. math:: f_t = \frac{\sum_{k=1}^{w} r_{t-k}}
                       {\operatorname{std}(r_{t-w}, \ldots, r_{t-1})}

    La ventana del retorno y la de la volatilidad coinciden y se
    calculan sobre ``returns.shift(1)``, de modo que no se incluye el
    retorno del día ``t``.  ``sigma_floor`` evita divisiones por cero
    cuando la volatilidad es nula.
    """
    shifted = returns.shift(1)
    ret_sum = shifted.rolling(window).sum()
    sigma = shifted.rolling(window, min_periods=window).std(ddof=ddof)
    if sigma_floor is not None:
        sigma = sigma.clip(lower=sigma_floor)
    return ret_sum / sigma


def future_return_over_past_volatility(
    returns: pd.Series,
    tau_p: float,
    tau_f: int,
    sigma_floor: float = 1e-12,
    ddof: int = 1,
    **_: object,
) -> pd.Series:
    r"""Retorno futuro escalado por la volatilidad histórica pasada.

    Para cada fecha ``t``:

    .. math:: y_t = \frac{\sum_{k=1}^{\tau_f} r_{t+k}}
                       {\operatorname{std}(r_{t-\tau_p+1}, \ldots, r_{t-1})}

    El denominador usa exclusivamente retornos hasta ``t-1`` (sin
    lookahead bias).  ``sigma_floor`` evita divisiones por cero.
    ``**_`` admite kwargs adicionales (``window_type``,
    ``ewm_adjust``) para interoperar con
    :func:`thesis.correlation.compute_correlation_matrix` y
    :func:`thesis.correlation.compute_cross_sectional_correlation` como
    ``target_transform``.
    """
    shifted = returns.shift(1)
    sigma = shifted.rolling(int(tau_p), min_periods=int(tau_p)).std(ddof=ddof)
    fut_sum = returns.rolling(int(tau_f)).sum().shift(-int(tau_f))
    if sigma_floor is not None:
        sigma = sigma.clip(lower=sigma_floor)
    return fut_sum / sigma


def make_scaled_return_op(
    sigma_floor: float = 1e-12,
    ddof: int = 1,
) -> Callable[[pd.Series], float]:
    """Callable para usar como ``feature_op`` en ventanas fijas.

    Recibe una ventana de retornos (ya desplazada para excluir ``r_t``)
    y devuelve ``sum(window) / std(window)``.  Pensado para combinar con
    :func:`thesis.correlation.compute_correlation_matrix` y
    :func:`thesis.correlation.compute_cross_sectional_correlation` cuando
    la característica es un retorno escalado por volatilidad.
    """
    def _op(window: pd.Series) -> float:
        if len(window) == 0:
            return np.nan
        sigma = window.std(ddof=ddof)
        if np.isnan(sigma) or sigma == 0:
            return np.nan
        if sigma_floor is not None and sigma < sigma_floor:
            sigma = sigma_floor
        return float(window.sum() / sigma)

    return _op
