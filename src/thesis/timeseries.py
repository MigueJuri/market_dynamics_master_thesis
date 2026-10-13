"""Utilidades de series temporales: retornos y ventanas móviles.

Convención de nomenclatura:

* ``feature``           : la característica cuya predictibilidad se mide.
* ``target``            : la variable que se intenta predecir (p. ej. retorno
                          futuro acumulado).
* ``feature_op``        : operación móvil aplicada a la característica
                          (``'sum'`` para momentum, ``'std'`` para
                          volatilidad, ``'mean'``, ``'var'`` o un callable).
* ``target_op``         : operación móvil aplicada al objetivo.  Por
                          defecto la suma de los retornos futuros.
* ``feature_windows``   : colección de ventanas de lookback (en sesiones).
* ``target_windows``    : colección de ventanas de lookahead (en sesiones).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


def log_returns(price: pd.Series) -> pd.Series:
    """Retorno logarítmico diario de ``price``."""
    return np.log(price / price.shift(1)).dropna()


def simple_returns(price: pd.Series) -> pd.Series:
    """Retorno simple diario de ``price``."""
    return (price / price.shift(1) - 1.0).dropna()


def rolling_forward_window(series: pd.Series, window: int) -> pd.Series:
    """Suma de las ``window`` observaciones futuras, anclada al día actual.

    ``series.rolling(window).sum().shift(-window)`` produce la suma de los
    próximos ``window`` valores terminando en ``t`` y desplazada hacia atrás
    para que el resultado esté alineado con el instante actual.
    """
    return series.rolling(window).sum().shift(-window)


RollingOp = Union[str, Callable[[pd.Series], float]]


def apply_window(
    series: pd.Series,
    window: float,
    operation: RollingOp,
    window_type: str = "fixed",
    ewm_adjust: bool = False,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Reduce ``series`` sobre una ventana móvil fija o exponencial.

    Parámetros
    ----------
    series : pd.Series
        Serie de entrada.
    window : float
        Tamaño de la ventana en sesiones.  Para ``window_type='fixed'``
        es el tamaño del rolling; para ``window_type='exponential'`` es
        el ``span`` pasado a :func:`pandas.Series.ewm`.
    operation : str o callable
        Reducción aplicada.  Con ``window_type='exponential'`` sólo se
        aceptan cadenas soportadas por ``Series.ewm`` (``'mean'``,
        ``'std'``, ``'var'``); con ``window_type='fixed'`` se aceptan
        además callables, que reciben la sub-serie móvil y devuelven un
        escalar.
    window_type : {'fixed', 'exponential'}
        Tipo de ventana.
    ewm_adjust : bool
        Si es ``True``, usa la forma con corrección de sesgo de
        ``Series.ewm`` (``adjust=True``).  Por defecto es ``False``, la
        forma recursiva equivalente a ``y_t = (1-α)·y_{t-1} + α·x_t``,
        que permite actualizar la serie de forma incremental sin
        recalcular toda la historia.
    min_periods : int, opcional
        Mínimo de observaciones requeridas.  Por defecto es ``window``
        para ventanas fijas y ``0`` para exponenciales.
    """
    if window_type == "fixed":
        window_i = int(window)
        mp = int(min_periods) if min_periods is not None else window_i
        rolling = series.rolling(window_i, min_periods=mp)
        if callable(operation):
            return rolling.agg(operation)
        method = getattr(rolling, operation, None)
        if method is None:
            raise ValueError(
                f"Operación móvil desconocida '{operation}'. Usa una cadena "
                "soportada por pd.Series.rolling ('sum', 'mean', 'std', "
                "'var', ...) o una función."
            )
        return method()

    if window_type == "exponential":
        if callable(operation):
            raise ValueError(
                "Las ventanas exponenciales no soportan operaciones callables; "
                "usa 'mean', 'std' o 'var'."
            )
        mp = int(min_periods) if min_periods is not None else 0
        ewm = series.ewm(span=float(window), adjust=ewm_adjust, min_periods=mp)
        method = getattr(ewm, operation, None)
        if method is None:
            raise ValueError(
                f"Operación '{operation}' no soportada por ewm. Usa 'mean', "
                "'std' o 'var'."
            )
        return method()

    raise ValueError(
        f"window_type debe ser 'fixed' o 'exponential', se recibió "
        f"'{window_type}'."
    )


def _apply_rolling(series: pd.Series, window: int, op: RollingOp) -> pd.Series:
    """Wrapper retro-compatible: ventana fija equivalente a ``apply_window``."""
    return apply_window(series, window, op, window_type="fixed")
