"""Cómputo del movimiento geométrico browniano (GBM) de parámetros constantes.

El módulo contiene únicamente cálculos: no descarga datos, no dibuja figuras y
no mantiene estado global.  Las figuras del experimento viven en
``notebooks/spy/GeometricBrownianMotion.py``.

Convenciones
------------
* Los retornos son **log-retornos diarios** ``x_t = ln(S_t / S_{t-1})``.
* Los parámetros se annualizan con ``TRADING_DAYS`` (252 sesiones).
* El drift **geométrico** ``m~ = E[x] / dt`` es el que se estima
  directamente; el drift **aritmético** ``mu`` (el de la SDE) exige la
  corrección de Itô ``mu = m~ + sigma² / 2``.
* El Lema de Itô da la distribución **exacta** de ``S_t`` bajo GBM: no hace
  falta simular Monte Carlo para dibujar las bandas de probabilidad.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from thesis.characterization import block_bootstrap_ci
from thesis.config import DATA_DIR, END, SEED, START, TICKER, TRADING_DAYS
from thesis.market_data import load_single_price

#: Cuantiles del *fan chart*: 95 %, 80 %, 50 % centrales y la mediana.
BAND_LEVELS: tuple[float, ...] = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)

#: Tolerancia para emparejar un cuantil nominal con una clave de ``bands``.
_QUANTILE_TOL = 1e-9

__all__ = [
    "BAND_LEVELS",
    "DATA_DIR",
    "END",
    "SEED",
    "START",
    "TICKER",
    "TRADING_DAYS",
    "block_bootstrap_ci",
    "gbm_coverage",
    "gbm_density_grid",
    "gbm_mle_params",
    "gbm_quantile_bands",
    "load_single_price",
    "rolling_gbm_params",
]


def _finite_1d(x: Any, name: str = "x") -> np.ndarray:
    """Convierte ``x`` en un vector finito unidimensional."""
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} no puede estar vacío")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contiene valores no finitos")
    return values


def _check_trading_days(trading_days: int) -> None:
    """Valida el factor de anualización."""
    if trading_days <= 0:
        raise ValueError("trading_days debe ser positivo")


def gbm_mle_params(x: Any, trading_days: int = TRADING_DAYS) -> np.ndarray:
    """Estimadores MLE de un GBM de parámetros constantes (anualizados).

    Los estimadores de máxima verosimilitud son la media y el desvío muestral
    de los log-retornos, anualizados.  El drift geométrico ``m~`` se estima
    directamente; el drift aritmético ``mu`` se obtiene con la corrección de
    Itô ``mu = m~ + sigma² / 2``.

    Returns
    -------
    np.ndarray
        Vector ``[drift_geo, sigma, mu]`` con los tres parámetros anualizados.
    """
    values = _finite_1d(x, "log_ret")
    if values.size < 2:
        raise ValueError("se requieren al menos dos retornos")
    _check_trading_days(trading_days)

    drift_geo = values.mean() * trading_days
    sigma = values.std(ddof=1) * np.sqrt(trading_days)
    mu = drift_geo + 0.5 * sigma**2
    return np.array([drift_geo, sigma, mu])


def gbm_quantile_bands(
    S0: float,
    drift_geo: float,
    sigma: float,
    t_grid: Any,
    levels: Any = BAND_LEVELS,
) -> dict[float, np.ndarray]:
    """Cuantiles exactos de ``S_t`` bajo GBM en el grid temporal ``t_grid``.

    Por el Lema de Itô, ``ln S_t ~ N(ln S0 + m~ t, sigma² t)``; el cuantil
    ``q`` es ``exp(norm.ppf(q, loc=ln S0 + m~ t, scale=sigma sqrt(t)))``.

    En ``t = 0`` la banda colapsa a una delta de Dirac (``S0``); la escala
    ``sigma sqrt(t)`` se sustituye por ``1e-12`` para evitar la división por
    cero.

    Returns
    -------
    dict
        Diccionario ``{cuantil: ndarray}``; cada array tiene la longitud de
        ``t_grid``.
    """
    if not np.isfinite(S0) or S0 <= 0.0:
        raise ValueError("S0 debe ser un número positivo")
    if not np.isfinite(drift_geo):
        raise ValueError("drift_geo debe ser finito")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma debe ser no negativo")

    times = _finite_1d(t_grid, "t_grid")
    if np.any(times < 0.0):
        raise ValueError("t_grid no puede tener valores negativos")

    level_tuple = tuple(float(q) for q in levels)
    if not level_tuple:
        raise ValueError("levels no puede estar vacío")
    for q in level_tuple:
        if not 0.0 < q < 1.0:
            raise ValueError("los niveles deben estar entre 0 y 1")

    mu_t = np.log(S0) + drift_geo * times
    sigma_t = sigma * np.sqrt(times)
    sigma_t = np.where(sigma_t <= 0.0, 1e-12, sigma_t)
    return {q: np.exp(stats.norm.ppf(q, loc=mu_t, scale=sigma_t)) for q in level_tuple}


def gbm_density_grid(
    S0: float,
    drift_geo: float,
    sigma: float,
    t_grid: Any,
    n_bins: int = 200,
    normalize: str = "level",
) -> tuple[np.ndarray, np.ndarray]:
    """Densidad lognormal de ``S_t`` en una grilla de precios y tiempos.

    Es el **cómputo puro** del mapa de calor: el enmascarado fuera de la banda
    del 95 % y el ``pcolormesh`` viven en el cuaderno.

    La grilla de precios cubre el rango entre los cuantiles 0.5 % y 99.5 % de
    la distribución lognormal en todos los instantes.  Cada **nivel de
    precio** se normaliza por su densidad máxima a lo largo del tiempo
    (densidad **relativa** por fila).  El instante ``t = 0`` es una delta de
    Dirac: se excluye del normalizador (su pico no es comparable con el resto
    de la densidad) y el resultado se recorta a ``[0, 1]``.

    Con ``normalize="time"`` cada **instante** se normaliza por su propio
    máximo: la columna es la forma de la distribución en esa fecha (pico = 1,
    campana de ancho ``sigma sqrt(t)``).  Es la lectura natural de un *fan
    chart*: el color no depende de cómo se reparte la densidad entre fechas,
    de modo que no exagera el contraste en la franja de precios donde se
    cruzan las colas de fechas distintas.

    Returns
    -------
    (log_grid, density)
        ``log_grid`` tiene forma ``(n_bins,)`` (precios en log) y ``density``
        forma ``(n_bins, n_t)``: filas = niveles de precio, columnas =
        instantes.  Cada fila llega a 1 en su máximo.
    """
    if not np.isfinite(S0) or S0 <= 0.0:
        raise ValueError("S0 debe ser un número positivo")
    if not np.isfinite(drift_geo):
        raise ValueError("drift_geo debe ser finito")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma debe ser no negativo")
    if n_bins < 2:
        raise ValueError("n_bins debe ser al menos 2")
    if normalize not in ("level", "time"):
        raise ValueError("normalize debe ser 'level' o 'time'")

    times = _finite_1d(t_grid, "t_grid")
    if np.any(times < 0.0):
        raise ValueError("t_grid no puede tener valores negativos")

    mu_t = np.log(S0) + drift_geo * times
    sigma_t = sigma * np.sqrt(times)
    # En t = 0 la densidad es una delta de Dirac: se evita la división por cero.
    sigma_t = np.where(sigma_t <= 0.0, 1e-8, sigma_t)

    lower = float(stats.norm.ppf(0.005, loc=mu_t, scale=sigma_t).min())
    upper = float(stats.norm.ppf(0.995, loc=mu_t, scale=sigma_t).max())
    log_grid = np.linspace(lower, upper, int(n_bins))

    density = stats.norm.pdf(
        log_grid[None, :],
        loc=mu_t[:, None],
        scale=sigma_t[:, None],
    )
    if normalize == "time":
        time_max = density.max(axis=1, keepdims=True)
        density = np.divide(
            density, time_max, out=np.zeros_like(density), where=time_max > 0.0
        )
        return log_grid, density.T

    # Normalización por nivel de precio: el máximo de cada nivel (columna de
    # la disposición interna) sobre el tiempo es 1 en la disposición final
    # (n_bins, n_t).  La columna t = 0 (delta de Dirac) se excluye del
    # normalizador: su pico (~1/sigma_t) no es comparable con el resto.
    if density.shape[0] > 1:
        level_max = density[1:].max(axis=0, keepdims=True)
    else:
        level_max = density.max(axis=0, keepdims=True)
    density = np.divide(
        density,
        level_max,
        out=np.zeros_like(density),
        where=level_max > 0.0,
    )
    np.clip(density, 0.0, 1.0, out=density)
    return log_grid, density.T


def _band_lookup(bands: dict[float, Any], quantile: float) -> np.ndarray:
    """Devuelve la banda asociada a ``quantile`` (con tolerancia numérica)."""
    for key, value in bands.items():
        if abs(float(key) - quantile) <= _QUANTILE_TOL:
            return np.asarray(value, dtype=float).reshape(-1)
    raise ValueError(f"las bandas no contienen el cuantil {quantile}")


def gbm_coverage(
    test_price: Any,
    bands: dict[float, Any],
    levels: Any = (0.5, 0.8, 0.95),
) -> pd.DataFrame:
    """Cobertura empírica de las bandas centrales del GBM.

    Para cada nivel nominal ``p`` se mide la fracción de observaciones del
    período de validación que caen dentro de la banda central
    ``[(1-p)/2, 1-(1-p)/2]`` (50 % -> 0.25/0.75, 80 % -> 0.10/0.90,
    95 % -> 0.025/0.975).

    Returns
    -------
    pd.DataFrame
        Columnas ``nominal``, ``lower_quantile``, ``upper_quantile`` y
        ``coverage`` (fracción observada, numérica).
    """
    prices = _finite_1d(test_price, "test_price")

    rows: list[dict[str, float]] = []
    for nominal_value in levels:
        nominal = float(nominal_value)
        if not 0.0 < nominal < 1.0:
            raise ValueError("los niveles nominales deben estar entre 0 y 1")

        lower_quantile = round((1.0 - nominal) / 2.0, 12)
        upper_quantile = round(1.0 - lower_quantile, 12)
        lower = _band_lookup(bands, lower_quantile)
        upper = _band_lookup(bands, upper_quantile)
        if lower.size != prices.size or upper.size != prices.size:
            raise ValueError(
                "las bandas deben tener la misma longitud que la serie de precios"
            )

        inside = (prices >= lower) & (prices <= upper)
        rows.append(
            {
                "nominal": nominal,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "coverage": float(inside.mean()),
            }
        )
    return pd.DataFrame(rows)


def rolling_gbm_params(
    log_ret: Any,
    window: int = 60,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Parámetros del GBM reestimados en una ventana móvil (anualizados).

    La ventana es **trailing**: la fila ``t`` usa únicamente los retornos
    ``t-window+1 ... t`` (causal).  Las tres columnas son ``drift_geo``,
    ``sigma`` y ``mu`` (aritmético, con la corrección de Itô local).

    Returns
    -------
    pd.DataFrame
        Mismo índice que la serie de entrada; las primeras ``window-1`` filas
        son ``NaN`` (ventana incompleta).
    """
    if window < 2:
        raise ValueError("window debe ser al menos 2")
    _check_trading_days(trading_days)

    if isinstance(log_ret, pd.Series):
        returns = log_ret.astype(float)
    else:
        returns = pd.Series(_finite_1d(log_ret, "log_ret"))
    if returns.empty:
        raise ValueError("log_ret no puede estar vacío")

    rolling = returns.rolling(int(window))
    drift_geo = rolling.mean() * trading_days
    sigma = rolling.std(ddof=1) * np.sqrt(trading_days)
    mu = drift_geo + 0.5 * sigma**2
    return pd.DataFrame(
        {
            "drift_geo": drift_geo,
            "sigma": sigma,
            "mu": mu,
        }
    )
