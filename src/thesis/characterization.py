"""Cómputo estadístico para caracterizar retornos y sus cuadrados.

El módulo contiene únicamente cálculos: no descarga datos, no dibuja figuras y
no mantiene estado global.  Las figuras del experimento viven en
``notebooks/spy/SPY_caracterizacion.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from arch.bootstrap import IIDBootstrap, MovingBlockBootstrap
from scipy import stats


def _finite_values(x: Any, name: str = "x") -> np.ndarray:
    """Convierte ``x`` en un vector finito unidimensional."""
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} no puede estar vacío")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contiene valores no finitos")
    return values


def describe_returns(x: Any, label: str) -> pd.Series:
    """Momentos muestrales y test de normalidad de Jarque-Bera."""
    values = _finite_values(x, "returns")
    jb_stat, jb_pvalue = stats.jarque_bera(values)
    return pd.Series(
        {
            "n": values.size,
            "media": values.mean(),
            "desvío": values.std(ddof=1),
            "mediana": np.median(values),
            "mín": values.min(),
            "máx": values.max(),
            "asimetría": stats.skew(values),
            "curtosis (exceso)": stats.kurtosis(values, fisher=True),
            "Jarque-Bera": jb_stat,
            "p-valor JB": jb_pvalue,
        },
        name=label,
    )


def freedman_diaconis_bins(x: Any) -> int:
    """Número de bines según la regla de Freedman-Diaconis.

    La regla usa ``h = 2 IQR n^(-1/3)``.  Una serie constante se representa
    con un único bin.
    """
    values = _finite_values(x, "returns")
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    value_range = values.max() - values.min()
    if iqr <= 0.0 or value_range <= 0.0:
        return 1
    bin_width = 2.0 * iqr * values.size ** (-1.0 / 3.0)
    return max(1, int(np.ceil(value_range / bin_width)))


def fit_return_distribution(x: Any) -> dict[str, float]:
    """Ajusta una normal y una t de Student por máxima verosimilitud."""
    values = _finite_values(x, "returns")
    mu, sigma = stats.norm.fit(values)
    degrees_of_freedom, loc, scale = stats.t.fit(values)
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "df": float(degrees_of_freedom),
        "loc": float(loc),
        "scale": float(scale),
    }


def fit_squared_return_distributions(x: Any) -> dict[str, float]:
    """Ajusta una log-normal a ``x²`` y una normal a ``log(x²)``.

    Los retornos nulos no tienen logaritmo y se excluyen de ambos ajustes.
    La log-normal se parametriza con ``loc=0`` para que su soporte sea
    estrictamente positivo.
    """
    values = _finite_values(x, "returns")
    squared = values**2
    positive_squared = squared[squared > 0.0]
    if positive_squared.size == 0:
        raise ValueError("debe haber al menos un retorno no nulo")

    log_squared = np.log(positive_squared)
    lognorm_shape, lognorm_loc, lognorm_scale = stats.lognorm.fit(
        positive_squared, floc=0.0
    )
    normal_mu, normal_sigma = stats.norm.fit(log_squared)
    return {
        "lognorm_shape": float(lognorm_shape),
        "lognorm_loc": float(lognorm_loc),
        "lognorm_scale": float(lognorm_scale),
        "log_mu": float(normal_mu),
        "log_sigma": float(normal_sigma),
        "n_positive": float(positive_squared.size),
    }


def excess_kurtosis(x: Any) -> np.ndarray:
    """Exceso de curtosis de Fisher (cero para una gaussiana)."""
    values = _finite_values(x, "returns")
    return np.array([stats.kurtosis(values, fisher=True)])


def annualized_stats(x: Any, trading_days: int = 252) -> np.ndarray:
    """Media y volatilidad anualizadas a partir de retornos diarios."""
    values = _finite_values(x, "returns")
    if trading_days <= 0:
        raise ValueError("trading_days debe ser positivo")
    if values.size < 2:
        raise ValueError("se requieren al menos dos retornos")
    return np.array(
        [
            values.mean() * trading_days,
            values.std(ddof=1) * np.sqrt(trading_days),
        ]
    )


def _percentile_interval(
    bootstrap: Any,
    statistic: Callable[[np.ndarray], np.ndarray],
    reps: int,
    alpha: float,
) -> np.ndarray:
    """Intervalo percentil ``(2, k)`` de un bootstrap de ``arch``."""
    return np.asarray(
        bootstrap.conf_int(
            statistic,
            reps=reps,
            method="percentile",
            size=1.0 - alpha,
        ),
        dtype=float,
    )


def _statistic_wrapper(func: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Normaliza el estadístico a un vector 1-D de floats."""
    def statistic(x: np.ndarray) -> np.ndarray:
        result = np.atleast_1d(np.asarray(func(x), dtype=float))
        return result.reshape(-1)

    return statistic


def block_bootstrap_ci(
    x: Any,
    func: Callable[[np.ndarray], Any],
    block_size: int = 10,
    reps: int = 3_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimación puntual e intervalo percentil por moving block bootstrap.

    Returns
    -------
    (estimación, intervalo)
        ``estimación`` tiene forma ``(k,)`` e ``intervalo`` forma ``(2, k)``;
        la primera fila contiene el límite inferior y la segunda el superior.
    """
    values = _finite_values(x, "returns")
    if block_size <= 0 or block_size > values.size:
        raise ValueError("block_size debe estar entre 1 y el tamaño de la muestra")
    if reps <= 0:
        raise ValueError("reps debe ser positivo")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")

    statistic = _statistic_wrapper(func)
    bootstrap = MovingBlockBootstrap(block_size, x=values, seed=seed)
    interval = _percentile_interval(bootstrap, statistic, reps, alpha)
    return statistic(values), interval


def bootstrap_ci(
    x: Any,
    func: Callable[[np.ndarray], Any],
    reps: int = 3_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimación puntual e intervalo percentil por bootstrap i.i.d.

    Cada réplica remuestrea observaciones individuales con reemplazo: la
    dependencia temporal **no** se preserva (a diferencia de
    :func:`block_bootstrap_ci`, que remuestrea bloques contiguos).

    Returns
    -------
    (estimación, intervalo)
        ``estimación`` tiene forma ``(k,)`` e ``intervalo`` forma ``(2, k)``;
        la primera fila contiene el límite inferior y la segunda el superior.
    """
    values = _finite_values(x, "returns")
    if reps <= 0:
        raise ValueError("reps debe ser positivo")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")

    statistic = _statistic_wrapper(func)
    bootstrap = IIDBootstrap(values, seed=seed)
    interval = _percentile_interval(bootstrap, statistic, reps, alpha)
    return statistic(values), interval


def heteroskedastic_robust_acf(
    returns: Any,
    max_lags: int = 50,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """ACF muestral con bandas robustas a heterocedasticidad.

    Implementa la corrección de Diebold (1986): el límite robusto sustituye
    la varianza de ruido blanco por
    ``E[r_t² r_(t-k)²] / gamma_0²``.  Las bandas nunca son más estrechas que
    las de Bartlett.

    Returns
    -------
    (acf, límite)
        Arrays de longitud ``max_lags + 1``; la banda es simétrica
        ``± límite``.
    """
    values = _finite_values(returns, "returns")
    if max_lags < 0:
        raise ValueError("max_lags no puede ser negativo")
    if max_lags >= values.size:
        raise ValueError("max_lags debe ser menor que el tamaño de la muestra")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")

    centered = values - values.mean()
    n_obs = centered.size
    gamma_0 = np.mean(centered**2)
    if gamma_0 <= 0.0:
        raise ValueError("los retornos deben tener varianza positiva")

    autocovariances = np.correlate(centered, centered, mode="full")[n_obs - 1 :]
    acf = np.empty(max_lags + 1)
    acf[0] = 1.0
    acf[1:] = autocovariances[1 : max_lags + 1] / (n_obs * gamma_0)

    squared_autocovariances = np.correlate(centered**2, centered**2, mode="full")[
        n_obs - 1 :
    ]
    variance_ratios = squared_autocovariances[1 : max_lags + 1] / (n_obs * gamma_0**2)
    z_value = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_bound = np.zeros(max_lags + 1)
    ci_bound[1:] = z_value * np.sqrt(np.maximum(variance_ratios, 1.0) / n_obs)
    return acf, ci_bound


def sample_acf(x: Any, max_lags: int = 50) -> np.ndarray:
    """ACF muestral simple, incluida la autocorrelación en el lag cero."""
    values = _finite_values(x, "values")
    if max_lags < 0:
        raise ValueError("max_lags no puede ser negativo")
    if max_lags >= values.size:
        raise ValueError("max_lags debe ser menor que el tamaño de la muestra")

    centered = values - values.mean()
    n_obs = centered.size
    gamma_0 = np.mean(centered**2)
    if gamma_0 <= 0.0:
        raise ValueError("la serie debe tener varianza positiva")
    autocovariances = np.correlate(centered, centered, mode="full")[n_obs - 1 :]
    acf = np.empty(max_lags + 1)
    acf[0] = 1.0
    acf[1:] = autocovariances[1 : max_lags + 1] / (n_obs * gamma_0)
    return acf


def hac_acf(
    x: Any,
    max_lags: int = 50,
    alpha: float = 0.05,
    bandwidth: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ACF muestral con errores estándar HAC (Newey-West) sobre ``rho_k``.

    Implementa el estimador robusto de Francq-Zakoïan / Lobato-Nankervis-Savin:
    ``rho_k`` se linealiza como ``mean(psi_t)`` con
    ``psi_t = [(z_t - m)(z_(t-k) - m) - rho_k (z_t - m)^2] / gamma_0`` y su
    varianza asintótica es la varianza de largo plazo de ``psi_t`` (núcleo de
    Bartlett) dividida por ``T``.  Es válido bajo heterocedasticidad y
    dependencia arbitraria (p. ej. ``z = x²``), sin elegir tamaño de bloque.

    Parameters
    ----------
    bandwidth : int, optional
        Truncamiento del núcleo; por defecto ``floor(4 (T/100)^(2/9))``.

    Returns
    -------
    (acf, half_width)
        Arrays de longitud ``max_lags + 1``; la banda es ``acf ± half_width``
        y el lag cero no tiene banda.
    """
    values = _finite_values(x, "values")
    if max_lags < 0:
        raise ValueError("max_lags no puede ser negativo")
    if max_lags >= values.size:
        raise ValueError("max_lags debe ser menor que el tamaño de la muestra")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")

    n_obs = values.size
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    if bandwidth < 0:
        raise ValueError("bandwidth no puede ser negativo")

    acf = sample_acf(values, max_lags)
    centered = values - values.mean()
    gamma_0 = np.mean(centered**2)
    z_value = stats.norm.ppf(1.0 - alpha / 2.0)

    half_width = np.zeros(max_lags + 1)
    for k in range(1, max_lags + 1):
        psi = (
            centered[k:] * centered[:-k] - acf[k] * centered[k:] ** 2
        ) / gamma_0
        psi = psi - psi.mean()
        m = psi.size
        long_run = np.dot(psi, psi) / m
        for j in range(1, min(bandwidth, m - 1) + 1):
            weight = 1.0 - j / (bandwidth + 1.0)
            long_run += 2.0 * weight * np.dot(psi[j:], psi[:-j]) / m
        half_width[k] = z_value * np.sqrt(max(long_run, 0.0) / m)
    return acf, half_width


def ljung_box_pvalues(x: Any, lags: Sequence[int] = (10, 20, 50)) -> pd.Series:
    """P-valores de Ljung-Box (``H0``: sin autocorrelación hasta el lag ``m``).

    Aplicado a ``x²`` equivale al test de McLeod-Li (efectos ARCH).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    values = _finite_values(x, "values")
    lag_list = [int(m) for m in lags]
    if not lag_list or min(lag_list) < 1 or max(lag_list) >= values.size:
        raise ValueError("lags debe contener enteros en [1, T - 1]")
    result = acorr_ljungbox(values, lags=lag_list, return_df=True)
    return pd.Series(result["lb_pvalue"].to_numpy(), index=lag_list, name="lb_pvalue")


def bootstrap_acf(
    x: Any,
    max_lags: int = 50,
    block_size: int = 10,
    reps: int = 3_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """ACF muestral con bandas de confianza por moving block bootstrap.

    El estimador de autocorrelación se recalcula sobre cada réplica del
    *moving block bootstrap* (Künsch, 1989): se concatenan bloques
    consecutivos de longitud ``block_size``, de modo que la dependencia
    temporal de corto plazo (dentro de cada bloque) se conserva.  La
    autocorrelación en lags ``k >= block_size`` se atenúa por las uniones
    entre bloques, por lo que ``block_size`` debería ser, al menos, del orden
    de ``max_lags`` si se desea cubrir toda la ACF.

    La banda es el intervalo percentil de las réplicas, **centrado en la
    estimación puntual** (no en cero): su semiancho es la mayor distancia
    entre la estimación y los extremos del intervalo.  Un lag es
    «significativo» (``|acf| > half_width``) cuando el intervalo no contiene
    al cero.

    Returns
    -------
    (acf, half_width)
        Arrays de longitud ``max_lags + 1``; la banda es ``acf ± half_width``
        y el lag cero no tiene banda.
    """

    values = _finite_values(x, "returns")
    if max_lags < 0:
        raise ValueError("max_lags no puede ser negativo")
    if max_lags >= values.size:
        raise ValueError("max_lags debe ser menor que el tamaño de la muestra")

    point, interval = block_bootstrap_ci(
        values,
        lambda v: sample_acf(v, max_lags),
        block_size=block_size,
        reps=reps,
        alpha=alpha,
        seed=seed,
    )
    half_width = np.maximum(
        np.abs(interval[1] - point), np.abs(point - interval[0])
    )
    half_width[0] = 0.0
    point[0] = 1.0
    return point, half_width


def compute_ewma_volatility(
    log_ret: pd.Series | Any,
    lambda_ewma: float = 0.94,
    trading_days: int = 252,
) -> pd.Series:
    """Volatilidad EWMA anualizada y causal.

    En la fecha ``t`` se usa únicamente ``x_(t-1)²``:

    ``sigma_t² = lambda sigma_(t-1)² + (1-lambda) x_(t-1)²``.
    """
    if not 0.0 < lambda_ewma < 1.0:
        raise ValueError("lambda_ewma debe estar entre 0 y 1")
    if trading_days <= 0:
        raise ValueError("trading_days debe ser positivo")
    if isinstance(log_ret, pd.Series):
        returns = log_ret.astype(float)
    else:
        returns = pd.Series(_finite_values(log_ret, "log_ret"))
    if returns.empty:
        raise ValueError("log_ret no puede estar vacío")

    lagged_squared_returns = returns.pow(2.0).shift(1)
    ewma_variance = lagged_squared_returns.ewm(
        alpha=1.0 - lambda_ewma,
        adjust=False,
    ).mean()
    volatility = np.sqrt(ewma_variance) * np.sqrt(trading_days)
    volatility.name = "ewma_volatility"
    return volatility


def high_volatility_mask(
    volatility: pd.Series,
    quantile: float = 0.90,
    index: pd.Index | Sequence[Any] | None = None,
) -> pd.Series:
    """Marca los valores de volatilidad por encima de un cuantil global.

    El cuantil se calcula sobre toda la muestra: es una comparación
    descriptiva, no una señal libre de look-ahead.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile debe estar entre 0 y 1")
    if not isinstance(volatility, pd.Series):
        volatility = pd.Series(np.asarray(volatility, dtype=float))
    valid = volatility.dropna()
    if valid.empty:
        raise ValueError("volatility no contiene valores finitos")

    threshold = float(valid.quantile(quantile))
    target_index = volatility.index if index is None else pd.Index(index)
    mask = volatility.gt(threshold).reindex(target_index).fillna(False).astype(bool)
    mask.name = "high_volatility"
    return mask


def unconditional_band_excursions(
    log_ret: pd.Series | Any,
    ks: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
) -> pd.DataFrame:
    """Excursiones fuera de bandas incondicionales frente a una gaussiana."""
    values = _finite_values(log_ret, "log_ret")
    if values.size < 2:
        raise ValueError("se requieren al menos dos retornos")
    sigma = values.std(ddof=1)
    if sigma <= 0.0:
        raise ValueError("los retornos deben tener varianza positiva")

    standardized = np.abs((values - values.mean()) / sigma)
    rows = []
    for k in ks:
        if k <= 0:
            raise ValueError("los umbrales k deben ser positivos")
        empirical = float(np.mean(standardized > k))
        gaussian = float(2.0 * stats.norm.sf(k))
        rows.append(
            {
                "k": float(k),
                "empirical": empirical,
                "gaussian": gaussian,
                "ratio": empirical / gaussian,
            }
        )
    return pd.DataFrame(rows)


def build_summary_table(
    log_ret: pd.Series | Any,
    annualized: Sequence[float] | np.ndarray | None = None,
    kurtosis: float | Sequence[float] | np.ndarray | None = None,
    distribution: Mapping[str, float] | None = None,
    acf_returns: tuple[np.ndarray, np.ndarray] | None = None,
    acf_squared: np.ndarray | None = None,
    squared_margin: np.ndarray | None = None,
    *,
    ticker: str = "SPY",
    alpha: float = 0.05,
    max_lags: int | None = None,
) -> pd.DataFrame:
    """Construye la tabla resumen de la caracterización empírica."""
    values = _finite_values(log_ret, "log_ret")
    if values.size < 2:
        raise ValueError("se requieren al menos dos retornos")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")

    annual_values = (
        annualized_stats(values)
        if annualized is None
        else np.asarray(annualized, dtype=float).reshape(-1)
    )
    if annual_values.size < 2:
        raise ValueError("annualized debe contener media y volatilidad")

    kurtosis_value = (
        float(excess_kurtosis(values)[0])
        if kurtosis is None
        else float(np.asarray(kurtosis, dtype=float).reshape(-1)[0])
    )
    fitted = (
        fit_return_distribution(values) if distribution is None else dict(distribution)
    )
    if "df" not in fitted:
        raise ValueError("distribution debe incluir los grados de libertad 'df'")

    if max_lags is None:
        if acf_returns is not None:
            max_lags = len(acf_returns[0]) - 1
        else:
            max_lags = min(50, values.size - 1)
    if max_lags < 0 or max_lags >= values.size:
        raise ValueError("max_lags no es válido para la muestra")

    if acf_returns is None:
        acf_returns = heteroskedastic_robust_acf(values, max_lags=max_lags, alpha=alpha)
    if acf_squared is None:
        acf_squared = sample_acf(values**2, max_lags=max_lags)
    if squared_margin is None:
        squared_margin = np.full(
            max_lags + 1,
            stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(values.size),
        )
        squared_margin[0] = 0.0

    acf_r, ci_r = acf_returns
    significant_returns = int(np.sum(np.abs(acf_r[1:]) > ci_r[1:]))
    significant_squared = int(np.sum(np.abs(acf_squared[1:]) > squared_margin[1:]))
    z_scores = np.abs((values - values.mean()) / values.std(ddof=1))
    fraction_beyond_three = float(np.mean(z_scores > 3.0))

    rows = [
        ("Observaciones", f"{values.size:,}", "—"),
        ("Media anualizada", f"{annual_values[0]:.2%}", "—"),
        ("Volatilidad anualizada", f"{annual_values[1]:.2%}", "—"),
        ("Asimetría", f"{stats.skew(values):.2f}", "0"),
        ("Exceso de curtosis", f"{kurtosis_value:.2f}", "0"),
        ("Grados de libertad (t)", f"{float(fitted['df']):.1f}", "∞"),
        (
            "Jarque-Bera (p-valor)",
            f"{stats.jarque_bera(values)[1]:.1e}",
            f"> {alpha:.2f}",
        ),
        (
            "Lags significativos en ACF(x)",
            f"{significant_returns}/{max_lags}",
            f"~{int(alpha * max_lags)}/{max_lags}",
        ),
        (
            "Lags significativos en ACF(x²)",
            f"{significant_squared}/{max_lags}",
            f"~{int(alpha * max_lags)}/{max_lags}",
        ),
        (
            "Fracción |z| > 3σ",
            f"{fraction_beyond_three:.3%}",
            f"{2.0 * stats.norm.sf(3.0):.3%}",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["Métrica", f"{ticker} (observado)", "Bajo normalidad i.i.d."],
    )
