"""Tests para ``thesis.gbm``.

Los tests son **sintéticos**: no requieren red ni datos reales.  Se comprueban
las propiedades que sostienen el análisis del cuaderno
``notebooks/spy/GeometricBrownianMotion.py``:

1. los estimadores MLE del GBM recuperan los parámetros de un proceso
   simulado;
2. las bandas de cuantiles colapsan a ``S0`` en ``t = 0`` y son monótonas en
   el cuantil y en el tiempo;
3. la grilla de densidad tiene la forma esperada y cada fila está normalizada
   a 1;
4. la cobertura empírica de las bandas reproduce el nivel nominal;
5. los parámetros en ventana móvil son **causales** (el pasado no cambia al
   corromper la cola) y annualizados.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from thesis import gbm
from thesis.gbm import (
    BAND_LEVELS,
    gbm_coverage,
    gbm_density_grid,
    gbm_mle_params,
    gbm_quantile_bands,
    rolling_gbm_params,
)

#: Semilla fija: los tests son reproducibles.
SEED = 42

#: Parámetros del GBM simulado (anualizados).
TRUE_MU = 0.08  # drift aritmético
TRUE_SIGMA = 0.20  # volatilidad anual
TRADING_DAYS = 252


def _simulate_log_returns(
    n: int = 200_000,
    mu: float = TRUE_MU,
    sigma: float = TRUE_SIGMA,
    trading_days: int = TRADING_DAYS,
    seed: int = SEED,
) -> np.ndarray:
    """Log-retornos diarios de un GBM con parámetros conocidos."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days
    return rng.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        n,
    )


# --------------------------------------------------------------------------
# gbm_mle_params
# --------------------------------------------------------------------------
def test_gbm_mle_params_recovers_simulated_parameters() -> None:
    """Los estimadores MLE recuperan (drift geo, sigma, mu) del proceso."""
    returns = _simulate_log_returns()

    drift_geo, sigma, mu = gbm_mle_params(returns)

    assert drift_geo == pytest.approx(TRUE_MU - 0.5 * TRUE_SIGMA**2, abs=0.03)
    assert sigma == pytest.approx(TRUE_SIGMA, abs=0.005)
    # La corrección de Itô: mu = drift_geo + sigma² / 2.
    assert mu == pytest.approx(drift_geo + 0.5 * sigma**2, rel=1e-12)
    assert mu == pytest.approx(TRUE_MU, abs=0.03)


def test_gbm_mle_params_is_annualized() -> None:
    """La anualización escala el drift linealmente y sigma con la raíz."""
    returns = _simulate_log_returns(n=10_000)

    daily = gbm_mle_params(returns, trading_days=1)
    annual = gbm_mle_params(returns, trading_days=TRADING_DAYS)

    assert annual[0] == pytest.approx(daily[0] * TRADING_DAYS, rel=1e-12)
    assert annual[1] == pytest.approx(daily[1] * np.sqrt(TRADING_DAYS), rel=1e-12)


def test_gbm_mle_params_rejects_invalid_inputs() -> None:
    """Entradas vacías, no finitas o demasiado cortas fallan rápido."""
    with pytest.raises(ValueError, match="vacío"):
        gbm_mle_params(np.array([]))
    with pytest.raises(ValueError, match="no finitos"):
        gbm_mle_params(np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="al menos dos"):
        gbm_mle_params(np.array([0.1]))
    with pytest.raises(ValueError, match="trading_days"):
        gbm_mle_params(_simulate_log_returns(n=100), trading_days=0)


# --------------------------------------------------------------------------
# gbm_quantile_bands
# --------------------------------------------------------------------------
def test_gbm_quantile_bands_collapse_at_zero() -> None:
    """En ``t = 0`` todos los cuantiles valen ``S0`` (delta de Dirac)."""
    t_grid = np.array([0.0, 0.25, 1.0])
    bands = gbm_quantile_bands(100.0, 0.05, 0.2, t_grid)

    assert set(bands) == set(BAND_LEVELS)
    for values in bands.values():
        assert values[0] == pytest.approx(100.0, rel=1e-9)


def test_gbm_quantile_bands_monotone_in_level() -> None:
    """Para un tiempo fijo, el cuantil crece con el nivel."""
    t_grid = np.array([0.5, 2.0])
    bands = gbm_quantile_bands(100.0, 0.05, 0.2, t_grid, levels=BAND_LEVELS)

    ordered = [bands[q] for q in sorted(BAND_LEVELS)]
    for column in range(len(t_grid)):
        values = [band[column] for band in ordered]
        assert all(a < b for a, b in pairwise(values))


def test_gbm_quantile_bands_monotone_in_time() -> None:
    """Para un cuantil fijo, la banda crece con el tiempo si ``q > 0.5``."""
    t_grid = np.linspace(0.0, 5.0, 25)
    bands = gbm_quantile_bands(100.0, 0.0, 0.2, t_grid, levels=BAND_LEVELS)

    upper = bands[0.90]
    lower = bands[0.10]
    assert np.all(np.diff(upper) > 0.0)
    assert np.all(np.diff(lower) < 0.0)


def test_gbm_quantile_bands_rejects_invalid_inputs() -> None:
    """``S0``, ``sigma``, ``t_grid`` y los niveles se validan."""
    t_grid = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="S0"):
        gbm_quantile_bands(0.0, 0.05, 0.2, t_grid)
    with pytest.raises(ValueError, match="S0"):
        gbm_quantile_bands(-1.0, 0.05, 0.2, t_grid)
    with pytest.raises(ValueError, match="sigma"):
        gbm_quantile_bands(100.0, 0.05, -0.1, t_grid)
    with pytest.raises(ValueError, match="t_grid"):
        gbm_quantile_bands(100.0, 0.05, 0.2, np.array([0.0, -1.0]))
    with pytest.raises(ValueError, match="entre 0 y 1"):
        gbm_quantile_bands(100.0, 0.05, 0.2, t_grid, levels=(0.0, 0.5))
    with pytest.raises(ValueError, match="entre 0 y 1"):
        gbm_quantile_bands(100.0, 0.05, 0.2, t_grid, levels=(0.5, 1.0))
    with pytest.raises(ValueError, match="vacío"):
        gbm_quantile_bands(100.0, 0.05, 0.2, np.array([]))


# --------------------------------------------------------------------------
# gbm_density_grid
# --------------------------------------------------------------------------
def test_gbm_density_grid_shape_and_normalization() -> None:
    """La grilla tiene forma ``(n_bins, n_t)`` y cada fila llega a 1."""
    t_grid = np.linspace(0.0, 2.0, 40)
    n_bins = 64

    log_grid, density = gbm_density_grid(100.0, 0.05, 0.2, t_grid, n_bins=n_bins)

    assert log_grid.shape == (n_bins,)
    assert density.shape == (n_bins, len(t_grid))
    assert np.allclose(density.max(axis=1), 1.0)
    assert np.all(density >= 0.0)
    assert np.all(density <= 1.0)


def test_gbm_density_grid_preserves_the_lognormal_profile() -> None:
    """Cada fila (nivel de precio) es la densidad lognormal escalada."""
    from scipy import stats

    t_grid = np.array([0.5, 1.0, 2.0])
    drift_geo, sigma, s0 = 0.05, 0.2, 100.0

    log_grid, density = gbm_density_grid(s0, drift_geo, sigma, t_grid, n_bins=400)

    mu_t = np.log(s0) + drift_geo * t_grid
    sigma_t = sigma * np.sqrt(t_grid)
    pdf = stats.norm.pdf(
        log_grid[None, :],
        loc=mu_t[:, None],
        scale=sigma_t[:, None],
    )

    # La normalización es por fila: dentro de una fila, las razones entre
    # instantes son las de la densidad lognormal sin escalar.
    row = 200
    assert density[row, 1] / density[row, 2] == pytest.approx(
        pdf[1, row] / pdf[2, row], rel=1e-6
    )


def test_gbm_density_grid_rejects_invalid_inputs() -> None:
    """``S0``, ``sigma``, ``t_grid`` y ``n_bins`` se validan."""
    t_grid = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="S0"):
        gbm_density_grid(0.0, 0.05, 0.2, t_grid)
    with pytest.raises(ValueError, match="sigma"):
        gbm_density_grid(100.0, 0.05, -0.2, t_grid)
    with pytest.raises(ValueError, match="t_grid"):
        gbm_density_grid(100.0, 0.05, 0.2, np.array([0.0, -1.0]))
    with pytest.raises(ValueError, match="n_bins"):
        gbm_density_grid(100.0, 0.05, 0.2, t_grid, n_bins=1)


# --------------------------------------------------------------------------
# gbm_coverage
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("nominal", "lower_q", "upper_q"),
    [(0.5, 0.25, 0.75), (0.8, 0.10, 0.90), (0.95, 0.025, 0.975)],
)
def test_gbm_coverage_matches_nominal(
    nominal: float, lower_q: float, upper_q: float
) -> None:
    """La cobertura empírica de la banda reproduce el nivel nominal."""
    rng = np.random.default_rng(SEED)
    n_paths, n_steps = 2_000, 252  # un año de sesiones hábiles
    dt = 1.0 / TRADING_DAYS

    log_returns = rng.normal(
        (TRUE_MU - 0.5 * TRUE_SIGMA**2) * dt,
        TRUE_SIGMA * np.sqrt(dt),
        (n_paths, n_steps),
    )
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=1))

    t_grid = np.arange(n_steps) * dt
    bands = gbm_quantile_bands(
        100.0,
        TRUE_MU - 0.5 * TRUE_SIGMA**2,
        TRUE_SIGMA,
        t_grid,
    )
    assert lower_q in bands and upper_q in bands

    # Cada camino aporta su propia fracción de cobertura; el promedio sobre
    # caminos es el estimador insesgado del nivel nominal (la dispersión
    # entre caminos es alta: la banda se ensancha con el tiempo).
    coverages = [
        gbm_coverage(path, bands, levels=(nominal,)).loc[0, "coverage"]
        for path in prices
    ]
    assert float(np.mean(coverages)) == pytest.approx(nominal, abs=0.02)


@pytest.mark.parametrize(
    ("nominal", "lower_q", "upper_q"),
    [(0.5, 0.25, 0.75), (0.8, 0.10, 0.90), (0.95, 0.025, 0.975)],
)
def test_gbm_quantile_bands_have_the_nominal_pointwise_coverage(
    nominal: float, lower_q: float, upper_q: float
) -> None:
    """En un instante fijo, la banda contiene el nominal de las realizaciones."""
    rng = np.random.default_rng(SEED)
    horizon = 1.0
    n_draws = 5_000

    prices = 100.0 * np.exp(
        rng.normal(
            (TRUE_MU - 0.5 * TRUE_SIGMA**2) * horizon,
            TRUE_SIGMA * np.sqrt(horizon),
            n_draws,
        )
    )

    bands = gbm_quantile_bands(
        100.0,
        TRUE_MU - 0.5 * TRUE_SIGMA**2,
        TRUE_SIGMA,
        np.array([horizon]),
    )
    inside = (prices >= bands[lower_q]) & (prices <= bands[upper_q])

    assert float(inside.mean()) == pytest.approx(nominal, abs=0.02)


def test_gbm_coverage_reports_all_levels() -> None:
    """La tabla de cobertura tiene una fila por nivel nominal."""
    t_grid = np.linspace(0.0, 1.0, 50)
    bands = gbm_quantile_bands(100.0, 0.05, 0.2, t_grid)
    prices = 100.0 * np.exp(0.05 * t_grid)

    coverage = gbm_coverage(prices, bands, levels=(0.5, 0.8, 0.95))

    assert list(coverage["nominal"]) == [0.5, 0.8, 0.95]
    assert coverage["coverage"].between(0.0, 1.0).all()


def test_gbm_coverage_rejects_invalid_inputs() -> None:
    """Las bandas deben tener la longitud de la serie y el nivel ser válido."""
    t_grid = np.linspace(0.0, 1.0, 10)
    bands = gbm_quantile_bands(100.0, 0.05, 0.2, t_grid)
    prices = 100.0 * np.exp(0.05 * t_grid)

    with pytest.raises(ValueError, match="vacío"):
        gbm_coverage(np.array([]), bands)
    with pytest.raises(ValueError, match="no finitos"):
        gbm_coverage(np.full(len(prices), np.nan), bands)
    with pytest.raises(ValueError, match="longitud"):
        gbm_coverage(prices[:3], bands)
    with pytest.raises(ValueError, match="entre 0 y 1"):
        gbm_coverage(prices, bands, levels=(1.5,))
    with pytest.raises(ValueError, match="cuantil"):
        gbm_coverage(prices, bands, levels=(0.6,))


# --------------------------------------------------------------------------
# rolling_gbm_params
# --------------------------------------------------------------------------
def test_rolling_gbm_params_is_causal() -> None:
    """Corromper la cola no altera las filas anteriores."""
    rng = np.random.default_rng(SEED)
    returns = pd.Series(
        rng.normal(0.0002, 0.012, 200),
        index=pd.date_range("2020-01-01", periods=200, freq="B"),
    )

    original = rolling_gbm_params(returns, window=60)

    corrupted = returns.copy()
    corrupted.iloc[-10:] = 100.0
    recomputed = rolling_gbm_params(corrupted, window=60)

    pd.testing.assert_frame_equal(original.iloc[:-10], recomputed.iloc[:-10])
    assert original["sigma"].iloc[-1] != recomputed["sigma"].iloc[-1]


def test_rolling_gbm_params_columns_and_annualization() -> None:
    """Las columnas son ``drift_geo``, ``sigma`` y ``mu`` (anualizados)."""
    rng = np.random.default_rng(SEED)
    returns = pd.Series(
        rng.normal(0.0002, 0.012, 300),
        index=pd.date_range("2020-01-01", periods=300, freq="B"),
    )
    window = 60

    params = rolling_gbm_params(returns, window=window, trading_days=252)

    assert list(params.columns) == ["drift_geo", "sigma", "mu"]
    assert len(params) == len(returns)
    assert params.iloc[: window - 1].isna().all().all()
    assert params.iloc[window - 1].notna().all()

    tail = returns.iloc[-window:]
    expected_drift = tail.mean() * 252
    expected_sigma = tail.std(ddof=1) * np.sqrt(252)
    assert params["drift_geo"].iloc[-1] == pytest.approx(expected_drift)
    assert params["sigma"].iloc[-1] == pytest.approx(expected_sigma)
    assert params["mu"].iloc[-1] == pytest.approx(
        expected_drift + 0.5 * expected_sigma**2
    )


def test_rolling_gbm_params_rejects_invalid_inputs() -> None:
    """``window`` debe ser al menos 2 y la serie no puede estar vacía."""
    returns = pd.Series(np.linspace(-0.01, 0.01, 100))
    with pytest.raises(ValueError, match="window"):
        rolling_gbm_params(returns, window=1)
    with pytest.raises(ValueError, match="trading_days"):
        rolling_gbm_params(returns, window=10, trading_days=0)
    with pytest.raises(ValueError, match="vacío"):
        rolling_gbm_params(pd.Series(dtype=float))


# --------------------------------------------------------------------------
# Módulo
# --------------------------------------------------------------------------
def test_module_has_no_matplotlib_dependency() -> None:
    """El módulo de dominio no dibuja: no importa ``matplotlib``."""
    source = gbm.__file__ or ""
    assert source.endswith("gbm.py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "matplotlib" not in text
    assert "plt." not in text


@pytest.mark.integration
def test_spy_gbm_pipeline_end_to_end() -> None:
    """Prueba de integración con SPY real (requiere red o caché)."""
    from thesis.market_data import load_single_price

    price = load_single_price("SPY", "1993-01-01", "2026-01-01")
    log_ret = np.log(price / price.shift(1)).dropna()
    split_idx = price.index.asof(pd.Timestamp("2016-01-01"))
    train_ret = log_ret.loc[:split_idx]
    test_price = price.loc[split_idx:]

    drift_geo, sigma, mu = gbm_mle_params(train_ret.values)
    assert sigma > 0.0
    assert mu == pytest.approx(drift_geo + 0.5 * sigma**2)

    t_grid = np.arange(len(test_price)) / TRADING_DAYS
    bands = gbm_quantile_bands(float(price.loc[split_idx]), drift_geo, sigma, t_grid)
    coverage = gbm_coverage(test_price.values, bands, levels=(0.5, 0.8, 0.95))
    assert coverage["coverage"].between(0.0, 1.0).all()
