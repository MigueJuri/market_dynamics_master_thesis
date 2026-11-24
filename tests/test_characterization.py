"""Tests para ``thesis.characterization``.

Los tests son **sintéticos**: no requieren red ni datos reales.  Se comprueban
las cinco propiedades que sostienen el análisis del cuaderno
``notebooks/spy/SPY_caracterizacion.py``:

1. la regla de Freedman–Diaconis es monótona en la dispersión;
2. los momentos muestrales de una gaussiana reproducen sus valores teóricos;
3. la ACF de un ruido blanco queda dentro de la banda de Bartlett;
4. la EWMA es **causal** (el pasado no cambia al corromper la cola) y las
   bandas de Diebold nunca son más estrechas que las de Bartlett;
5. las bandas de la ACF por bootstrap i.i.d. cubren un ruido blanco y no
   heredan la dependencia temporal de la serie.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from thesis import characterization
from thesis.characterization import (
    annualized_stats,
    block_bootstrap_ci,
    bootstrap_acf,
    bootstrap_ci,
    build_summary_table,
    compute_ewma_volatility,
    describe_returns,
    excess_kurtosis,
    fit_return_distribution,
    fit_squared_return_distributions,
    freedman_diaconis_bins,
    hac_acf,
    heteroskedastic_robust_acf,
    high_volatility_mask,
    sample_acf,
    unconditional_band_excursions,
)

#: Semilla fija: los tests son reproducibles.
SEED = 42


def _white_noise(n: int, seed: int = SEED) -> np.ndarray:
    """Ruido blanco gaussiano de ``n`` observaciones."""
    return np.random.default_rng(seed).standard_normal(n)


def _arch_returns(
    n: int = 4_000, alpha0: float = 0.05, alpha1: float = 0.30, seed: int = SEED
) -> np.ndarray:
    """Retornos de un proceso ARCH(1): ``sigma_t² = alpha0 + alpha1 r_(t-1)²``."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    returns = np.empty(n)
    variance = alpha0
    for t in range(n):
        returns[t] = np.sqrt(variance) * shocks[t]
        variance = alpha0 + alpha1 * returns[t] ** 2
    return returns


def _business_index(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    """Índice de sesiones hábiles para las series de los tests."""
    return pd.date_range(start, periods=n, freq="B")


# ---------------------------------------------------------------------------
# Freedman–Diaconis
# ---------------------------------------------------------------------------


def test_freedman_diaconis_bins_minimum_is_one() -> None:
    """Una serie constante se representa con un único bin."""
    assert freedman_diaconis_bins(np.ones(100)) == 1


def test_freedman_diaconis_bins_grows_with_dispersion() -> None:
    """Más dispersión (misma masa central, outliers) ⇒ más bines."""
    base = _white_noise(2_000, seed=1)
    tight = freedman_diaconis_bins(base)
    wide = freedman_diaconis_bins(np.concatenate([base, [30.0, -30.0]]))

    assert tight >= 1
    assert wide > tight


# ---------------------------------------------------------------------------
# Momentos muestrales
# ---------------------------------------------------------------------------


def test_describe_returns_gaussian_moments() -> None:
    """En una gaussiana: asimetría nula y exceso de curtosis nulo."""
    x = _white_noise(30_000, seed=2)
    described = describe_returns(x, "gaussiana")

    assert described["n"] == 30_000
    assert described["asimetría"] == pytest.approx(0.0, abs=0.04)
    assert described["curtosis (exceso)"] == pytest.approx(0.0, abs=0.08)
    assert described["media"] == pytest.approx(0.0, abs=0.02)
    assert described["desvío"] == pytest.approx(1.0, abs=0.02)


def test_describe_returns_rejects_jarque_bera_on_fat_tails() -> None:
    """Una mezcla de normales con colas pesadas rechaza la normalidad."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5_000) + 0.5 * rng.standard_t(df=3, size=5_000)
    described = describe_returns(x, "colas pesadas")

    assert described["p-valor JB"] < 0.01
    assert described["curtosis (exceso)"] > 0.0


def test_excess_kurtosis_and_annualized_stats() -> None:
    """Exceso de curtosis Fisher y anualización de media y volatilidad."""
    x = _white_noise(5_000, seed=4)
    assert excess_kurtosis(x)[0] == pytest.approx(0.0, abs=0.1)

    annualized = annualized_stats(x, trading_days=252)
    assert annualized.shape == (2,)
    assert annualized[0] == pytest.approx(x.mean() * 252)
    assert annualized[1] == pytest.approx(x.std(ddof=1) * np.sqrt(252))


def test_fit_return_distribution_recovers_gaussian_parameters() -> None:
    """El ajuste normal reproduce mu y sigma; la t tiene muchos g.l."""
    rng = np.random.default_rng(5)
    x = 0.5 + 2.0 * rng.standard_normal(20_000)
    params = fit_return_distribution(x)

    # La normal se ajusta por MLE sobre una gaussiana: mu y sigma coinciden con
    # los momentos muestrales.
    assert params["mu"] == pytest.approx(x.mean(), abs=1e-8)
    assert params["sigma"] == pytest.approx(x.std(ddof=1), rel=1e-3)
    # La t de Student ajustada a una gaussiana tiene muchos grados de libertad.
    assert params["df"] > 20.0


def test_fit_squared_return_distributions_recovers_parameters() -> None:
    """Los ajustes recuperan una log-normal y una normal en escala log."""
    rng = np.random.default_rng(10)
    log_squared = 0.25 + 0.7 * rng.standard_normal(20_000)
    returns = np.exp(log_squared / 2.0)
    params = fit_squared_return_distributions(returns)

    assert params["lognorm_shape"] == pytest.approx(0.7, abs=0.02)
    assert params["lognorm_loc"] == 0.0
    assert params["lognorm_scale"] == pytest.approx(np.exp(0.25), rel=0.02)
    assert params["log_mu"] == pytest.approx(0.25, abs=0.02)
    assert params["log_sigma"] == pytest.approx(0.7, abs=0.02)
    assert params["n_positive"] == 20_000


def test_fit_squared_return_distributions_ignores_zero_returns() -> None:
    """Los retornos cero se excluyen porque ``log(0)`` no está definido."""
    params = fit_squared_return_distributions(np.array([0.0, 1.0, -2.0]))

    assert params["n_positive"] == 2


# ---------------------------------------------------------------------------
# Autocorrelación
# ---------------------------------------------------------------------------


def test_sample_acf_white_noise_within_bartlett_band() -> None:
    """En un ruido blanco la ACF de los lags ≥ 1 cae dentro de ±z/√n."""
    n = 20_000
    x = _white_noise(n, seed=6)
    acf = sample_acf(x, max_lags=50)

    assert acf.shape == (51,)
    assert acf[0] == pytest.approx(1.0)

    band = stats.norm.ppf(0.975) / np.sqrt(n)
    outside = int(np.sum(np.abs(acf[1:]) > band))
    # Con 50 lags y un nivel nominal del 5 % se esperan ~2,5 rechazos.
    assert outside <= 12


def test_heteroskedastic_robust_acf_band_never_narrower_than_bartlett() -> None:
    """La banda robusta de Diebold es al menos tan ancha como la de Bartlett."""
    n = 20_000
    x = _white_noise(n, seed=7)
    acf, ci = heteroskedastic_robust_acf(x, max_lags=20, alpha=0.05)

    bartlett = stats.norm.ppf(0.975) / np.sqrt(n)
    assert np.all(ci[1:] >= bartlett - 1e-12)
    assert ci[0] == 0.0
    assert acf[0] == pytest.approx(1.0)


def test_heteroskedastic_robust_acf_detects_arch_in_squared_returns() -> None:
    """En un ARCH(1) la ACF de los retornos al cuadrado es significativa."""
    returns = _arch_returns()
    acf, ci = heteroskedastic_robust_acf(returns**2, max_lags=20, alpha=0.05)

    assert acf[1] > 0.0
    assert acf[1] > ci[1]
    # El decaimiento es lento: los primeros lags siguen siendo significativos.
    assert np.sum(np.abs(acf[1:]) > ci[1:]) >= 3


# ---------------------------------------------------------------------------
# EWMA y regímenes
# ---------------------------------------------------------------------------


def test_ewma_volatility_is_causal() -> None:
    """Corromper la cola no altera el pasado de la volatilidad estimada."""
    n = 500
    returns = pd.Series(
        _white_noise(n, seed=8), index=_business_index(n), name="log_ret"
    )
    volatility = compute_ewma_volatility(returns, lambda_ewma=0.94, trading_days=252)

    corrupted = returns.copy()
    corrupted.iloc[-5:] = corrupted.iloc[-5:] * 100.0
    corrupted_volatility = compute_ewma_volatility(
        corrupted, lambda_ewma=0.94, trading_days=252
    )

    # El pasado es idéntico (bit a bit) y sólo cambia la cola.
    pd.testing.assert_series_equal(
        volatility.iloc[:-5], corrupted_volatility.iloc[:-5], check_names=False
    )
    assert volatility.iloc[-1] != corrupted_volatility.iloc[-1]


def test_ewma_volatility_uses_only_previous_return() -> None:
    """sigma_t depende de x_(t-1): el valor de sigma_(t+1) es el de sigma_t."""
    returns = pd.Series(
        _white_noise(200, seed=9), index=_business_index(200), name="log_ret"
    )
    volatility = compute_ewma_volatility(returns, lambda_ewma=0.94, trading_days=1)

    # Primer valor no definido (no hay x_(t-1)); a partir de ahí la recurrencia
    # es sigma_t² = lambda sigma_(t-1)² + (1-lambda) x_(t-1)².
    assert np.isnan(volatility.iloc[0])
    squared = volatility.pow(2).to_numpy()
    lagged = returns.pow(2).shift(1).to_numpy()
    for t in range(2, len(returns)):
        expected = 0.94 * squared[t - 1] + 0.06 * lagged[t]
        assert squared[t] == pytest.approx(expected, rel=1e-9)


def test_high_volatility_mask_fraction_and_index_alignment() -> None:
    """La máscara cubre ~1−cuantil y se alinea con un índice mayor."""
    n = 1_000
    rng = np.random.default_rng(10)
    volatility = pd.Series(
        rng.uniform(0.1, 0.5, n), index=_business_index(n), name="volatility"
    )

    mask = high_volatility_mask(volatility, quantile=0.90)
    assert mask.mean() == pytest.approx(0.10, abs=0.02)

    extended = _business_index(n + 2)
    extended_mask = high_volatility_mask(volatility, quantile=0.90, index=extended)
    assert len(extended_mask) == n + 2
    assert not extended_mask.iloc[-2:].any()


# ---------------------------------------------------------------------------
# Bootstrap y bandas incondicionales
# ---------------------------------------------------------------------------


def test_block_bootstrap_ci_shapes_and_reproducibility() -> None:
    """El bootstrap devuelve (est, ci) con la forma esperada y es reproducible."""
    x = _white_noise(400, seed=11)

    estimate, interval = block_bootstrap_ci(
        x, excess_kurtosis, block_size=10, reps=200, alpha=0.05, seed=SEED
    )
    assert estimate.shape == (1,)
    assert interval.shape == (2, 1)
    assert interval[0, 0] < interval[1, 0]

    estimate_2, interval_2 = block_bootstrap_ci(
        x, excess_kurtosis, block_size=10, reps=200, alpha=0.05, seed=SEED
    )
    np.testing.assert_allclose(estimate, estimate_2)
    np.testing.assert_allclose(interval, interval_2)


def test_block_bootstrap_ci_rejects_invalid_arguments() -> None:
    """Los parámetros inválidos fallan rápido."""
    x = _white_noise(100, seed=12)
    with pytest.raises(ValueError):
        block_bootstrap_ci(x, excess_kurtosis, block_size=0, reps=10)
    with pytest.raises(ValueError):
        block_bootstrap_ci(x, excess_kurtosis, block_size=10, reps=0)
    with pytest.raises(ValueError):
        block_bootstrap_ci(x, excess_kurtosis, block_size=10, reps=10, alpha=0.0)


def test_bootstrap_ci_shapes_and_reproducibility() -> None:
    """El bootstrap i.i.d. devuelve (est, ci) con la forma esperada."""
    x = _white_noise(400, seed=11)

    estimate, interval = bootstrap_ci(x, excess_kurtosis, reps=200, alpha=0.05, seed=SEED)
    assert estimate.shape == (1,)
    assert interval.shape == (2, 1)
    assert interval[0, 0] < interval[1, 0]

    estimate_2, interval_2 = bootstrap_ci(x, excess_kurtosis, reps=200, alpha=0.05, seed=SEED)
    np.testing.assert_allclose(estimate, estimate_2)
    np.testing.assert_allclose(interval, interval_2)


def test_bootstrap_ci_rejects_invalid_arguments() -> None:
    """Los parámetros inválidos fallan rápido."""
    x = _white_noise(100, seed=12)
    with pytest.raises(ValueError):
        bootstrap_ci(x, excess_kurtosis, reps=0)
    with pytest.raises(ValueError):
        bootstrap_ci(x, excess_kurtosis, reps=10, alpha=0.0)


def test_bootstrap_acf_shapes_and_reproducibility() -> None:
    """La ACF con block bootstrap devuelve (acf, banda) con la forma esperada."""
    x = _white_noise(600, seed=15)

    acf, ci = bootstrap_acf(x, max_lags=20, block_size=10, reps=200, alpha=0.05, seed=SEED)
    assert acf.shape == (21,)
    assert ci.shape == (21,)
    assert acf[0] == pytest.approx(1.0)
    assert ci[0] == 0.0
    assert np.all(ci[1:] >= 0.0)

    acf_2, ci_2 = bootstrap_acf(x, max_lags=20, block_size=10, reps=200, alpha=0.05, seed=SEED)
    np.testing.assert_allclose(acf, acf_2)
    np.testing.assert_allclose(ci, ci_2)


def test_bootstrap_acf_white_noise_within_band() -> None:
    """En ruido blanco casi ningún lag queda fuera de la banda por bloques."""
    x = _white_noise(1_000, seed=16)
    acf, ci = bootstrap_acf(x, max_lags=20, block_size=10, reps=300, alpha=0.05, seed=SEED)

    outside = int(np.sum(np.abs(acf[1:]) > ci[1:]))
    # Banda nominal del 5 %: se esperan ~1 lag fuera; el umbral es holgado.
    assert outside <= 8


def test_bootstrap_acf_block_band_widens_with_persistence() -> None:
    """El bootstrap por bloques conserva la dependencia: la banda se ensancha."""
    n = 1_000
    rng = np.random.default_rng(23)
    persistent = np.empty(n)
    persistent[0] = rng.standard_normal()
    for t in range(1, n):
        persistent[t] = 0.9 * persistent[t - 1] + rng.standard_normal()
    white = _white_noise(n, seed=24)

    kwargs = {"max_lags": 10, "block_size": 20, "reps": 400, "alpha": 0.05, "seed": SEED}
    acf_white, ci_white = bootstrap_acf(white, **kwargs)
    acf_persist, ci_persist = bootstrap_acf(persistent, **kwargs)

    assert acf_white[1] == pytest.approx(0.0, abs=0.1)
    assert acf_persist[1] > 0.7
    # Con dependencia, la varianza del estimador es mayor que la de ruido blanco.
    assert ci_persist[1] > ci_white[1]


def test_hac_acf_white_noise_matches_bartlett() -> None:
    """En ruido blanco el error HAC coincide (aprox.) con 1/sqrt(T)."""
    n = 5_000
    x = _white_noise(n, seed=31)
    acf, hw = hac_acf(x, max_lags=20, alpha=0.05)
    assert acf.shape == hw.shape == (21,)
    assert hw[0] == 0.0
    bartlett = 1.959964 / np.sqrt(n)
    np.testing.assert_allclose(hw[1:], bartlett, rtol=0.3)


def test_hac_acf_ar1_standard_error_matches_theory() -> None:
    """En AR(1) gaussiano, se(rho_1) ~ sqrt((1 - phi^2)/T)."""
    n, phi = 20_000, 0.5
    rng = np.random.default_rng(32)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    acf, hw = hac_acf(x, max_lags=5, alpha=0.05)
    assert acf[1] == pytest.approx(phi, abs=0.05)
    assert hw[1] == pytest.approx(1.959964 * np.sqrt((1 - phi**2) / n), rel=0.25)


def test_hac_acf_arch_squared_band_wider_than_bartlett() -> None:
    """Con ARCH, la banda HAC de x² supera a la de Bartlett."""
    n = 8_000
    rng = np.random.default_rng(33)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = np.sqrt(0.2 + 0.5 * x[t - 1] ** 2) * eps[t]
    _, hw = hac_acf(x**2, max_lags=10, alpha=0.05)
    assert np.mean(hw[1:4]) > 1.959964 / np.sqrt(n)


def test_hac_acf_rejects_invalid_args() -> None:
    x = _white_noise(100, seed=34)
    with pytest.raises(ValueError):
        hac_acf(x, max_lags=100)
    with pytest.raises(ValueError):
        hac_acf(x, max_lags=10, alpha=0.0)


def test_bootstrap_acf_rejects_invalid_args() -> None:
    """Los parámetros inválidos fallan rápido."""
    x = _white_noise(100, seed=17)
    with pytest.raises(ValueError):
        bootstrap_acf(x, max_lags=100, reps=10)
    with pytest.raises(ValueError):
        bootstrap_acf(x, max_lags=10, reps=0)
    with pytest.raises(ValueError):
        bootstrap_acf(x, max_lags=10, reps=10, alpha=0.0)
    with pytest.raises(ValueError):
        bootstrap_acf(x, max_lags=10, block_size=0, reps=10)


def test_unconditional_band_excursions_heavy_tails() -> None:
    """Con colas pesadas hay más excursiones que las gaussianas predicen."""
    rng = np.random.default_rng(13)
    x = rng.standard_normal(20_000) + 0.5 * rng.standard_t(df=3, size=20_000)
    excursions = unconditional_band_excursions(x, ks=(1.0, 2.0, 3.0, 5.0))

    assert list(excursions.columns) == ["k", "empirical", "gaussian", "ratio"]
    assert len(excursions) == 4
    row = excursions.set_index("k").loc[3.0]
    assert row["empirical"] > row["gaussian"]
    assert row["ratio"] > 1.0


def test_build_summary_table_shape_and_content() -> None:
    """La tabla resumen tiene las filas esperadas y las columnas del activo."""
    x = _white_noise(1_000, seed=14)
    summary = build_summary_table(x, ticker="SPY", alpha=0.05, max_lags=10)

    assert list(summary.columns) == [
        "Métrica",
        "SPY (observado)",
        "Bajo normalidad i.i.d.",
    ]
    assert len(summary) == 10
    assert summary.iloc[0, 0] == "Observaciones"
    assert summary.iloc[0, 1] == "1,000"


def test_module_has_no_matplotlib_or_network_dependencies() -> None:
    """``thesis.characterization`` no dibuja ni descarga nada."""
    source = Path(characterization.__file__).read_text(encoding="utf-8")
    assert "matplotlib" not in source
    assert "yfinance" not in source
    assert "plt." not in source
