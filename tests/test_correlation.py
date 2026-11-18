"""Tests para ``thesis.correlation``.

Los tests son **sintéticos**: no requieren red ni datos reales.  Se comprueban
las propiedades del estimador de correlación momentum–retorno futuro con
ventana simétrica (``τ = 2h``) y su banda nula por bootstrap i.i.d.:

1. la firma del estimador es reproducible con semilla fija;
2. en ruido blanco i.i.d. la curva observada cae dentro de la banda nula y la
   banda está centrada en ≈ 0;
3. los parámetros inválidos fallan rápido;
4. el estimador coincide con la definición explícita (``rolling`` + ``shift``)
   y usa ventanas disjuntas (sin look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.correlation import bootstrap_momentum_forward_correlation

#: Semilla fija: los tests son reproducibles.
SEED = 42


def _white_noise(n: int, seed: int = SEED) -> pd.Series:
    """Ruido blanco gaussiano de ``n`` observaciones como serie de retornos."""
    values = np.random.default_rng(seed).standard_normal(n)
    return pd.Series(values)


# ---------------------------------------------------------------------------
# Firma y reproducibilidad
# ---------------------------------------------------------------------------


def test_shapes_and_reproducibility() -> None:
    """Las tres salidas tienen la forma de la grilla y son reproducibles."""
    log_ret = _white_noise(1_500, seed=1)
    half_windows = [5, 10, 20, 40]

    point, ci_low, ci_high = bootstrap_momentum_forward_correlation(
        log_ret, half_windows, n_bootstrap=200, alpha=0.05, seed=SEED,
    )

    assert point.shape == (len(half_windows),)
    assert ci_low.shape == (len(half_windows),)
    assert ci_high.shape == (len(half_windows),)
    assert np.all(ci_low <= ci_high)
    assert np.all(np.isfinite(point))

    point_2, ci_low_2, ci_high_2 = bootstrap_momentum_forward_correlation(
        log_ret, half_windows, n_bootstrap=200, alpha=0.05, seed=SEED,
    )
    np.testing.assert_allclose(point, point_2)
    np.testing.assert_allclose(ci_low, ci_low_2)
    np.testing.assert_allclose(ci_high, ci_high_2)


def test_matches_pandas_rolling_definition() -> None:
    """El estimador coincide con ``rolling(h).sum().shift(±h)`` (sin look-ahead)."""
    log_ret = _white_noise(800, seed=2)
    h = 25
    r = log_ret.to_numpy()

    rolling = pd.Series(r).rolling(h).sum()
    momentum = rolling.shift(1).to_numpy()
    forward = rolling.shift(-h).to_numpy()
    valid = ~(np.isnan(momentum) | np.isnan(forward))
    expected = np.corrcoef(momentum[valid], forward[valid])[0, 1]

    point, _, _ = bootstrap_momentum_forward_correlation(
        log_ret, [h], n_bootstrap=1, seed=SEED,
    )
    assert point[0] == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Banda nula i.i.d.
# ---------------------------------------------------------------------------


def test_white_noise_null_centered() -> None:
    """En ruido blanco la curva observada cae dentro de la banda nula."""
    log_ret = _white_noise(2_000, seed=3)
    half_windows = [5, 10, 20, 40]

    point, ci_low, ci_high = bootstrap_momentum_forward_correlation(
        log_ret, half_windows, n_bootstrap=300, alpha=0.05, seed=SEED,
    )

    inside = (point >= ci_low) & (point <= ci_high)
    assert inside.mean() >= 0.5

    # La banda i.i.d. es la distribución nula del estimador: centrada en 0 y
    # con un ancho que escala como ~1.96 * sqrt(h / n) (las ventanas
    # solapadas reducen el tamaño efectivo de la muestra).
    assert np.all(ci_low < 0.0)
    assert np.all(ci_high > 0.0)
    half_width = np.maximum(np.abs(ci_low), np.abs(ci_high))
    expected = 1.959964 * np.sqrt(np.asarray(half_windows, dtype=float) / len(log_ret))
    assert np.all(half_width < 1.6 * expected)
    assert np.all(half_width > 0.4 * expected)


def test_null_band_is_independent_of_temporal_dependence() -> None:
    """La banda i.i.d. no se ensancha con la persistencia de la serie."""
    n = 1_500
    rng = np.random.default_rng(4)
    persistent = np.empty(n)
    persistent[0] = rng.standard_normal()
    for t in range(1, n):
        persistent[t] = 0.9 * persistent[t - 1] + rng.standard_normal()

    _, ci_low_white, ci_high_white = bootstrap_momentum_forward_correlation(
        _white_noise(n, seed=5), [10], n_bootstrap=300, seed=SEED,
    )
    point_p, ci_low_p, ci_high_p = bootstrap_momentum_forward_correlation(
        pd.Series(persistent), [10], n_bootstrap=300, seed=SEED,
    )

    # La serie persistente tiene una correlación puntual claramente positiva...
    assert point_p[0] > 0.3
    # ... pero la banda nula i.i.d. es (casi) la misma que en ruido blanco:
    # el remuestreo i.i.d. rompe la dependencia temporal.
    assert ci_low_p[0] == pytest.approx(ci_low_white[0], rel=0.35)
    assert ci_high_p[0] == pytest.approx(ci_high_white[0], rel=0.35)


# ---------------------------------------------------------------------------
# Validación de argumentos
# ---------------------------------------------------------------------------


def test_invalid_args() -> None:
    """Los parámetros inválidos fallan rápido."""
    log_ret = _white_noise(200, seed=6)

    with pytest.raises(ValueError):
        bootstrap_momentum_forward_correlation(log_ret, [], n_bootstrap=10)
    with pytest.raises(ValueError):
        bootstrap_momentum_forward_correlation(log_ret, [0], n_bootstrap=10)
    with pytest.raises(ValueError):
        bootstrap_momentum_forward_correlation(log_ret, [-5], n_bootstrap=10)
    with pytest.raises(ValueError):
        bootstrap_momentum_forward_correlation(log_ret, [5], n_bootstrap=0)
    with pytest.raises(ValueError):
        bootstrap_momentum_forward_correlation(log_ret, [5], n_bootstrap=10, alpha=0.0)
    with pytest.raises(ValueError):
        bootstrap_momentum_forward_correlation(log_ret, [5], n_bootstrap=10, alpha=1.0)


def test_nan_input_is_rejected() -> None:
    """Una serie con NaN no es admisible."""
    log_ret = _white_noise(100, seed=7)
    log_ret.iloc[10] = np.nan
    with pytest.raises(ValueError):
        bootstrap_momentum_forward_correlation(log_ret, [5], n_bootstrap=10)
