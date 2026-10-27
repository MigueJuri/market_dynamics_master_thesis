"""Pruebas del diagnóstico y la validación de backtests de pair trading.

El foco está en tres aspectos:

1. La corrección del bug del hedge ratio efectivo: cada operación debe usar
   su propio beta, no el de la última entrada.
2. Las métricas de diagnóstico deben ser coherentes con el DataFrame de
   resultados del backtest.
3. La batería de comprobaciones automáticas debe detectar los errores
   habituales (look-ahead, desalineación, turnover inconsistente, etc.).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from thesis.backtest_diagnostics import (
    PairBacktestDiagnostics,
    ValidationReport,
    compute_hedge_diagnostics,
    compute_pair_backtest_diagnostics,
    compute_return_diagnostics,
    compute_risk_metrics,
    compute_signal_diagnostics,
    detect_rebalance_mode,
    extract_pair_trades,
    validate_pair_backtest,
    verify_hedge_is_causal,
)
from thesis.pair_trading import (
    pair_trading_backtest,
    simulate_ou_process,
    summarize_pair_window,
)
from thesis.plotting import (
    plot_backtest_dashboard,
    plot_equity_diagnostics,
    plot_hedge_diagnostics,
    plot_pair_window_summary,
    plot_return_diagnostics,
    plot_signal_diagnostics,
)

HEDGE_WINDOW = 120
Z_WINDOW = 40
ENTRY_Z = 2.0
EXIT_Z = 0.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _cerrar_figuras():
    """Cierra las figuras abiertas al terminar cada prueba."""
    yield
    plt.close("all")


@pytest.fixture
def log_prices_pair() -> pd.DataFrame:
    """Par cointegrado sintético con dos shocks que generan señales."""
    n = 400
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    log_b = np.cumsum(rng.normal(0.0, 0.01, size=n))
    spread = simulate_ou_process(
        theta=12.0, mu=0.0, sigma=0.15, x0=0.0, T=n / 252, dt=1 / 252, seed=0
    )
    spread[150:190] += np.linspace(0.0, 0.30, 40)
    spread[250:290] -= np.linspace(0.0, 0.30, 40)
    log_a = 0.5 + 0.8 * log_b + spread
    return pd.DataFrame({"A": log_a, "B": log_b}, index=idx)


@pytest.fixture
def backtest(log_prices_pair):
    """Backtest walk-forward sobre el par sintético."""
    return pair_trading_backtest(
        log_prices_pair,
        "A",
        "B",
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        cost=0.001,
    )


# =============================================================================
# Corrección del hedge ratio efectivo
# =============================================================================


def test_beta_efectivo_es_cero_fuera_de_posicion(backtest) -> None:
    """El beta efectivo debe ser 0 siempre que la estrategia esté en cash."""
    results = backtest.results
    flat = results["position"] == 0
    assert results.loc[flat, "effective_beta"].eq(0).all()


def test_beta_efectivo_constante_dentro_de_cada_operacion(backtest) -> None:
    """Dentro de una operación el beta efectivo no puede cambiar."""
    results = backtest.results
    in_position = results[results["position"] != 0]
    assert not in_position.empty
    blocks = (
        in_position["effective_beta"] != in_position["effective_beta"].shift()
    ).cumsum()
    for _, block in in_position.groupby(blocks):
        assert block["effective_beta"].nunique() == 1


def test_beta_efectivo_difiere_entre_operaciones(backtest) -> None:
    """Cada operación debe usar su propio beta (regresión del bug)."""
    results = backtest.results
    in_position = results[results["position"] != 0]
    blocks = (
        in_position["effective_beta"] != in_position["effective_beta"].shift()
    ).cumsum()
    betas = [
        block["effective_beta"].iloc[0] for _, block in in_position.groupby(blocks)
    ]
    assert len(betas) > 1
    assert len({round(float(b), 10) for b in betas}) > 1


def test_beta_diario_no_es_constante(backtest) -> None:
    """La estimación walk-forward del beta debe variar día a día."""
    beta = backtest.results["beta"].dropna()
    assert beta.nunique() > 10


def test_beta_aplicado_retrasa_una_sesion(backtest) -> None:
    """El beta aplicado en t es el beta efectivo de t-1."""
    results = backtest.results
    esperado = results["effective_beta"].shift(1).fillna(0.0)
    assert np.allclose(esperado, results["held_beta"])


def test_posicion_aplicada_retrasa_una_sesion(backtest) -> None:
    """La posición aplicada en t es la posición señalada en t-1."""
    results = backtest.results
    esperado = results["position"].shift(1).fillna(0.0)
    assert np.allclose(esperado, results["held_position"])


def test_beta_diario_igual_al_beta_efectivo_en_modo_daily(log_prices_pair) -> None:
    """En modo diario el beta efectivo coincide con la estimación diaria."""
    bt = pair_trading_backtest(
        log_prices_pair,
        "A",
        "B",
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        cost=0.0,
        rebalance_hedge="daily",
    )
    results = bt.results
    mask = (results["position"] != 0) & results["beta"].notna()
    assert np.allclose(
        results.loc[mask, "effective_beta"], results.loc[mask, "beta"], atol=1e-12
    )


def test_modo_daily_genera_mas_turnover(log_prices_pair) -> None:
    """Rebalancear el beta cada día debe generar más turnover."""
    kwargs = {
        "hedge_window": HEDGE_WINDOW,
        "z_window": Z_WINDOW,
        "entry_z": ENTRY_Z,
        "exit_z": EXIT_Z,
        "cost": 0.0,
    }
    bt_entry = pair_trading_backtest(
        log_prices_pair, "A", "B", rebalance_hedge="on_entry", **kwargs
    )
    bt_daily = pair_trading_backtest(
        log_prices_pair, "A", "B", rebalance_hedge="daily", **kwargs
    )
    assert bt_daily.results["turnover"].sum() > bt_entry.results["turnover"].sum()


def test_el_backtest_guarda_el_modo_de_rebalanceo(log_prices_pair) -> None:
    """La columna ``rebalance_hedge`` permite deducir el modo utilizado."""
    bt = pair_trading_backtest(
        log_prices_pair,
        "A",
        "B",
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        cost=0.0,
        rebalance_hedge="daily",
    )
    assert detect_rebalance_mode(bt.results) == "daily"


def test_la_validacion_se_adapta_al_modo_daily(log_prices_pair) -> None:
    """En modo diario la validación no debe exigir beta constante."""
    bt = pair_trading_backtest(
        log_prices_pair,
        "A",
        "B",
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        cost=0.0,
        rebalance_hedge="daily",
    )
    report = validate_pair_backtest(
        bt.results, entry_z=ENTRY_Z, exit_z=EXIT_Z, hedge_window=HEDGE_WINDOW
    )
    assert report.passed, report.as_dataframe().to_string()
    nombres = {check.name for check in report.checks}
    assert "beta_efectivo_igual_al_diario" in nombres
    assert "beta_efectivo_constante_en_bloque" not in nombres


# =============================================================================
# Extracción de operaciones
# =============================================================================


def test_extract_pair_trades_empareja_entradas_y_salidas(backtest) -> None:
    """Cada entrada tiene su salida y las duraciones son coherentes."""
    trades = extract_pair_trades(backtest.results)
    assert not trades.empty
    assert (trades["entry_date"] <= trades["exit_date"]).all()
    assert (trades["days"] >= 1).all()
    assert trades["direction"].isin([-1.0, 1.0]).all()
    assert np.allclose(
        trades["net_return"], trades["gross_return"] - trades["costs"], atol=1e-12
    )


def test_extract_pair_trades_vacia_sin_operaciones(log_prices_pair) -> None:
    """Sin umbrales alcanzados no hay operaciones que extraer."""
    bt = pair_trading_backtest(
        log_prices_pair,
        "A",
        "B",
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        entry_z=100.0,
        exit_z=0.0,
        cost=0.0,
    )
    trades = extract_pair_trades(bt.results)
    assert trades.empty


def test_extract_pair_trades_marca_operacion_abierta(backtest) -> None:
    """Una operación sin salida se marca con ``is_open=True``."""
    trades = extract_pair_trades(backtest.results)
    if trades["is_open"].any():
        abiertas = trades[trades["is_open"]]
        assert (abiertas["exit_reason"] == "end").all()


# =============================================================================
# Métricas por bloque
# =============================================================================


def test_hedge_diagnostics_detecta_beta_reutilizado(backtest) -> None:
    """El diagnóstico marca si todas las operaciones comparten un beta."""
    diag = compute_hedge_diagnostics(backtest.results)
    assert diag.effective_beta_unique > 1
    assert not diag.effective_beta_constant
    assert diag.effective_beta_block_consistent
    assert np.isfinite(diag.beta_mean)
    assert diag.beta_min <= diag.beta_median <= diag.beta_max


def test_hedge_diagnostics_cuenta_betas_extremos(backtest) -> None:
    """``max_abs_beta`` permite contar los días con hedge ratio inestable."""
    diag = compute_hedge_diagnostics(backtest.results, max_abs_beta=1.0)
    assert diag.beta_out_of_bounds >= 0


def test_signal_diagnostics_cuenta_entradas_y_salidas(backtest) -> None:
    """El número de entradas coincide con los cambios de posición."""
    diag = compute_signal_diagnostics(
        backtest.results,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        hedge_window=HEDGE_WINDOW,
    )
    results = backtest.results
    position = results["position"]
    entradas = int(((position != 0) & (position.shift(1).fillna(0) == 0)).sum())
    assert diag.entries == entradas
    assert diag.exits >= 0
    assert diag.long_entries + diag.short_entries == diag.entries
    assert 0.0 <= diag.flat_fraction <= 1.0
    assert diag.entry_threshold_ok
    assert diag.exit_threshold_ok


def test_return_diagnostics_cuadra_contribuciones(backtest) -> None:
    """La contribución de las dos patas debe sumar el retorno neto."""
    diag = compute_return_diagnostics(backtest.results)
    assert diag.contribution_max_error < 1e-12
    assert diag.total_turnover >= 0.0
    assert diag.total_costs >= 0.0
    assert np.isclose(diag.gross_total - diag.total_costs, diag.net_total, atol=1e-9)


def test_return_diagnostics_turnover_por_operacion(backtest) -> None:
    """El turnover medio por operación es consistente con el total."""
    diag = compute_return_diagnostics(backtest.results)
    assert diag.turnover_per_operation >= 0.0


def test_risk_metrics_coherentes(backtest) -> None:
    """Las métricas de riesgo cuadran con la curva de equity."""
    trades = extract_pair_trades(backtest.results)
    risk = compute_risk_metrics(backtest.results, backtest.equity, trades)
    assert risk.max_drawdown <= 0.0
    assert risk.volatility >= 0.0
    assert 0.0 <= risk.win_rate <= 1.0
    assert risk.best_trade >= risk.worst_trade
    assert risk.n_trades == len(trades[~trades["is_open"]])


# =============================================================================
# Comprobaciones automáticas
# =============================================================================


def test_validacion_pasa_en_un_backtest_correcto(backtest) -> None:
    """Todas las comprobaciones internas deben pasar."""
    report = validate_pair_backtest(
        backtest.results,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        hedge_window=HEDGE_WINDOW,
    )
    assert report.passed, report.as_dataframe().to_string()
    assert report.failures == ()
    assert "comprobaciones OK" in report.summary()


def test_validacion_detecta_turnover_incoherente(backtest) -> None:
    """Si el turnover no cuadra con los pesos, la comprobación falla."""
    results = backtest.results.copy()
    results.loc[results.index[200], "turnover"] += 1.0
    report = validate_pair_backtest(
        results, entry_z=ENTRY_Z, exit_z=EXIT_Z, hedge_window=HEDGE_WINDOW
    )
    assert not report.passed
    fallidas = {check.name for check in report.failures}
    assert "turnover_coherente_con_pesos" in fallidas
    assert "costes_coherentes_con_turnover" in fallidas


def test_validacion_detecta_posicion_desalineada(backtest) -> None:
    """Si la posición aplicada no lleva un día de retraso, falla."""
    results = backtest.results.copy()
    results["held_position"] = results["position"]
    report = validate_pair_backtest(
        results, entry_z=ENTRY_Z, exit_z=EXIT_Z, hedge_window=HEDGE_WINDOW
    )
    assert not report.passed
    fallidas = {check.name for check in report.failures}
    assert "posicion_aplicada_con_lag" in fallidas


def test_validacion_detecta_beta_efectivo_incoherente(backtest) -> None:
    """Un beta efectivo no nulo en cash debe detectarse."""
    results = backtest.results.copy()
    results.loc[results["position"] == 0, "effective_beta"] = 1.23
    report = validate_pair_backtest(
        results, entry_z=ENTRY_Z, exit_z=EXIT_Z, hedge_window=HEDGE_WINDOW
    )
    assert not report.passed
    fallidas = {check.name for check in report.failures}
    assert "beta_efectivo_cero_en_cash" in fallidas


def test_validacion_detecta_calculo_incoherente_de_retorno(backtest) -> None:
    """Si el retorno neto no cuadra con bruto menos costes, falla."""
    results = backtest.results.copy()
    results.loc[results.index[300], "strategy_ret"] += 0.5
    report = validate_pair_backtest(
        results, entry_z=ENTRY_Z, exit_z=EXIT_Z, hedge_window=HEDGE_WINDOW
    )
    assert not report.passed
    fallidas = {check.name for check in report.failures}
    assert "bruto_menos_coste_es_neto" in fallidas


def test_validacion_detecta_operaciones_durante_el_warmup(backtest) -> None:
    """Una posición antes del warm-up debe invalidar el backtest."""
    results = backtest.results.copy()
    results.iloc[5, results.columns.get_loc("position")] = 1.0
    report = validate_pair_backtest(
        results, entry_z=ENTRY_Z, exit_z=EXIT_Z, hedge_window=HEDGE_WINDOW
    )
    assert not report.passed
    fallidas = {check.name for check in report.failures}
    assert "warmup_respetado" in fallidas


def test_validacion_detecta_umbral_de_entrada_incorrecto(backtest) -> None:
    """Una entrada por debajo del umbral debe invalidar el backtest."""
    results = backtest.results.copy()
    position = results["position"]
    entrada = int(((position != 0) & (position.shift(1).fillna(0) == 0)).sum())
    if entrada == 0:
        pytest.skip("El backtest no generó entradas")
    results.loc[position.ne(0) & position.shift(1).fillna(0).eq(0), "z_score"] = 0.0
    report = validate_pair_backtest(
        results, entry_z=ENTRY_Z, exit_z=EXIT_Z, hedge_window=HEDGE_WINDOW
    )
    assert not report.passed
    fallidas = {check.name for check in report.failures}
    assert "entradas_superan_umbral" in fallidas


def test_verify_hedge_is_causal_confirma_el_estimador(
    backtest, log_prices_pair
) -> None:
    """Recalcular el hedge con datos pasados debe reproducir el almacenado."""
    report = verify_hedge_is_causal(
        log_prices_pair,
        "A",
        "B",
        backtest.results,
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        sample_size=8,
    )
    assert report.passed, report.as_dataframe().to_string()
    nombres = {check.name for check in report.checks}
    assert {"causalidad_alpha", "causalidad_beta", "causalidad_z_score"} <= nombres


def test_verify_hedge_is_causal_detecta_lookahead(backtest, log_prices_pair) -> None:
    """Si el beta almacenado usa datos futuros, la comprobación falla."""
    results = backtest.results.copy()
    results["beta"] = results["beta"] + 10.0
    report = verify_hedge_is_causal(
        log_prices_pair,
        "A",
        "B",
        results,
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        sample_size=8,
    )
    assert not report.passed
    fallidas = {check.name for check in report.failures}
    assert "causalidad_beta" in fallidas


# =============================================================================
# Agregación
# =============================================================================


def test_compute_pair_backtest_diagnostics_agrega_todo(
    backtest, log_prices_pair
) -> None:
    """La función principal devuelve todos los bloques y el informe."""
    diag = compute_pair_backtest_diagnostics(
        backtest.results,
        backtest.equity,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        log_prices=log_prices_pair,
        ticker_a="A",
        ticker_b="B",
        max_abs_beta=5.0,
    )
    assert isinstance(diag, PairBacktestDiagnostics)
    assert isinstance(diag.validation, ValidationReport)
    assert diag.validation.passed, diag.validation.as_dataframe().to_string()
    assert isinstance(diag.trades, pd.DataFrame)
    assert diag.hedge.effective_beta_unique >= 1
    assert diag.signal.entries >= 0
    assert diag.risk.n_trades >= 0


def test_compute_pair_backtest_diagnostics_sin_precios_no_verifica_causalidad(
    backtest,
) -> None:
    """Sin log-precios la comprobación de causalidad se omite."""
    diag = compute_pair_backtest_diagnostics(
        backtest.results,
        backtest.equity,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        hedge_window=HEDGE_WINDOW,
        z_window=Z_WINDOW,
        log_prices=None,
    )
    nombres = {check.name for check in diag.validation.checks}
    assert "causalidad_beta" not in nombres


# =============================================================================
# Gráficas de diagnóstico
# =============================================================================


def test_plot_hedge_diagnostics_genera_figura(backtest) -> None:
    """La gráfica del hedge ratio se genera sin errores."""
    plot_hedge_diagnostics(backtest.results, in_sample_beta=0.8)
    assert plt.get_fignums()


def test_plot_signal_diagnostics_genera_figura(backtest) -> None:
    """La gráfica de señales se genera sin errores."""
    plot_signal_diagnostics(backtest.results, entry_z=ENTRY_Z, exit_z=EXIT_Z)
    assert plt.get_fignums()


def test_plot_return_diagnostics_genera_figura(backtest) -> None:
    """La gráfica de retornos se genera sin errores."""
    plot_return_diagnostics(backtest.results)
    assert plt.get_fignums()


def test_plot_equity_diagnostics_genera_figura(backtest) -> None:
    """La gráfica de equity se genera sin errores."""
    plot_equity_diagnostics(backtest.results)
    assert plt.get_fignums()


def test_plot_backtest_dashboard_genera_figura(backtest) -> None:
    """El dashboard completo se genera sin errores."""
    plot_backtest_dashboard(
        backtest.results, entry_z=ENTRY_Z, exit_z=EXIT_Z, in_sample_beta=0.8
    )
    assert plt.get_fignums()


def test_plot_diagnosticos_con_resultados_vacios_no_falla() -> None:
    """Un DataFrame vacío no debe romper las funciones de graficado."""
    vacio = pd.DataFrame()
    plot_hedge_diagnostics(vacio)
    plot_signal_diagnostics(vacio)
    plot_return_diagnostics(vacio)
    plot_equity_diagnostics(vacio)
    plot_backtest_dashboard(vacio)


# =============================================================================
# Resumen de la ventana reciente
# =============================================================================


def test_summarize_pair_window_usa_solo_el_final_de_la_muestra(
    log_prices_pair,
) -> None:
    """El hedge de la ventana reciente se calcula sólo con las últimas filas."""
    summary = summarize_pair_window(
        log_prices_pair, "A", "B", window=HEDGE_WINDOW, min_obs=30
    )

    assert summary.n_obs == HEDGE_WINDOW
    assert summary.start == log_prices_pair.index[-HEDGE_WINDOW]
    assert summary.end == log_prices_pair.index[-1]
    assert len(summary.spread) == HEDGE_WINDOW


def test_summarize_pair_window_cambia_si_cambia_el_final(log_prices_pair) -> None:
    """Modificar sólo la parte final debe alterar el hedge de la ventana."""
    resumen_inicial = summarize_pair_window(
        log_prices_pair, "A", "B", window=HEDGE_WINDOW
    )

    alterado = log_prices_pair.copy()
    alterado.iloc[-1, alterado.columns.get_loc("A")] += 0.5
    resumen_alterado = summarize_pair_window(alterado, "A", "B", window=HEDGE_WINDOW)

    assert resumen_alterado.hedge.beta != resumen_inicial.hedge.beta


def test_summarize_pair_window_ignora_el_pasado_lejano(log_prices_pair) -> None:
    """Alterar el inicio de la muestra no altera la ventana reciente."""
    resumen_inicial = summarize_pair_window(
        log_prices_pair, "A", "B", window=HEDGE_WINDOW
    )

    alterado = log_prices_pair.copy()
    alterado.iloc[0, alterado.columns.get_loc("A")] += 5.0
    resumen_alterado = summarize_pair_window(alterado, "A", "B", window=HEDGE_WINDOW)

    assert resumen_alterado.hedge.beta == pytest.approx(
        resumen_inicial.hedge.beta, rel=1e-12
    )


def test_summarize_pair_window_compara_con_la_muestra_completa(
    log_prices_pair,
) -> None:
    """La comparación incluye el hedge completo y sus diferencias."""
    summary = summarize_pair_window(
        log_prices_pair, "A", "B", window=HEDGE_WINDOW, compare_full=True
    )

    assert summary.full_hedge is not None
    assert summary.full_adf is not None
    assert summary.beta_change == pytest.approx(
        summary.hedge.beta - summary.full_hedge.beta
    )
    assert summary.adf_pvalue_change == pytest.approx(
        summary.adf.pvalue - summary.full_adf.pvalue
    )
    assert summary.stationarity_lost is not None


def test_summarize_pair_window_tabla_de_comparacion(log_prices_pair) -> None:
    """La tabla de comparación tiene las filas esperadas."""
    summary = summarize_pair_window(log_prices_pair, "A", "B", window=HEDGE_WINDOW)
    tabla = summary.comparison_table()

    assert list(tabla.columns) == ["Muestra completa", "Última ventana"]
    assert "Beta" in tabla.index
    assert "ADF p-value" in tabla.index
    assert "Estacionario" in tabla.index
    assert tabla.loc["Beta", "Última ventana"] == pytest.approx(summary.hedge.beta)


def test_summarize_pair_window_incluye_el_half_life(log_prices_pair) -> None:
    """La ventana reciente incluye el ajuste OU con su vida media."""
    summary = summarize_pair_window(log_prices_pair, "A", "B", window=HEDGE_WINDOW)

    assert summary.ou is not None
    assert summary.ou.half_life > 0
    assert summary.ou.theta > 0
    assert summary.ou.sigma_eq > 0
    assert summary.ou_error is None


def test_summarize_pair_window_compara_el_half_life(log_prices_pair) -> None:
    """La comparación de vida media se calcula entre ambas muestras."""
    summary = summarize_pair_window(
        log_prices_pair, "A", "B", window=HEDGE_WINDOW, compare_full=True
    )

    assert summary.full_ou is not None
    assert summary.half_life_change == pytest.approx(
        summary.ou.half_life - summary.full_ou.half_life
    )
    assert summary.half_life_change_pct is not None


def test_summarize_pair_window_tabla_incluye_el_half_life(log_prices_pair) -> None:
    """La tabla de comparación incluye la vida media de ambas muestras."""
    summary = summarize_pair_window(log_prices_pair, "A", "B", window=HEDGE_WINDOW)
    tabla = summary.comparison_table()

    assert "Half-Life (sesiones)" in tabla.index
    assert "Theta (OU)" in tabla.index
    assert tabla.loc["Half-Life (sesiones)", "Última ventana"] == pytest.approx(
        summary.ou.half_life
    )
    assert tabla.loc["Half-Life (sesiones)", "Muestra completa"] == pytest.approx(
        summary.full_ou.half_life
    )


def test_summarize_pair_window_ou_no_identificable() -> None:
    """Si el spread no admite un AR(1) estacionario, se informa sin fallar."""
    idx = pd.date_range("2023-01-01", periods=80, freq="B")
    # B constante => el spread es una caminata determinista: el AR(1) falla.
    plano = pd.DataFrame({"A": np.linspace(0.0, 5.0, 80), "B": np.zeros(80)}, index=idx)
    summary = summarize_pair_window(plano, "A", "B", window=60, min_obs=10)

    assert summary.ou is None
    assert summary.ou_error is not None
    tabla = summary.comparison_table()
    assert np.isnan(tabla.loc["Half-Life (sesiones)", "Última ventana"])


def test_summarize_pair_window_datos_insuficientes(log_prices_pair) -> None:
    """Con menos observaciones de las exigidas se lanza ``ValueError``."""
    with pytest.raises(ValueError, match="observaciones"):
        summarize_pair_window(
            log_prices_pair, "A", "B", window=HEDGE_WINDOW, min_obs=10000
        )


def test_summarize_pair_window_columna_inexistente(log_prices_pair) -> None:
    """Si falta un ticker se lanza ``ValueError``."""
    with pytest.raises(ValueError, match="Faltan columnas"):
        summarize_pair_window(log_prices_pair, "A", "ZZZ", window=HEDGE_WINDOW)


def test_plot_pair_window_summary_genera_figura(log_prices_pair) -> None:
    """La gráfica de la ventana reciente se genera sin errores."""
    summary = summarize_pair_window(log_prices_pair, "A", "B", window=HEDGE_WINDOW)
    plot_pair_window_summary(summary, full_spread=summary.spread)
    assert plt.get_fignums()


def test_plot_pair_window_summary_sin_spread_completo(log_prices_pair) -> None:
    """La gráfica funciona también sin aportar el spread completo."""
    summary = summarize_pair_window(log_prices_pair, "A", "B", window=HEDGE_WINDOW)
    plot_pair_window_summary(summary)
    assert plt.get_fignums()
