# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: tesis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Practical Pair Trading — Backtest walk-forward y diagnóstico
#
# Este notebook ilustra una estrategia de *pair trading* sencilla y aplica un
# backtest **walk-forward** que evita el look-ahead bias, acompañado de un
# dashboard de diagnóstico para auditar que el código funciona correctamente.
#
# ## Estructura
#
# 1. **Análisis descriptivo in-sample**: estima el hedge ratio, el test ADF
#    y el proceso OU sobre todo el periodo.  Sirve para entender el par, pero
#    **no** se usa para generar señales de trading.
# 2. **Backtest walk-forward**: cada día estima el hedge ratio y el z-score
#    únicamente con datos anteriores, abre posición cuando el z-score cruza
#    los umbrales y ejecuta al día siguiente.
# 3. **Diagnóstico**: métricas de hedge ratio, señales, retorno, riesgo y
#    comprobaciones automáticas de causalidad y coherencia interna.
#
# > **Nota:** este notebook no usa filtro de Kalman ni beta dinámico.  El beta
# > se reestima cada día con una ventana histórica fija, pero se mantiene
# > constante durante cada operación (`rebalance_hedge="on_entry"`).

# %%
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.backtest_diagnostics import compute_pair_backtest_diagnostics
from thesis.config import DATA_DIR, END, START
from thesis.market_data import load_multiple_prices
from thesis.pair_trading import (
    adf_test,
    build_strategy_comparison,
    compute_hedge_ratio,
    fit_ou_process,
    pair_trading_backtest,
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

# %% [markdown]
# ## Selección del par y parámetros

# %%
TICKER_A = "FRT"
TICKER_B = "REG"
START_DATE = "2006-01-01" #START
END_DATE = END

# Ventanas del backtest walk-forward.
Z_WINDOW = 63#60  # Días de historia para estimar media y std del spread.
HEDGE_WINDOW = 252#Z_WINDOW * 4  # Días de historia para estimar el hedge ratio.

ENTRY_Z = 2.0
EXIT_Z = 0.0
COST = 0.0

# Límite de sanity check para el hedge ratio (sólo diagnóstico).
MAX_ABS_BETA = 5.0

# %% [markdown]
# ## Carga de datos
#
# Se reutiliza `load_multiple_prices` para aprovechar la caché CSV de `data/`.

# %%
prices = load_multiple_prices(
    [TICKER_A, TICKER_B], START_DATE, END_DATE, DATA_DIR
).dropna()

log_prices = prices.apply(lambda s: s[s > 0]).dropna().apply("log")

# %% [markdown]
# ## 1. Análisis descriptivo in-sample
#
# Esta sección calcula el hedge ratio, el ADF y el OU sobre todo el periodo.
# Su único propósito es diagnosticar si el par parece cointegrado.  Estos
# valores **no** entran en el backtest.

# %%
hedge_full = compute_hedge_ratio(log_prices[TICKER_A], log_prices[TICKER_B])
spread_full = log_prices[TICKER_A] - (
    hedge_full.alpha + hedge_full.beta * log_prices[TICKER_B]
)

print("=== Análisis descriptivo in-sample ===")
print(f"Hedge Ratio (β): {hedge_full.beta:.4f}")
print(f"Intercept (α):   {hedge_full.alpha:.4f}")
print(f"R-squared:       {hedge_full.r_squared:.4f}")

adf_full = adf_test(spread_full)
print(f"\nADF Statistic: {adf_full.statistic:.4f}")
print(f"p-value:       {adf_full.pvalue:.4f}")
print(
    "El spread parece estacionario."
    if adf_full.is_stationary
    else "ALERTA: el spread NO parece estacionario."
)

ou_full = fit_ou_process(spread_full, dt=1.0)
print(f"\nHalf-Life in-sample: {ou_full.half_life:.2f} días")

# %% [markdown]
# ## 1.b. Estado actual del par: última ventana
#
# Además de los estadísticos de la muestra completa, calculamos sobre las
# últimas `HEDGE_WINDOW` observaciones el hedge ratio, el test ADF y el
# ajuste del proceso de Ornstein-Uhlenbeck (de donde sale la **vida media**
# de la reversión).  Esto describe el estado *actual* del par y permite
# detectar degradaciones: un hedge ratio que se aleja del valor histórico,
# una cointegración que ha desaparecido o una reversión que se ha ralentizado.
#
# > Esta ventana **incluye** la última observación, por lo que es
# > exclusivamente diagnóstica.  No debe usarse para generar señales
# > históricas: dentro del backtest, la ventana disponible en cada fecha
# > termina en ``t-1``.

# %%
window_summary = summarize_pair_window(
    log_prices,
    TICKER_A,
    TICKER_B,
    window=HEDGE_WINDOW,
)

print(
    f"Ventana reciente: {window_summary.start:%Y-%m-%d} → "
    f"{window_summary.end:%Y-%m-%d} ({window_summary.n_obs} observaciones)"
)
print()
print(window_summary.comparison_table().to_string())

if window_summary.ou is None:
    print(
        f"\nALERTA: no se pudo ajustar el OU en la ventana reciente: "
        f"{window_summary.ou_error}"
    )
else:
    print(
        f"\nHalf-Life de la última ventana: {window_summary.ou.half_life:.2f} sesiones "
        f"(θ = {window_summary.ou.theta:.4f}, σ_eq = {window_summary.ou.sigma_eq:.4f})"
    )
    if window_summary.half_life_change_pct is not None:
        print(
            f"Variación de la vida media respecto a la muestra completa: "
            f"{window_summary.half_life_change_pct:+.2%} "
            f"({window_summary.half_life_change:+.2f} sesiones)"
        )

if window_summary.stationarity_lost:
    print(
        "\nALERTA: el spread era estacionario en la muestra completa pero "
        "deja de serlo en la última ventana."
    )
if window_summary.beta_change_pct is not None:
    print(
        f"Variación del beta respecto a la muestra completa: "
        f"{window_summary.beta_change_pct:+.2%}"
    )

# %% [markdown]
# ## 2. Backtest walk-forward
#
# Cada día se reestiman alpha y beta con los últimos `HEDGE_WINDOW` días, se
# normaliza el spread con los últimos `Z_WINDOW` días y se genera la señal.
# La posición se abre al cierre del día de señal y el retorno se contabiliza
# desde el día siguiente.

# %%
bt = pair_trading_backtest(
    log_prices,
    TICKER_A,
    TICKER_B,
    hedge_window=HEDGE_WINDOW,
    z_window=Z_WINDOW,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    cost=COST,
    rebalance_hedge="daily",
)

print("=== Backtest walk-forward ===")
print(f"Sharpe Ratio:    {bt.sharpe:.2f}")
print(f"Total Return:    {bt.total_return:.2%}")
print(f"Max Drawdown:    {bt.max_drawdown:.2%}")
print(f"Operaciones:     {bt.trades}")

# %% [markdown]
# ## 2.b Comparativa con SPY y buy & hold de los componentes
#
# Las métricas de la estrategia sólo no dicen si el resultado es bueno: hay
# que contrastarlas con alternativas que hubiesen sido posibles sobre el
# **mismo** capital y el **mismo** periodo.  Se comparan cuatro
# alternativas:
#
# 1. **Estrategia pair trading**: la del backtest walk-forward anterior.
# 2. **SPY**: buy & hold del ETF del S&P 500, el benchmark de mercado.
# 3. **Buy & hold de ``TICKER_A``**.
# 4. **Buy & hold de ``TICKER_B``**.
#
# ## Mismo periodo para todos
#
# La comparación usa **exactamente** el índice de resultados del backtest, de
# modo que las cuatro alternativas acumulan el mismo número de sesiones y
# el mismo calendario.  Sin este recorte la comparación no sería homogénea:
# un buy & hold con más historia acumularía más retorno por el solo hecho de
# haber existido durante más tiempo.
#
# Esto incluye el **warm-up** del backtest: durante las primeras
# `HEDGE_WINDOW` sesiones la estrategia todavía no puede operar y permanece
# en efectivo, con retorno cero, mientras que los benchmarks sí acumulan su
# retorno.  Es el tratamiento honesto de "habría desplegado esta estrategia
# con estos datos en esta fecha", y hace que la comparación sea
# conservadora para la estrategia.
#
# ## Métricas y convención
#
# `build_strategy_comparison` calcula el **retorno total** y el **Sharpe**
# con la misma convención que usa el propio backtest:
#
# - Retorno total: `exp(sum(retornos logarítmicos)) - 1`.
# - Sharpe: `media / desviación * sqrt(252)`, **sin tasa libre de riesgo**.
#
# El Sharpe no lleva tasa libre de riesgo por coherencia con `bt.sharpe`,
# que es la cifra que el resto del cuaderno reporta y audita.  Es la
# convención habitual para comparar estrategias entre sí, pero conviene
# tener presente que un Sharpe sin ``rf`` no es el Sharpe "de mercado": al
# omitir el término ``rf`` el valor queda inflado en la misma proporción
# que el retorno sin riesgo del periodo.
#
# Los benchmarks se calculan con **retornos logarítmicos**, igual que la
# estrategia, de modo que el retorno total sea directamente comparable.

# %%
BENCHMARK_TICKER = "SPY"

# El índice efectivo del backtest: es la referencia temporal de la tabla.
backtest_index = bt.results.index

# SPY se carga con la misma caché CSV que el resto de precios del cuaderno.
spy_prices = load_multiple_prices(
    [BENCHMARK_TICKER], START_DATE, END_DATE, DATA_DIR
)[BENCHMARK_TICKER]


def log_returns_from_prices(
    price_series: pd.Series, index: pd.DatetimeIndex
) -> pd.Series:
    """Retornos logarítmicos diarios a partir de una serie de **precios**.

    La entrada debe ser el precio, no su logaritmo: aplicar ``log`` a un
    log-precio daría ``log(log(P))``, que no es un retorno y subestimaría
    el rendimiento de forma silenciosa.  Por eso ``prices`` (precio crudo) es
    la entrada correcta y ``log_prices`` sólo se usa para el backtest.

    El primer día no tiene precio previo, así que su retorno es ``NaN``.  Se
    rellena con ``0.0`` por la misma razón que ``bh_equity_A`` y
    ``bh_equity_B`` dentro del backtest: el capital parte de 1.0 en la
    primera sesión, sin movimiento previo que contabilizar.
    """
    aligned = price_series.reindex(index)
    return np.log(aligned).diff().fillna(0.0)


# Todos los benchmarks se construyen desde el **precio crudo**:
# ``prices`` para los componentes del par y ``spy_prices`` para el ETF.
benchmarks = {
    f"Buy & hold {BENCHMARK_TICKER}": log_returns_from_prices(
        spy_prices, backtest_index
    ),
    f"Buy & hold {TICKER_A}": log_returns_from_prices(
        prices[TICKER_A], backtest_index
    ),
    f"Buy & hold {TICKER_B}": log_returns_from_prices(
        prices[TICKER_B], backtest_index
    ),
}

# La curva de capital de SPY se añade a ``results`` para que las gráficas de
# equity la incluyan junto a los benchmarks de los componentes, que ya
# venían como ``bh_equity_A`` y ``bh_equity_B``.
bt.results["bh_equity_SPY"] = np.exp(
    benchmarks[f"Buy & hold {BENCHMARK_TICKER}"].cumsum()
)

comparativa = build_strategy_comparison(
    bt.results["strategy_ret"],
    benchmarks,
)

print("=== Comparativa con benchmarks (mismo periodo) ===")
print(comparativa.to_string(index=False))
print(
    f"\nPeriodo común: {comparativa['inicio'].iloc[0]:%Y-%m-%d} → "
    f"{comparativa['fin'].iloc[0]:%Y-%m-%d} "
    f"({int(comparativa['n_sesiones'].iloc[0])} sesiones)"
)

# %% [markdown]
# ### Coherencia con las métricas del backtest
#
# La fila de la estrategia de la tabla debe coincidir con `bt.total_return`
# y `bt.sharpe`, que son las cifras reportadas en la sección anterior.  La
# comprobación fija ese contrato: si cambia el recorte del periodo o la
# convención de anualización, la diferencia dejaría de ser nula.

# %%
fila_estrategia = comparativa.loc[
    comparativa["estrategia"] == "Estrategia pair trading"
].iloc[0]

print(
    f"Retorno total: tabla {fila_estrategia['retorno_total']:.6%} "
    f"vs bt.total_return {bt.total_return:.6%}"
)
print(
    f"Sharpe:        tabla {fila_estrategia['sharpe']:.6f} "
    f"vs bt.sharpe {bt.sharpe:.6f}"
)
assert np.isclose(fila_estrategia["retorno_total"], bt.total_return)
assert np.isclose(fila_estrategia["sharpe"], bt.sharpe)
print("\nCoherencia verificada: la tabla reproduce las métricas del backtest.")

# %% [markdown]
# ### Validación de los benchmarks contra el ratio de precios
#
# Un buy & hold no puede tener un retorno distinto del que dan sus propios
# precios: partiendo de 1.0 y revaluando cada día, el capital final es
# exactamente ``P_fin / P_ini``.  La comprobación contrasta esa identidad
# para cada benchmark.
#
# > Esta validación es la que detecta el error de aplicar el logaritmo dos
# > veces (``log(log(P))``), que subestimaba el retorno de los componentes
# > del par de forma silenciosa: el resultado seguía pareciendo plausible,
# > pero era incorrecto.

# %%
for etiqueta, serie_precios in [
    (f"Buy & hold {BENCHMARK_TICKER}", spy_prices),
    (f"Buy & hold {TICKER_A}", prices[TICKER_A]),
    (f"Buy & hold {TICKER_B}", prices[TICKER_B]),
]:
    precios = serie_precios.reindex(backtest_index)
    retorno_por_precios = float(precios.iloc[-1] / precios.iloc[0] - 1.0)
    fila = comparativa.loc[comparativa["estrategia"] == etiqueta].iloc[0]
    print(
        f"{etiqueta:22s} ratio de precios {retorno_por_precios:>10.2%} "
        f"| tabla {fila['retorno_total']:>10.2%}"
    )
    assert np.isclose(fila["retorno_total"], retorno_por_precios), (
        f"{etiqueta}: el retorno de la tabla no coincide con el ratio de "
        f"precios.  Revisar que los retornos se calculan sobre el precio "
        f"crudo y no sobre su logaritmo."
    )

print("\nValidación superada: los benchmarks coinciden con sus precios.")

# %% [markdown]
# ## 3. Diagnóstico del backtest
#
# `compute_pair_backtest_diagnostics` agrega cuatro bloques de métricas y
# ejecuta una batería de comprobaciones automáticas que detectan los errores
# más habituales: look-ahead, desalineación de posiciones, beta efectivo
# incoherente, turnover inconsistente o contribuciones que no cuadran.

# %%
diag = compute_pair_backtest_diagnostics(
    bt.results,
    bt.equity,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    hedge_window=HEDGE_WINDOW,
    z_window=Z_WINDOW,
    log_prices=log_prices,
    ticker_a=TICKER_A,
    ticker_b=TICKER_B,
    max_abs_beta=MAX_ABS_BETA,
)

# %% [markdown]
# ### 3.1 Hedge ratio
#
# `beta` es la estimación walk-forward diaria; `effective_beta` es el hedge
# ratio realmente aplicado, constante dentro de cada operación.  Si
# `effective_beta_unique` fuese 1 con varias operaciones, el backtest estaría
# reutilizando el beta de la última entrada.

# %%
print("=== Hedge ratio ===")
for key, value in diag.hedge.as_dict().items():
    if isinstance(value, float):
        print(f"{key:>36}: {value: .6f}")
    else:
        print(f"{key:>36}: {value}")

# %% [markdown]
# ### 3.2 Señales y ejecución

# %%
print("=== Señales y ejecución ===")
for key, value in diag.signal.as_dict().items():
    if isinstance(value, float):
        print(f"{key:>36}: {value: .6f}")
    else:
        print(f"{key:>36}: {value}")

# %% [markdown]
# ### 3.3 Retorno, notional y costes

# %%
print("=== Retorno y notional ===")
for key, value in diag.returns.as_dict().items():
    if isinstance(value, float):
        print(f"{key:>36}: {value: .6f}")
    else:
        print(f"{key:>36}: {value}")

# %% [markdown]
# ### 3.4 Riesgo y rendimiento

# %%
print("=== Riesgo y rendimiento ===")
for key, value in diag.risk.as_dict().items():
    if isinstance(value, float):
        print(f"{key:>36}: {value: .6f}")
    else:
        print(f"{key:>36}: {value}")

# %% [markdown]
# ### 3.5 Comprobaciones automáticas
#
# Cada comprobación devuelve `passed` y una descripción del resultado.  Un
# fallo en `causalidad_beta` o `causalidad_z_score` indica look-ahead; un
# fallo en `beta_efectivo_constante_en_bloque` indica que el beta se está
# reutilizando entre operaciones.

# %%
validation = diag.validation.as_dataframe()
print(f"Resumen: {diag.validation.summary()}")
validation

# %% [markdown]
# ### 3.6 Operaciones individuales
#
# Una fila por operación con la fecha de entrada y salida, el z-score de
# entrada, el beta aplicado, la duración y el resultado neto.

# %%
if diag.trades.empty:
    print("No se han registrado operaciones.")
else:
    display(diag.trades)

# %% [markdown]
# ## 4. Gráficas de diagnóstico
#
# ### 4.1 Dashboard completo (6 paneles)

# %%
plot_backtest_dashboard(
    bt.results,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    in_sample_beta=hedge_full.beta,
)

# %% [markdown]
# ### 4.2 Hedge ratio
#
# El panel superior contrasta el beta walk-forward (línea continua) con el
# beta efectivo (escalonado).  El escalonado confirma que cada operación
# usa su propio hedge ratio.

# %%
plot_hedge_diagnostics(bt.results, in_sample_beta=hedge_full.beta)

# %% [markdown]
# ### 4.3 Señales y ejecución

# %%
plot_signal_diagnostics(bt.results, entry_z=ENTRY_Z, exit_z=EXIT_Z)

# %% [markdown]
# ### 4.4 Retorno, notional y costes

# %%
plot_return_diagnostics(bt.results)

# %% [markdown]
# ### 4.5 Equity, drawdown y estabilidad

# %%
plot_equity_diagnostics(bt.results)

# %% [markdown]
# ### 4.6 Estado actual del par: muestra completa vs última ventana
#
# La gráfica superpone el spread completo y el de la última ventana, con
# las bandas de equilibrio de la ventana reciente, y contrasta los tres
# hedge ratio: walk-forward, muestra completa y ventana reciente.

# %%
plot_pair_window_summary(
    window_summary,
    full_spread=spread_full,
    beta_history=bt.results["beta"],
)


# %%
# ## 5. Curva de capital comparativa
#
# Curvas de capital de la estrategia frente a los tres benchmarks, todas
# sobre el mismo periodo y con capital inicial de 1.0.  La escala logarítmica
# permite comparar rutas de pendientes muy distintas de forma legible.

# %%
results = bt.results

plt.figure()

equity = results["equity"]
gross_equity = np.exp(results["gross_return"].cumsum())
plt.semilogy(
    equity.index, equity, color="green", linewidth=1.8, label="Estrategia"
)
if "bh_equity_A" in results.columns:
    plt.plot(
        results.index,
        results["bh_equity_A"],
        color="orange",
        linewidth=1.0,
        linestyle=":",
        label="Buy & Hold A",
    )
    plt.plot(
        results.index,
        results["bh_equity_B"],
        color="blue",
        linewidth=1.0,
        linestyle=":",
        label="Buy & Hold B",
    )
# if "bh_equity_SPY" in results.columns:
#     plt.plot(
#         results.index,
#         results["bh_equity_SPY"],
#         color="black",
#         linewidth=1.2,
#         linestyle="-.",
#         label="Buy & Hold SPY",
#     )
plt.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
plt.ylabel("Curvas de Capital (log scale)")
plt.title("Curva de capital", fontweight="bold")
plt.legend(loc="best", fontsize=8)
plt.grid(True, alpha=0.3)
# %%
