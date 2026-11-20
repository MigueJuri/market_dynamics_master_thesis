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

# %%
# # Análisis de Momentum
#
# Este cuaderno aplica las funciones generales de ``feature_analysis/functions.py``
# para cuantificar la predictibilidad del momentum sobre el ETF ``SPY``.

# %%
from pathlib import Path
import sys

# Permite ejecutar el archivo desde ``notebooks/`` o desde ``notebooks/feature_analysis/``.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # raíz del repositorio (``thesis/``)
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thesis.config import DATA_DIR, END, SEED, START, TICKER, TRADING_DAYS
from thesis.market_data import load_multiple_prices, load_single_price
from thesis.plotting import (
    plot_correlation_matrix,
    plot_cross_sectional_curves,
    plot_feature_boxplots,
    plot_scatter_grid,
)
from thesis.correlation import (
    bootstrap_momentum_forward_correlation,
    compute_correlation_matrix,
    compute_cross_sectional_correlation,
)
from thesis.timeseries import log_returns, simple_returns

# %%
# ## Carga de datos y retornos
price = load_single_price(TICKER, START, END, DATA_DIR)
log_ret = log_returns(price)
simple_ret = simple_returns(price)

print(f"Ticker          : {TICKER}")
print(f"Rango           : {price.index[0]:%Y-%m-%d} a {price.index[-1]:%Y-%m-%d}")
print(f"Observaciones   : {len(price):,} precios  |  {len(log_ret):,} retornos")
print(f"Precio inicial  : {price.iloc[0]:,.2f} USD")
print(f"Precio final    : {price.iloc[-1]:,.2f} USD")
print(f"Múltiplo total  : x{price.iloc[-1] / price.iloc[0]:,.1f}")

# %%
# ## Construcción de la característica momentum
momentum_lags = [TRADING_DAYS //4  , TRADING_DAYS //2 , 2 * TRADING_DAYS]
ret_to_predict_lags = [TRADING_DAYS //4  , TRADING_DAYS //2 , 2 * TRADING_DAYS]


def momentum_feature(price: pd.Series, lag: int) -> pd.Series:
    """Cumulative log-return over the past ``lag`` sessions (excluding today)."""
    return np.log(price.shift(1)) - np.log(price.shift(lag + 1))


def forward_return(price: pd.Series, lag: int) -> pd.Series:
    """Cumulative log-return over the next ``lag`` sessions (excluding today)."""
    return np.log(price.shift(-lag)) - np.log(price.shift(-1))


momentum_by_lag = {m: momentum_feature(price, m) for m in momentum_lags}
target_by_lag = {r: forward_return(price, r) for r in ret_to_predict_lags}

# %%
# ## Grilla de scatter momentum × retorno futuro
scale = 0.7
plot_scatter_grid(
    momentum_by_lag, target_by_lag,
    figsize=(12.0*scale, 10.0*scale),
    feature_label="Momentum", target_label="Retorno Futuro",
    feature_format=lambda k: rf"$\tau_p=${k/252:.2f} años", target_format=lambda k: rf"$\tau_f=${k/252:.2f} años",
)

# %%
# ## Análisis por cuantiles de momentum (ventana única)
mom_k = 180
ret_k = 100

avg_momentum = (momentum_feature(price, mom_k) / mom_k) * TRADING_DAYS
future_return = forward_return(price, ret_k)

pearson_corr = plot_feature_boxplots(
    avg_momentum, future_return,
    n_quantiles= 10,
    x_axis = "quantiles",
    feature_name="Momentum", target_name="Retorno futuro",
    feature_window=mom_k, target_window=ret_k,
)
print(f"Pearson Correlation: {pearson_corr:.3f}")

# %%
# ## Matriz general de correlación (momentum = suma móvil)
mom_corr_matrix, lookbacks, lookaheads = compute_correlation_matrix(
    log_ret, log_ret,
    max_lookback=2 * TRADING_DAYS, max_lookahead=2 * TRADING_DAYS,
    steps=300, overlap=True,
    feature_op="sum", target_op="sum", correlation="pearson",
)
plot_correlation_matrix(
    mom_corr_matrix, lookbacks, lookaheads,
    title="Momentum–Forward Pearson Correlation",
    cbar_label="Correlación de Pearson",
)

# %%
# ## Correlación momentum–retorno futuro en $\tau$ simétrico con banda nula
#
# Para cada media ventana ``h`` (la ventana simétrica es ``τ = 2h``) se
# correlaciona el momentum pasado (``t-h..t-1``) con el retorno futuro
# (``t+1..t+h``).  Las dos ventanas son disjuntas y excluyen el día ``t``,
# así que no hay look-ahead.
#
# La banda del 95 % se obtiene con un bootstrap **i.i.d.** sobre los
# log-retornos: cada réplica remuestrea observaciones individuales con
# reemplazo, lo que **destruye la dependencia temporal** de la serie.  Por eso
# la banda es la distribución **nula** del estimador (centrada en ≈ 0) y no un
# intervalo de incertidumbre del estimador bajo dependencia real: en series
# heterocedásticas la banda subestima el ancho verdadero.  Es la misma
# motivación que las bandas i.i.d. de la ACF en
# ``notebooks/spy/SPY_caracterizacion.py``.

half_windows = np.arange(5, TRADING_DAYS + 1, 5)
tau_years = 2 * half_windows / TRADING_DAYS

point, ci_low, ci_high = bootstrap_momentum_forward_correlation(
    log_ret, half_windows, n_bootstrap=1000, seed=SEED,
)

fig, ax = plt.subplots(figsize=(7.0, 3.5))
ax.plot(tau_years, point, color="tab:blue", label="Observado")
ax.fill_between(
    tau_years, ci_low, ci_high, color="tab:blue", alpha=0.3,
    label="IC 95 % (bootstrap i.i.d.)",
)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$\tau$ (años)")
ax.set_ylabel("Correlación de Pearson")
ax.set_title(
    "Momentum–retorno futuro: correlación en $\\tau$ con banda nula i.i.d."
)
ax.legend()
plt.tight_layout()
plt.show()

 # %%
# ## Análisis cross-sectional de momentum por sector
sector_tickers = {
    "Tecnología": ["XLK"],
    "Comunicaciones": ["XLC"],
    "Financieros": ["XLF"],
    "Consumo Discrecional": ["XLY"],
    "Salud": ["XLV"],
    "Industriales": ["XLI"],
    "SPY": ["SPY"],
}
sector_colors = {
    "Tecnología": "tab:blue",
    "Financieros": "tab:green",
    "Industriales": "tab:orange",
    "Comunicaciones": "tab:cyan",
    "Salud": "tab:red",
    "Consumo Discrecional": "tab:purple",
    "SPY": "black",
}

all_tickers = [t for tickers in sector_tickers.values() for t in tickers]
raw = load_multiple_prices(all_tickers)
log_ret_df = np.log(raw / raw.shift(1)).dropna(how="all")

trading_days_per_year = 252
max_lag = int(1.5 * trading_days_per_year)   # keep below N/3 per earlier discussion
step = max_lag // 150  # Adjust step size for better resolution
lags = np.arange(step, max_lag + step, step)

cross_corr, avg_corr = compute_cross_sectional_correlation(
    log_ret_df, lags,
    feature_op="sum", target_op="sum",
    correlation="pearson", min_valid_n=50,
    overlap=True, shift_lookahead_by_lookback=True,
)
plot_cross_sectional_curves(
    cross_corr, avg_corr,
    asset_groups=sector_tickers, group_colors=sector_colors,
    avg_label="Promedio de sectores",
    tau_to_yf=lambda L: 2 * L / TRADING_DAYS,
)
# %%
# ## Publicación de métricas en el ledger
#
# Celda final que envía las métricas disponibles al ledger estandarizado de la
# tesis.  No recalcula resultados; captura las variables del notebook.

if __name__ == "__main__":
    from thesis.results import publish

    _payload = {}
    for _key in ("CONFIG", "config", "metrics", "results", "summary", "backtest"):
        if _key in globals():
            _payload[_key.lower()] = globals()[_key]

    publish(
        experiment_id="feature_analysis/momentum",
        payload=_payload,
        chapter="Momentum",
    )

# %%
