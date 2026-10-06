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
# # Momentum Rescalado por Volatilidad
#
# Análisis de una característica tipo Sharpe Ratio: retorno acumulado
# pasado dividido por la volatilidad pasada.  El target es el retorno
# futuro escalado por la *misma* volatilidad pasada para mantener la
# comparabilidad y evitar lookahead bias.
#
# .. math::
#
#     f_t(\tau_p) = \frac{\sum_{k=1}^{\tau_p} r_{t-k}}
#                        {\operatorname{std}(r_{t-\tau_p}, \ldots, r_{t-1})}
#
# .. math::
#
#     y_t(\tau_p, \tau_f) = \frac{\sum_{k=1}^{\tau_f} r_{t+k}}
#                                {\operatorname{std}(r_{t-\tau_p+1}, \ldots, r_{t-1})}
#
# donde :math:`r_t = \log(P_t / P_{t-1})`.  Tanto el numerador del
# feature como el denominador de ambas fórmulas usan únicamente
# información disponible en :math:`t-1`.

# %%
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_analysis.functions import (
    TICKER, START, END, TRADING_DAYS, DATA_DIR,
    load_single_price, load_multiple_prices, log_returns,
    compute_correlation_matrix, plot_correlation_matrix,
    compute_cross_sectional_correlation, plot_cross_sectional_curves,
    plot_scatter_grid, plot_feature_boxplots,
    volatility_scaled_return, future_return_over_past_volatility,
    make_scaled_return_op,
)

# %%
# ## Carga de datos y retornos
price = load_single_price(TICKER, START, END, DATA_DIR)
log_ret = log_returns(price)

print(f"Ticker          : {TICKER}")
print(f"Rango           : {price.index[0]:%Y-%m-%d} a {price.index[-1]:%Y-%m-%d}")
print(f"Observaciones   : {len(price):,} precios  |  {len(log_ret):,} retornos")
print(f"Precio inicial  : {price.iloc[0]:,.2f} USD")
print(f"Precio final    : {price.iloc[-1]:,.2f} USD")
print(f"Múltiplo total  : x{price.iloc[-1] / price.iloc[0]:,.1f}")

# %%
# ## Definición de la característica y el target
#
# ``SIGMA_FLOOR`` evita divisiones por cero en ventanas con volatilidad
# nula (mercados planos).

SIGMA_FLOOR = 1e-12

scaled_return_op = make_scaled_return_op(sigma_floor=SIGMA_FLOOR)


def sharpe_feature(returns: pd.Series, lag: int) -> pd.Series:
    """Momentum escalado por volatilidad pasada."""
    return volatility_scaled_return(returns, lag, sigma_floor=SIGMA_FLOOR)


def sharpe_target(returns: pd.Series, tau_p: int, tau_f: int) -> pd.Series:
    """Retorno futuro escalado por la volatilidad pasada ``tau_p``."""
    return future_return_over_past_volatility(
        returns, tau_p=tau_p, tau_f=tau_f, sigma_floor=SIGMA_FLOOR,
    )


# %%
# ## Grilla de scatter feature × target
feature_lags = [TRADING_DAYS // 4, TRADING_DAYS // 2, 2 * TRADING_DAYS]
ret_lags = [TRADING_DAYS // 4, TRADING_DAYS // 2, 2 * TRADING_DAYS]

feature_by_lag = {l: sharpe_feature(log_ret, l) for l in feature_lags}
target_by_lag = {
    l: sharpe_target(log_ret, tau_p=l, tau_f=l)
    for l in ret_lags
}

plot_scatter_grid(
    feature_by_lag, target_by_lag,
    feature_label="Momentum / vol", target_label="Future return / vol",
    feature_format=lambda k: f"{k}d", target_format=lambda k: f"{k}d",
)

# %%
# ## Análisis por cuantiles (ventana única)
tau_p = 150
tau_f = 170

feat_series = sharpe_feature(log_ret, tau_p)
tgt_series = sharpe_target(log_ret, tau_p=tau_p, tau_f=tau_f)

pearson_corr = plot_feature_boxplots(
    feat_series, tgt_series,
    n_quantiles=10,
    x_axis = "bins",
    feature_name="momentum / vol", target_name="future return / vol",
    feature_window=tau_p, target_window=tau_f,
)
print(f"Pearson Correlation: {pearson_corr:.3f}")

# %%
# ## Matriz de correlación (τ_p × τ_f)
#
# La serie de características es ``log_ret.shift(1)`` para que el
# callable ``scaled_return_op`` vea únicamente retornos pasados.  El
# target se construye con ``future_return_over_past_volatility`` usando
# ``log_ret`` sin desplazar; dentro de la función el denominador se
# computa con ``returns.shift(1)``.

sharpe_corr_matrix, lookbacks, lookaheads = compute_correlation_matrix(
    log_ret.shift(1), log_ret,
    max_lookback=2 * TRADING_DAYS, max_lookahead=2 * TRADING_DAYS,
    steps=100, overlap=True,
    feature_op=scaled_return_op,
    target_transform=future_return_over_past_volatility,
    target_window_type="fixed",
    correlation="pearson",
)
plot_correlation_matrix(
    sharpe_corr_matrix, lookbacks, lookaheads,
    title="Momentum/Volatility → Future Return/Volatility Correlation",
    xlabel=r"$\tau_p$ (volatility-scaled momentum)",
    ylabel=r"$\tau_f$ (forward window)",
    cbar_label="Correlación de Pearson",
)

# %%
# ## Análisis cross-sectional por sector
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
max_lag = int(1.5 * trading_days_per_year)
step = max_lag // 150
lags = np.arange(step, max_lag + step, step)

# ``feature_df`` se desplaza para excluir el retorno del día ``t``;
# ``target_df`` se deja sin desplazar porque ``target_transform``
# maneja la parte futura internamente.
cross_corr, avg_corr = compute_cross_sectional_correlation(
    log_ret_df, lags,
    feature_df=log_ret_df.shift(1),
    feature_op=scaled_return_op,
    target_transform=future_return_over_past_volatility,
    target_window_type="fixed",
    correlation="pearson", min_valid_n=50,
    overlap=True,
)
plot_cross_sectional_curves(
    cross_corr, avg_corr,
    asset_groups=sector_tickers, group_colors=sector_colors,
    avg_label="Promedio de sectores",
    tau_to_yf=lambda L: 2 * L / TRADING_DAYS,
)

# %%
 