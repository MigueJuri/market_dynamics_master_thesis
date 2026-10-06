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
# # Análisis de Distancia a la Media Móvil
#
# Cuantifica la predictibilidad de la característica
#
# .. math:: f_t(\tau_p) = \log P_t - \log \overline{P}_{t,\tau_p}
#
# donde :math:`\log \overline{P}_{t,\tau_p}` es la media (simple o
# exponencial) de los últimos :math:`\tau_p` logaritmos de precio.  En la
# matriz de correlación, :math:`\tau_p` juega el rol del lookback (eje
# horizontal) y :math:`\tau_f` el del lookahead.
#
# Para que los valores sean comparables entre regímenes de precios, el
# target también se mide en términos relativos a la misma media
# histórica:
#
# .. math:: y_t(\tau_p, \tau_f) = \log P_{t+\tau_f} - \log \overline{P}_{t,\tau_p}
#
# La media sólo usa información disponible en :math:`t`, por lo que no
# introduce lookahead bias.

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
    apply_window, compute_correlation_matrix, plot_correlation_matrix,
    compute_cross_sectional_correlation, plot_cross_sectional_curves,
    plot_scatter_grid, plot_feature_boxplots,
    future_distance_to_moving_average,
)

# %%
# ## Parámetro del análisis
#
# ``WINDOW_TYPE`` controla si la media móvil usada en la característica es
# una media fija (``'fixed'``) o una media exponencial (``'exponential'``).
# Con ``'exponential'`` se utiliza ``ewm(span=τ_p, adjust=False)``, que
# puede actualizarse de forma incremental sin recalcular la historia.

WINDOW_TYPE = "fixed"   # 'fixed' | 'exponential'

# %%
# ## Carga de datos
price = load_single_price(TICKER, START, END, DATA_DIR)
log_ret = log_returns(price)

# ``log_price`` se calcula una sola vez y se reutiliza para todas las
# variantes de la característica y para alimentar ``compute_correlation_matrix``.
log_price = np.log(price)

# Para usar ``log_price`` como ``feature_series`` y ``target_series``
# dentro de ``compute_correlation_matrix``, ambos deben compartir índice.
# ``log_ret_aligned`` deja un NaN en la primera fecha y sólo se usa
# para mantener compatibilidad con cálculos que no usan target relativo.
log_ret_aligned = log_price - np.log(price.shift(1))

print(f"Ticker          : {TICKER}")
print(f"Tipo de ventana : {WINDOW_TYPE}")
print(f"Rango           : {price.index[0]:%Y-%m-%d} a {price.index[-1]:%Y-%m-%d}")
print(f"Observaciones   : {len(price):,} precios  |  {len(log_ret):,} retornos")
print(f"Precio inicial  : {price.iloc[0]:,.2f} USD")
print(f"Precio final    : {price.iloc[-1]:,.2f} USD")
print(f"Múltiplo total  : x{price.iloc[-1] / price.iloc[0]:,.1f}")

# %%
# ## Definición de la característica
#
# La transformación combina ``apply_window`` (media fija o exponencial) con
# un ``feature_transform`` que devuelve ``log_price - MA(log_price)``.


def distance_feature(log_price_series: pd.Series, lag: float) -> pd.Series:
    """Versión vectorizada: ``log P_t - media(log P, lag)``.

    El tipo de ventana (fija o exponencial) viene dado por ``WINDOW_TYPE``.
    """
    moving = apply_window(log_price_series, float(lag), "mean",
                         window_type=WINDOW_TYPE)
    return log_price_series - moving


# ``feature_transform`` combina la salida de ``apply_window`` con la base.
def subtract_from_log_price(windowed: pd.Series, base: pd.Series) -> pd.Series:
    return base - windowed


# %%
# ## Grilla de scatter distancia vs retorno futuro
feature_lags = [50, 189, 2 * TRADING_DAYS]
ret_lags = [50, 189, 2 * TRADING_DAYS]

feature_by_lag = {l: distance_feature(log_price, l) for l in feature_lags}
target_by_lag = {
    l: future_distance_to_moving_average(
        log_price, tau_p=l, tau_f=l, window_type=WINDOW_TYPE,
    )
    for l in ret_lags
}

plot_scatter_grid(
    feature_by_lag, target_by_lag,
    feature_label="Distance to MA", target_label="Future distance to MA",
    feature_format=lambda k: f"{k}d", target_format=lambda k: f"{k}d",
)

# %%
# ## Análisis por cuantiles (ventana única)
tau_p = 6*252
tau_f = 100

feat_series = distance_feature(log_price, tau_p)
tgt_series = future_distance_to_moving_average(
    log_price, tau_p=tau_p, tau_f=tau_f, window_type=WINDOW_TYPE,
)

pearson_corr = plot_feature_boxplots(
    feat_series, tgt_series,
    n_quantiles=5,
    x_axis = "bins",
    feature_name="distance to MA", target_name="future distance to MA",
    feature_window=tau_p, target_window=tau_f,
)
print(f"Pearson Correlation: {pearson_corr:.3f}")

# %%
# ## Matriz de correlación (τ_p = lookback, τ_f = lookahead)
#
# ``feature_op='mean'`` aplica la media móvil sobre ``log_price``;
# ``feature_transform`` la convierte en la distancia al restarla de la
# serie original.  ``target_transform`` redefine el target como la
# distancia futura a la misma media móvil histórica (calculada con la
# misma ``τ_p`` que la característica), de modo que cada celda mide la
# asociación entre ``log P_t - MA_t`` y ``log P_{t+τ_f} - MA_t``.  Esta
# normalización hace los valores comparables entre regímenes de precios.
# ``feature_window_type`` y ``target_window_type`` eligen entre ventana
# fija y exponencial.

dist_corr_matrix, lookbacks, lookaheads = compute_correlation_matrix(
    log_price, log_price,
    max_lookback=10 * TRADING_DAYS, max_lookahead=10 * TRADING_DAYS,
    steps=100, overlap=True,
    feature_op="mean",
    feature_window_type=WINDOW_TYPE,
    feature_transform=subtract_from_log_price,
    target_transform=future_distance_to_moving_average,
    target_window_type=WINDOW_TYPE,
    correlation="pearson",
)
window_label = {"fixed": "SMA", "exponential": "EMA"}[WINDOW_TYPE]
plot_correlation_matrix(
    dist_corr_matrix, lookbacks, lookaheads,
    title=f"Distance-to-{window_label} → Future Distance Pearson Correlation",
    xlabel=r"$\tau_p$ (MA window)", ylabel=r"$\tau_f$ (forward window)",
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

# Para que ``feature_df`` y el target compartan índice descartamos la primera
# fecha, donde el retorno logarítmico es NaN.
prices_df = raw.iloc[1:]
log_price_df = np.log(prices_df)
log_ret_df = (log_price_df - log_price_df.shift(1)).iloc[1:].dropna(how="all")
log_price_df = log_price_df.iloc[1:]

trading_days_per_year = 252
max_lag = int(1.5 * trading_days_per_year)   # keep below N/3 per earlier discussion
step = max_lag // 150  # Adjust step size for better resolution
lags = np.arange(step, max_lag + step, step)

cross_corr, avg_corr = compute_cross_sectional_correlation(
    log_ret_df, lags,
    feature_df=log_price_df,
    target_df=log_price_df,
    feature_op="mean",
    feature_window_type=WINDOW_TYPE,
    feature_transform=subtract_from_log_price,
    target_transform=future_distance_to_moving_average,
    target_window_type=WINDOW_TYPE,
    correlation="pearson", min_valid_n=50,
    overlap=True,
)
plot_cross_sectional_curves(
    cross_corr, avg_corr,
    asset_groups=sector_tickers, group_colors=sector_colors,
    avg_label="Promedio de sectores",
    tau_to_yf=lambda L: 2 * L / TRADING_DAYS,
)