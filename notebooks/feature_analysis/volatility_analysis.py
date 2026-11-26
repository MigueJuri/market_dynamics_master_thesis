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
# # Análisis de Volatilidad
#
# Este cuaderno aplica las funciones generales de ``feature_analysis/functions.py``
# para cuantificar la predictibilidad de la volatilidad móvil sobre el ETF ``SPY``.
# Se reutilizan ``compute_correlation_matrix``, ``plot_correlation_matrix`` y
# ``plot_feature_boxplots`` parametrizando la operación móvil (``'std'``).

# %%
from pathlib import Path
import sys
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # raíz del repositorio (``thesis/``)
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thesis.config import DATA_DIR, END, START, TICKER, TRADING_DAYS
from thesis.market_data import load_multiple_prices, load_single_price
from thesis.plotting import (
    plot_correlation_matrix,
    plot_cross_sectional_curves,
    plot_feature_boxplots,
    plot_scatter_grid,
)
from thesis.correlation import (
    compute_correlation_matrix,
    compute_cross_sectional_correlation,
)
from thesis.timeseries import log_returns

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/volatility_analysis/`` en la raíz del repositorio.
FIG_DIR = _ROOT / "figures" / "volatility_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# %%
# ## Carga de datos y construcción de la característica
price = load_single_price(TICKER, START, END, DATA_DIR)
log_ret = log_returns(price)

print(f"Ticker          : {TICKER}")
print(f"Rango           : {price.index[0]:%Y-%m-%d} a {price.index[-1]:%Y-%m-%d}")
print(f"Observaciones   : {len(price):,} precios  |  {len(log_ret):,} retornos")


def realised_volatility(log_returns_series: pd.Series, lag: int) -> pd.Series:
    """Volatilidad móvil: desviación estándar sobre los últimos ``lag`` días."""
    return log_returns_series.rolling(lag).std()


vol_windows = [21, 63, 252]

# %%
# ## Scatter volatilidad vs retorno futuro (todas las combinaciones de ventanas)
ret_windows = vol_windows
vol_by_window = {w: realised_volatility(log_ret, w) for w in vol_windows}

# Para el scatter usamos la suma de retornos futuros como objetivo.
target_by_window = {w: log_ret.rolling(w).sum().shift(-w) for w in ret_windows}

fig = plot_scatter_grid(
    vol_by_window, target_by_window,
    feature_label="Volatility", target_label="Forward return",
    feature_format=lambda k: f"{k}d", target_format=lambda k: f"{k}d",
)
fig.savefig(FIG_DIR / "volatility_spy_scatter_grid.pdf")

# %%
# ## Análisis por cuantiles de volatilidad (ventana única)
vol_window = 21
target_window = 49

vol_series = realised_volatility(log_ret, vol_window)
target_series = log_ret.rolling(target_window).sum().shift(-target_window)

pearson_corr = plot_feature_boxplots(
    vol_series, target_series,
    n_quantiles=7,
    x_axis="bins",
    feature_name=r"$\sigma$", target_name=r"$F_t$",
    show_cumulative=True,
    feature_window=vol_window, target_window=target_window,
)
plt.gcf().savefig(FIG_DIR / "volatility_spy_quantile_boxplot.pdf")
print(f"Pearson Correlation (vol → forward return): {pearson_corr:.3f}")

# %%
# ## Matriz general de correlación (volatilidad = std móvil)
vol_corr_matrix, lookbacks, lookaheads = compute_correlation_matrix(
    log_ret, log_ret,
    max_lookback=1 * TRADING_DAYS, max_lookahead=2 * TRADING_DAYS,
    steps=100, overlap=True,
    feature_op="std", target_op="sum", correlation="pearson",
)
fig = plot_correlation_matrix(
    vol_corr_matrix, lookbacks, lookaheads,
    title="Volatility–Forward Pearson Correlation",
    cbar_label="Correlación de Pearson",
)
fig.savefig(FIG_DIR / "volatility_spy_corr_matrix.pdf")

# %%
# ## Análisis cross-sectional de volatilidad por sector
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
max_lag = int(2 * trading_days_per_year)   # keep below N/3 per earlier discussion
step = max_lag // 150  # Adjust step size for better resolution
lags = np.arange(step, max_lag + step, step)

cross_corr, avg_corr = compute_cross_sectional_correlation(
    log_ret_df, lags,
    feature_op="std", target_op="sum",
    correlation="pearson", min_valid_n=50,
    overlap=True, shift_lookahead_by_lookback=True,
)
fig = plot_cross_sectional_curves(
    cross_corr, avg_corr,
    asset_groups=sector_tickers, group_colors=sector_colors,
    avg_label="Promedio de sectores",
    tau_to_yf=lambda L: 2 * L / TRADING_DAYS,
)
fig.savefig(FIG_DIR / "volatility_spy_cross_sectional_corr.pdf")
# %%
# ## Configuración del experimento
#
# ``CONFIG`` reúne los parámetros que definen el análisis.

CONFIG: dict[str, Any] = {
    "ticker": TICKER,
    "start_date": START,
    "end_date": END,
    "vol_windows": list(vol_windows),
    "vol_window": vol_window,
    "target_window": target_window,
    "matrix_correlation": "pearson",
}

# %%
# ## Métricas del experimento
#
# ``metrics_df`` resume las magnitudes ya calculadas en las celdas
# anteriores (no se recalcula nada): la correlación de la ventana única, el
# máximo de la matriz de correlación volatilidad–retorno futuro y el
# promedio cross-sectional por sector.

metrics_df = pd.DataFrame(
    [
        {
            "model": "Volatilidad",
            "pearson_corr": float(pearson_corr),
            "max_corr_matrix": float(np.nanmax(vol_corr_matrix)),
            "mean_avg_corr": float(np.nanmean(avg_corr.to_numpy())),
        }
    ]
)

# %%

# %%
# ## Publicación de métricas en el ledger
#
# Celda final que envía las métricas disponibles al ledger estandarizado de la
# tesis.  No recalcula resultados; captura las variables del notebook.

if __name__ == "__main__":
    from thesis.results import publish

    publish(
        experiment_id="feature_analysis/volatility",
        payload={"config": CONFIG, "metrics_df": metrics_df},
        chapter="Volatilidad",
    )

# %%
