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
# # Pair Trading
#
# Asumimos que los precios logarítmicos $X_t = \ln(P^{(1)}_t)$ e
# $Y_t = \ln(P^{(2)}_t)$ son caminantes aleatorios.  Buscamos el vector de
# cointegración $(1, -\beta)$ de modo que el spread $Z_t = Y_t - \beta X_t$
# sea estacionario y lo modelamos como un proceso de
# **Ornstein-Uhlenbeck**:
#
# $$dZ_t = \theta (\mu - Z_t)\, dt + \sigma\, dW_t$$
#
# La estacionariedad se contrasta con el test Augmented Dickey-Fuller y los
# parámetros continuos del OU se estiman ajustando un AR(1) a la versión
# discreta de la ecuación.

# %% [markdown]
# ## Hedge Ratio ($\beta$)
#
# Regresión OLS sobre $\log A_t = \beta \log B_t + \alpha + \epsilon_t$.
# El residuo es el spread que se contrasta con ADF.

# %%
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.config import DATA_DIR
from thesis.market_data import load_multiple_prices
from thesis.pair_trading import (
    extract_and_construct_spread,
    fit_ou_process,
)
from thesis.plotting import plot_ou_process, plot_pair_prices

# %%
# ## Selección del par y construcción del spread
TICKER_A = "INVH"
TICKER_B = "WBD"
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"

spread_series, hedge_ratio = extract_and_construct_spread(
    TICKER_A, TICKER_B, START_DATE, END_DATE, DATA_DIR,
)

# %% [markdown]
# ## Ajuste del proceso de Ornstein-Uhlenbeck
#
# Usamos la solución discreta $\Delta t$ constante:
#
# $$Z_t = Z_{t-1} e^{-\theta \Delta t} + \mu (1 - e^{-\theta \Delta t}) + \epsilon_t$$
#
# $$\mathrm{Var}(\epsilon_t) = \sigma^2 \frac{1 - e^{-2\theta \Delta t}}{2\theta}$$

# %%
ou_params = fit_ou_process(spread_series, dt=1.0)

print(f"Mean Reversion Rate (θ): {ou_params.theta:.4f}")
print(f"Equilibrium Mean (μ):    {ou_params.mu:.4f}")
print(f"Half-Life (días):        {ou_params.half_life:.2f}")
print(f"R² (AR(1)):         {ou_params.r_squared:.4f}")

z_score_t = (spread_series.iloc[-1] - ou_params.mu) / ou_params.sigma_eq
print(f"Z-score actual:          {z_score_t:+.4f}")

# %%
# ## Visualización del proceso OU
plot_ou_process(spread_series, ou_params)

# %%
# ## Precios del par (crudos y normalizados)
prices = load_multiple_prices(
    [TICKER_A, TICKER_B], START_DATE, END_DATE, DATA_DIR,
).dropna()

plot_pair_prices(prices, TICKER_A, TICKER_B)