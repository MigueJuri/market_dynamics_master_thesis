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
# # Pair Trading — Construcción del Spread y Modelo OU
#
# ## Idea central
#
# Asumimos que los precios logarítmicos $X_t = \ln(P^{(1)}_t)$ e
# $Y_t = \ln(P^{(2)}_t)$ son caminantes aleatorios ($I(1)$).
# Buscamos el vector de cointegración $(1, -\beta)$ de modo que el
# *spread*
#
# $$Z_t = Y_t - \beta\, X_t$$
#
# sea estacionario ($I(0)$).  Modelamos entonces $Z_t$ como un proceso
# de **Ornstein-Uhlenbeck (OU)**, la versión continua del AR(1):
#
# $$dZ_t = \theta\,(\mu - Z_t)\,dt + \sigma\,dW_t$$
#
# donde:
#
# - $\theta > 0$ es la **rigidez** de reversión (velocidad con la que
#   el spread vuelve a $\mu$);
# - $\mu$ es la **media de equilibrio**;
# - $\sigma$ es la **volatilidad** de los shocks;
# - $W_t$ es un movimiento Browniano estándar.
#
# ## Flujo del notebook
#
# 1. Descargar precios ajustados de los dos activos.
# 2. Estimar el *hedge ratio* $\beta$ por OLS.
# 3. Construir el spread residual y aplicar el test ADF para confirmar
#    estacionariedad.
# 4. Ajustar el modelo OU por máxima verosimilitud (vía AR(1)).
# 5. Graficar el spread, los precios y una vista conjunta con subplots.

# %% [markdown]
# ## Hedge Ratio ($\beta$)
#
# Regresión OLS sobre
# $\log A_t = \alpha + \beta \log B_t + \varepsilon_t$, donde
# $\beta$ es el *hedge ratio*.  El residuo
# $\varepsilon_t = \log A_t - \alpha - \beta \log B_t$ es el spread
# estacionario candidato.
#
# El test ADF (Augmented Dickey-Fuller) contrasta la hipótesis nula
# de raíz unitaria:
#
# $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} +
#   \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$
#
# Si el p-value es $< 0.05$, rechazamos la hipótesis nula y
# aceptamos que el spread es estacionario.

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
from thesis.plotting import plot_pair_overview, plot_pair_prices, plot_spread

# %%
# ## Selección del par y construcción del spread
#
# ``extract_and_construct_spread`` descarga ambos activos con la
# caché CSV de ``market_data``, aplica log-precios, ajusta la OLS,
# construye el spread y aplica el test ADF, imprimiendo un resumen.
TICKER_A = "NSC"
TICKER_B = "UNP"
START_DATE = "1993-01-01"
END_DATE = "2026-01-01"

spread_series, hedge_ratio = extract_and_construct_spread(
    TICKER_A, TICKER_B, START_DATE, END_DATE, DATA_DIR,
)

# %% [markdown]
# ## Ajuste del proceso de Ornstein-Uhlenbeck
#
# Para series discretas con paso $\Delta t$ constante, la solución
# analítica del OU es:
#
# $$Z_t = Z_{t-1} e^{-\theta \Delta t}
#   + \mu (1 - e^{-\theta \Delta t}) + \varepsilon_t$$
#
# con
#
# $$\mathrm{Var}(\varepsilon_t)
#   = \sigma^2\, \frac{1 - e^{-2\theta \Delta t}}{2\theta}.$$
#
# Esto equivale a un AR(1):
#
# $$Z_t = a + b\, Z_{t-1} + \varepsilon_t,$$
#
# donde $a = \mu (1 - e^{-\theta\Delta t})$ y
# $b = e^{-\theta \Delta t}$.  Invertimos las relaciones:
#
# $$\theta = -\frac{\log b}{\Delta t}, \quad
#   \mu = \frac{a}{1 - b}, \quad
#   \sigma = \sqrt{\frac{2\theta\, \mathrm{Var}(\varepsilon)}
#                        {1 - b^2}}.$$
#
# La **vida media** de la reversión (tiempo típico para volver a $\mu$
# después de un shock) es:
#
# $$t_{1/2} = \frac{\log 2}{\theta}.$$
#
# El z-score estacionario compara la posición actual del spread con la
# distribución de equilibrio del OU:

# %%
ou_params = fit_ou_process(spread_series, dt=1.0)

print(f"Mean Reversion Rate (θ): {ou_params.theta:.4f}")
print(f"Equilibrium Mean (μ):    {ou_params.mu:.4f}")
print(f"Half-Life (días):        {ou_params.half_life:.2f}")
print(f"R² (AR(1)):         {ou_params.r_squared:.4f}")

z_score_t = (spread_series.iloc[-1] - ou_params.mu) / ou_params.sigma_eq
print(f"Z-score actual:          {z_score_t:+.4f}")

# %%
# ## Los precios del par
#
# Descargamos los precios ajustados una sola vez y los reutilizamos en
# las dos gráficas siguientes.
prices = load_multiple_prices(
    [TICKER_A, TICKER_B], START_DATE, END_DATE, DATA_DIR,
).dropna()

# %%
# ## 1. Spread con bandas de equilibrio del OU
#
# La banda de equilibrio se define como
# $\sigma_{eq} = \sqrt{\sigma^2 / (2\theta)}$.  Un spread fuera de
# $\pm 2\sigma_{eq}$ es una señal teórica de entrada; dentro de
# $\pm\sigma_{eq}$ sugiere cierre de posición.
plot_spread(
    spread_series,
    ou_params,
    title=f"Spread {TICKER_A}–{TICKER_B} (hedge ratio β = {hedge_ratio:.3f})",
)

# %%
# ## 2. Precios del par
#
# Escala logarítmica para comparar activos con niveles de precio muy
# distintos.
plot_pair_prices(
    prices,
    TICKER_A,
    TICKER_B,
    title=f"Precios ajustados: {TICKER_A} vs {TICKER_B}",
    log_scale=True,
)

# %%
# ## 3. Vista conjunta (subplots)
#
# Precios arriba y spread abajo, compartiendo el eje temporal.  Permite
# verificar visualmente que las divergencias de precio se compensan en
# el spread gracias al hedge ratio.
plot_pair_overview(
    prices,
    spread_series,
    TICKER_A,
    TICKER_B,
    ou_params,
    log_scale=True,
)
