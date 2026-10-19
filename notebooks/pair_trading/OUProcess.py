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
# # Simulación del proceso de Ornstein-Uhlenbeck
#
# ## Proceso OU
#
# El proceso de Ornstein-Uhlenbeck es el análogo continuo de un
# AR(1) estacionario y representa una partícula sometida a movimiento
# Browniano dentro de un potencial armónico.  Su ecuación diferencial
# estocástica es:
#
# $$dX_t = \theta\,(\mu - X_t)\,dt + \sigma\,dW_t$$
#
# - $\theta > 0$ controla la **rigidez** del muelle (velocidad de
#   reversión a la media $\mu$).
# - $\mu$ es la **media de equilibrio**.
# - $\sigma$ es la **volatilidad** del ruido.
# - $W_t$ es un movimiento Browniano estándar.
#
# La distribución estacionaria es
# $\mathcal{N}\!\left(\mu,\ \sigma^2/(2\theta)\right)$.
#
# ## Discretización Euler-Maruyama
#
# ``simulate_ou_process`` aproxima la ecuación con paso $\Delta t$:
#
# $$X_{t+\Delta t} = X_t + \theta(\mu - X_t)\Delta t
#   + \sigma\,\sqrt{\Delta t}\,Z_t,$$
#
# donde $Z_t \sim \mathcal{N}(0, 1)$.
#
# ## Movimiento Browniano Geométrico (GBM)
#
# Para los precios sintéticos se simula un GBM:
#
# $$S_t = S_0\,
#   \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)t
#   + \sigma\,W_t\right].$$
#
# ## Construcción del par
#
# A partir del spread OU y del primer activo GBM, se define el
# segundo activo de modo que su spread logarítmico con el primero
# reproduzca exactamente la trayectoria OU:
#
# $$\mathrm{spread} = \log P^{(1)} - \log P^{(2)}
#   \quad \Longrightarrow \quad
#   P^{(2)} = \exp\!\left(\frac{\log P^{(1)} - \mathrm{spread}}
#                                {\beta}\right).$$

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

from thesis.pair_trading import simulate_gbm, simulate_ou_process

# %%
# ## Parámetros de la simulación
#
# - ``T``     : horizonte total en años (5).
# - ``dt``    : paso en años (1/252 sesiones de trading).
# - ``theta`` : rigidez del muelle (15, reversión rápida).
# - ``mu_ou`` : media de equilibrio del spread (0).
# - ``sigma_ou``: volatilidad del spread (0.2).
# - ``S0``    : precio inicial del primer activo (1).
# - ``mu_gbm``: deriva anual del GBM (0.3).
# - ``sigma_gbm``: volatilidad anual del GBM (0.20).
# - ``beta``  : ponderación del spread al reconstruir el activo 2.
T = 5.0
dt = 1 / 252
time_axis = np.linspace(0, T, int(T / dt))

mu_ou = 0.0
theta = 15.0
sigma_ou = 0.2
x0 = 0.0

S0 = 1.0
mu_gbm = 0.3
sigma_gbm = 0.20

beta = 2

# %%
# ## Simulación de los procesos
spread = simulate_ou_process(theta, mu_ou, sigma_ou, x0, T, dt)
price1 = simulate_gbm(s0=S0, mu=mu_gbm, sigma=sigma_gbm, T=T, dt=dt)
price2 = np.exp((np.log(price1) - spread) / beta)

# %%
# ## Visualización
#
# - Panel (a): ambos precios en escala logarítmica.  Las trayectorias
#   se mueven juntas pero con un desfase controlado por el spread.
# - Panel (b): el proceso OU simulado, con una línea horizontal en
#   $\mu = 0$ como referencia.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(time_axis, price1, label="Activo A", color="navy")
ax1.plot(time_axis, price2, label="Activo B", color="darkorange")
ax1.set_ylabel("Precio")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(time_axis, spread, label="Proceso OU", color="teal")
ax2.axhline(mu_ou, linestyle="--", label=r"$\mu$", color="black", alpha=0.5)
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Spread")
ax2.legend()
ax2.grid(True, alpha=0.3)

for label, ax in zip(["(a)", "(b)"], (ax1, ax2)):
    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )

plt.tight_layout()
plt.show()