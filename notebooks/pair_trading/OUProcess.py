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
# # Proceso de Ornstein-Uhlenbeck (OU)
#
# En este cuaderno se simula un spread ``OU`` junto con dos activos GBM
# cuyo spread logarítmico coincide con la trayectoria OU.  Las funciones
# de simulación viven en :mod:`thesis.pair_trading`.

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
# Time parameters
T = 5.0
dt = 1 / 252
time_axis = np.linspace(0, T, int(T / dt))

# OU Spread Parameters
mu_ou = 0.0
theta = 15.0
sigma_ou = 0.2
x0 = 0.0

# GBM Parameters for Asset 1
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
