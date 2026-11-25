# %% [markdown]
# # Esquema de señales de trend following
#
# Cruce de medias móviles: la corta cruza a la larga hacia arriba (compra) o
# hacia abajo (venta).  Las series son sintéticas (semilla fija) y solo
# ilustran la regla.  Se guarda en
# ``figures/esquema_trend_following/trend_esquema.pdf``.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/esquema_trend_following/``.
FIG_DIR = _ROOT / "figures" / "esquema_trend_following"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_FAST = "#D55E00"   # media móvil corta
C_SLOW = "#0072B2"   # media móvil larga
C_BUY = "#2E8B57"
C_SELL = "#C0392B"
C_PRICE = "#9A9A9A"

# %%
rng = np.random.default_rng(7)
t = np.linspace(0, 10, 400)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def crossing(t, fast, slow):
    """Instante (interpolado) del cruce entre ambas medias."""
    d = fast - slow
    i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0][0]
    t0 = t[i] - d[i] * (t[i + 1] - t[i]) / (d[i + 1] - d[i])
    y0 = np.interp(t0, t, slow)
    return t0, y0


# Medias móviles esquemáticas
slow = 100 + 0.35 * (t - 5)
fast_up = slow + 6.0 * (sigmoid(1.6 * (t - 5)) - 0.5) * 2   # cruza hacia arriba
fast_dn = slow - 6.0 * (sigmoid(1.6 * (t - 5)) - 0.5) * 2   # cruza hacia abajo

# "Precio" esquemático: media corta + ruido suave
noise = np.convolve(rng.normal(0, 1.0, t.size), np.ones(8) / 8, mode="same")
price_up = fast_up + 1.6 * noise
price_dn = fast_dn + 1.6 * noise

fig, axes = plt.subplots(1, 2, figsize=(W, 0.55 * W), sharey=True)

cases = [
    dict(ax=axes[0], fast=fast_up, price=price_up, kind="buy"),
    dict(ax=axes[1], fast=fast_dn, price=price_dn, kind="sell"),
]

for c in cases:
    ax, fast, price = c["ax"], c["fast"], c["price"]
    buy = c["kind"] == "buy"
    col = C_BUY if buy else C_SELL
    tc, yc = crossing(t, fast, slow)

    # Región posterior al cruce (régimen que dispara la señal)
    ax.axvspan(tc, t[-1], color=col, alpha=0.08, lw=0)
    ax.axvline(tc, color=col, ls=":", lw=1.2, alpha=0.8)

    ax.plot(t, price, color=C_PRICE, lw=0.8, alpha=0.8, label="Precio")
    ax.plot(t, slow, color=C_SLOW, lw=1.8, label="Media móvil larga")
    ax.plot(t, fast, color=C_FAST, lw=1.8, label="Media móvil corta")

    # Marcador de la señal
    ax.scatter([tc], [yc], s=70, marker="^" if buy else "v", color=col,
               edgecolor="white", linewidth=1.0, zorder=6)
    ax.annotate(
        "COMPRA" if buy else "VENTA",
        xy=(tc, yc),
        xytext=(tc - 1.6, yc + (6.5 if buy else -6.5)),
        ha="center", va="center", color=col,
        arrowprops=dict(arrowstyle="->", color=col, lw=1.3),
    )

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Tiempo")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

axes[0].set_ylabel("Precio")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3)

fig.savefig(FIG_DIR / "trend_esquema.pdf")
plt.show()
