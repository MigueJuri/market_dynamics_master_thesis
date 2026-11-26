# %% [markdown]
# # Esquema de análisis técnico
#
# Guía visual para la introducción del capítulo del mercado: una serie de velas
# japonesas sobre la que se dibujan figuras típicas del análisis técnico
# (canal alcista, triángulo ascendente con su resistencia y ruptura).  Es una figura ilustrativa: la serie es sintética (semilla fija)
# y está construida para que las figuras se vean; no representa datos reales.
# Se guarda en ``figures/esquema_analisis_tecnico/analisis_tecnico.pdf``.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/esquema_analisis_tecnico/``.
FIG_DIR = _ROOT / "figures" / "esquema_analisis_tecnico"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GREEN = "#2E9E5B"  # velas alcistas
RED = "#D64545"    # velas bajistas
LINE = "#4477AA"   # rectas trazadas: canal y triángulo
RES = "0.2"        # resistencia y ruptura
BODY_W = 0.65     # ancho del cuerpo de la vela (unidades del eje x)
SEED = 7

# %% [markdown]
# ## Serie sintética
#
# El cierre sigue una trayectoria guía por tramos más ruido pequeño:
# 1. canal alcista: oscila entre dos rectas paralelas crecientes;
# 2. triángulo ascendente: techo plano (resistencia) y mínimos crecientes;
# 3. ruptura alcista de la resistencia.
# Las rectas dibujadas son las envolventes exactas de la trayectoria guía.

# %%
rng = np.random.default_rng(SEED)

N1, N2, N3 = 44, 36, 14
t1 = np.linspace(0.0, 1.0, N1)
t2 = np.linspace(0.0, 1.0, N2)
t3 = np.linspace(0.0, 1.0, N3)

# 1. Canal alcista: centro creciente, semiancho A1, 3.5 ciclos (termina en el techo)
A1, P0, DP = 2.6, 100.0, 14.0
guide1 = P0 + DP * t1 + A1 * np.sin(2 * np.pi * 3.5 * t1 - np.pi / 2)
R = P0 + DP + A1  # resistencia = último techo del canal

# 2. Triángulo ascendente: oscila entre el piso creciente y R
floor2 = lambda t: R - 6.5 + 5.0 * t
guide2 = floor2(t2) + (R - floor2(t2)) * 0.5 * (1 + np.cos(2 * np.pi * 3.0 * t2))

# 3. Ruptura: subida con pendiente mayor
guide3 = guide2[-1] + 9.0 * t3

close = np.concatenate([guide1, guide2[1:], guide3[1:]])
close = close + rng.normal(0.0, 0.3, close.size)
n = close.size
x = np.arange(n)

open_ = np.empty(n)
open_[0] = close[0] - 0.4
open_[1:] = close[:-1] + rng.normal(0.0, 0.15, n - 1)
wick = np.abs(rng.normal(0.0, 0.4, (2, n))) + 0.1
high = np.maximum(open_, close) + wick[0]
low = np.minimum(open_, close) - wick[1]

# %% [markdown]
# ## Figura

# %%
fig, ax = plt.subplots(figsize=(W, 0.6 * W))

# Velas: verdes si cierran por encima de la apertura, rojas si no
for xi, o, h, lo_, c in zip(x, open_, high, low, close):
    col = GREEN if c >= o else RED
    ax.plot([xi, xi], [lo_, h], color=col, lw=0.7, solid_capstyle="butt", zorder=3)
    b0, b1 = min(o, c), max(o, c)
    ax.add_patch(Rectangle((xi - BODY_W / 2, b0), BODY_W, max(b1 - b0, 0.08),
                           facecolor=col, edgecolor=col, lw=0.5, zorder=4))

# 1. Canal alcista: envolventes del tramo 1
e1 = N1 - 1
xc = np.array([-0.5, e1 + 2.0])
ax.plot(xc, P0 + DP * xc / e1 - A1, color=LINE, lw=1.0, zorder=5)
ax.plot(xc, P0 + DP * xc / e1 + A1, color=LINE, lw=1.0, zorder=5)

# 2. Triángulo ascendente: piso creciente que converge con la resistencia
e2 = e1 + N2 - 1
xt = np.array([e1, e2 + 1.0])
ax.plot(xt, floor2((xt - e1) / (N2 - 1)), color=LINE, lw=1.0, zorder=5)
ax.plot([e1 - 10, e2 + 4], [R, R], color=RES, lw=1.0, ls=(0, (5, 2)), zorder=5)

# 3. Ruptura
ax.annotate("", xy=(n - 1, high[-1] + 0.8), xytext=(e2 + 1, R + 0.8),
            arrowprops=dict(arrowstyle="-|>", color=RES, lw=1.0), zorder=6)

# Etiquetas
ax.text(14, P0 + DP * 14 / e1 - A1 - 2.2, "Canal alcista", color=LINE,
        ha="left", va="top")
ax.text(e1 - 10, R + 0.6, "Resistencia", color=RES, ha="left", va="bottom")
ax.text(e1 + (N2 - 1) * 0.5, floor2(0.5) - 2.6, "Triángulo", color=LINE,
        ha="center", va="top")
ax.text(e2 + 3, R + 4.5, "Ruptura", color=RES, ha="right", va="center")

for sp in ("top", "right", "bottom", "left"):
    ax.spines[sp].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.set_xlabel("Tiempo")
ax.set_ylabel("Precio")
ax.set_xlim(-1.5, n + 0.5)

fig.savefig(FIG_DIR / "analisis_tecnico.pdf")
plt.show()
