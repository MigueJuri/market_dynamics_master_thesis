# %% [markdown]
# # Esquema de velas japonesas
#
# Diagrama de dos velas (alcista y bajista) con sus precios OHLC.  Es una
# figura ilustrativa, sin datos: la geometría está en unidades arbitrarias.
# Se guarda en ``figures/esquema_velas/velas_japonesas.pdf``.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/esquema_velas/`` en la raíz del repositorio.
FIG_DIR = _ROOT / "figures" / "esquema_velas"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GREEN = "#2E9E5B"
RED = "#D64545"
GRAY = "#444444"
BODY_W = 0.7  # ancho del cuerpo de la vela (unidades del eje x)

# %%
fig, ax = plt.subplots(figsize=(W, 0.5 * W))


def candle(x, o, h, l, c, color):
    lo, hi = min(o, c), max(o, c)
    ax.plot([x, x], [l, lo], color=color, lw=2.5, solid_capstyle="butt", zorder=2)
    ax.plot([x, x], [hi, h], color=color, lw=2.5, solid_capstyle="butt", zorder=2)
    ax.add_patch(Rectangle((x - BODY_W / 2, lo), BODY_W, hi - lo, facecolor=color,
                           edgecolor=color, lw=2, zorder=3))


def label(text, xy, xytext, ha):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center", color=GRAY,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=1, ls="--",
                                shrinkA=2, shrinkB=0))


# Vela 1 alcista: Close > Open
x1, o1, h1, l1, c1 = 1.0, 3.0, 6.3, 2.5, 5.4
# Vela 2 bajista: Open = Close de la vela 1, Close < Open
x2, o2, h2, l2, c2 = 2.6, c1, 6.6, 3.0, 3.7

candle(x1, o1, h1, l1, c1, GREEN)
candle(x2, o2, h2, l2, c2, RED)

# Continuidad: Close(1) = Open(2)
ax.plot([x1 + BODY_W / 2, x2 - BODY_W / 2], [c1, c1], color=GRAY, lw=1, ls=":",
        zorder=1)

# Etiquetas vela 1 (izquierda)
d = 1.1
label("High", (x1, h1), (x1 - d, h1), "right")
label("Close", (x1 - BODY_W / 2, c1), (x1 - d, c1), "right")
label("Open", (x1 - BODY_W / 2, o1), (x1 - d, o1), "right")
label("Low", (x1, l1), (x1 - d, l1), "right")

# Etiquetas vela 2 (derecha)
label("High", (x2, h2), (x2 + d, h2), "left")
label("Open", (x2 + BODY_W / 2, o2), (x2 + d, o2), "left")
label("Close", (x2 + BODY_W / 2, c2), (x2 + d, c2), "left")
label("Low", (x2, l2), (x2 + d, l2), "left")

# Clasificación según el signo de Close - Open
ax.text(x1 - 0.2, 1.8, "Alcista\nClose $-$ Open $> 0$", ha="center", va="center",
        color=GREEN)
ax.text(x2 + 0.2, 1.8, "Bajista\nClose $-$ Open $< 0$", ha="center", va="center",
        color=RED)

for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
# Límites explícitos: las etiquetas OHLC quedan fuera del rango de los datos.
ax.set_xlim(-0.75, 4.35)
ax.set_ylim(1.1, 6.9)

fig.savefig(FIG_DIR / "velas_japonesas.pdf")
plt.show()
