# %% [markdown]
# # Arbitraje entre mercados
#
# Esquema de la convergencia de precios por arbitraje: a la izquierda, el
# flujo de órdenes de los agentes entre los mercados $A$ y $B$; a la derecha,
# la convergencia de $S_A(t)$ y $S_B(t)$ al precio de equilibrio $S^*$.
# Reemplaza a ``arbitrage.tex`` (TikZ).  Se guarda en
# ``figures/esquema_arbitraje/arbitraje.pdf``.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/esquema_arbitraje/``.
FIG_DIR = _ROOT / "figures" / "esquema_arbitraje"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Texto a ``\small`` (10.95 pt), como en el TikZ original.
DIAGRAM_RC = {"font.size": 10.95, "axes.labelsize": 10.95}

#: Colores del esquema original.
C_A, C_B, C_G = "#0054A7", "#E91106", "#000000"

#: Convergencia ilustrativa: S(t) = S* ± G0/2 · exp(-k t).
S_EQ, HALF_GAP, K_RATE = 100.0, 4.0, 0.5


# %%
def draw_arbitrage() -> plt.Figure:
    """Esquema de agentes (izquierda) y convergencia de precios (derecha)."""
    x_max, y_max = W, 3.0
    fig = plt.figure(figsize=(x_max, y_max), layout="none")

    # ---------- Esquema de agentes (coordenadas en pulgadas) ----------
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    x_c, w_box, h_box, r_ag = 1.85, 1.25, 0.56, 0.36
    y_a, y_g, y_b = 2.55, 1.50, 0.45

    def market(cy, color, text):
        ax.add_patch(FancyBboxPatch(
            (x_c - w_box / 2, cy - h_box / 2), w_box, h_box,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor="white", edgecolor=color, lw=1.4, zorder=3,
        ))
        ax.text(x_c, cy, text, ha="center", va="center", color=color,
                multialignment="center", zorder=4)

    def flow(p1, p2):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>,head_length=4.5,head_width=2.6",
            color=C_G, lw=1.3, mutation_scale=1.0, zorder=2,
            shrinkA=0, shrinkB=0,
        ))

    market(y_a, C_A, "Mercado $A$\n$S_A$ alto")
    market(y_b, C_B, "Mercado $B$\n$S_B$ bajo")
    ax.add_patch(Circle((x_c, y_g), r_ag, facecolor="white", edgecolor=C_G,
                        lw=1.3, zorder=3))
    ax.text(x_c, y_g, "agentes", ha="center", va="center", color=C_G, zorder=4)

    flow((x_c, y_b + h_box / 2), (x_c, y_g - r_ag))
    flow((x_c, y_g + r_ag), (x_c, y_a - h_box / 2))
    y_low = (y_b + h_box / 2 + y_g - r_ag) / 2
    y_up = (y_g + r_ag + y_a - h_box / 2) / 2
    ax.text(x_c + 0.08, y_low, "compran en $B$", ha="left", va="center", color=C_G)
    ax.text(x_c + 0.08, y_up, "venden en $A$", ha="left", va="center", color=C_G)

    x_side = x_c - w_box / 2 - 0.07
    ax.text(x_side, y_a, r"oferta $\uparrow$\; $S_A\downarrow$",
            ha="right", va="center", color=C_A)
    ax.text(x_side, y_b, r"demanda $\uparrow$\; $S_B\uparrow$",
            ha="right", va="center", color=C_B)

    # ---------- Convergencia de precios ----------
    left, bottom, right, top = 3.40, 0.40, 0.42, 0.12
    axp = fig.add_axes((left / x_max, bottom / y_max,
                        (x_max - left - right) / x_max,
                        (y_max - bottom - top) / y_max))
    t = np.linspace(0.0, 10.0, 200)
    gap = HALF_GAP * np.exp(-K_RATE * t)
    axp.plot(t, S_EQ + gap, color=C_A, lw=1.8)
    axp.plot(t, S_EQ - gap, color=C_B, lw=1.8)
    axp.axhline(S_EQ, color=C_G, lw=0.9, ls="--")
    axp.set_xlim(0, 10)
    axp.set_ylim(S_EQ - 5, S_EQ + 5)
    axp.set_xticks([])
    axp.set_yticks([])
    axp.grid(False)
    axp.set_xlabel(r"tiempo $t$")
    axp.set_ylabel("precio")
    for spine in ("left", "bottom"):
        axp.spines[spine].set_color(C_G)
        axp.spines[spine].set_linewidth(0.9)
    # Puntas de flecha en los ejes
    axp.plot(1, 0, ">", color=C_G, ms=4, transform=axp.transAxes, clip_on=False)
    axp.plot(0, 1, "^", color=C_G, ms=4, transform=axp.transAxes, clip_on=False)

    axp.text(0.3, S_EQ + HALF_GAP * 0.92, r"$S_A(t)$", color=C_A,
             ha="left", va="bottom")
    axp.text(0.3, S_EQ - HALF_GAP * 0.92, r"$S_B(t)$", color=C_B,
             ha="left", va="top")
    axp.text(10.15, S_EQ, r"$S^{*}$", color=C_G, ha="left", va="center")

    t_g = 1.6
    g = HALF_GAP * np.exp(-K_RATE * t_g)
    axp.annotate("", xy=(t_g, S_EQ + g), xytext=(t_g, S_EQ - g),
                 arrowprops=dict(arrowstyle="<|-|>,head_length=0.4,head_width=0.18",
                                 color=C_G, lw=0.9, shrinkA=0, shrinkB=0))
    axp.text(t_g + 0.3, S_EQ - 2.9, r"arbitraje: $G>0$", color=C_G,
             ha="left", va="center")
    axp.text(9.9, S_EQ + 0.4, r"$G\to 0$", color=C_G, ha="right", va="bottom")
    return fig


with plt.rc_context(DIAGRAM_RC):
    fig = draw_arbitrage()
    fig.savefig(FIG_DIR / "arbitraje.pdf")
plt.show()
