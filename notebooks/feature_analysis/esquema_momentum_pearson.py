# %% [markdown]
# # Esquema de la correlación momentum–retorno futuro
#
# (a) En cada instante $t$ la serie se parte en una ventana retrospectiva
# $\tau_p$ (predictor $M_t$) y un horizonte futuro $\tau_f$ (objetivo $F_t$).
# (b) Al desplazar $t$ se obtiene una muestra de pares $(M_t, F_t)$.
# (c) La correlación de Pearson se estima sobre esa muestra.
# Los retornos son sintéticos (semilla fija).  Se guarda en
# ``figures/esquema_momentum_pearson/momentum_pearson.pdf``.

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

#: Figuras de este cuaderno: ``figures/esquema_momentum_pearson/``.
FIG_DIR = _ROOT / "figures" / "esquema_momentum_pearson"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Rótulos a ``\small`` (10.95 pt en un documento de 12 pt): el panel (b)
#: lleva texto entre celdas que a tamaño de cuerpo no cabe.
DIAGRAM_RC = {"font.size": 10.95, "axes.titlesize": 10.95, "axes.labelsize": 10.95,
              "xtick.labelsize": 10.95, "ytick.labelsize": 10.95}

C_M, C_F, C_N = "#2E86AB", "#E07A1F", "#B8BEC6"  # momentum, futuro, neutro

# %%
# Retornos log diarios sintéticos con algo de persistencia (drift lento).
rng = np.random.default_rng(7)
N, tau_p, tau_f = 3000, 20, 10
reg = np.zeros(N)
for k in range(1, N):
    reg[k] = 0.995 * reg[k - 1] + rng.normal(0, 0.00012)
x = reg + rng.normal(0, 0.01, N)

cs = np.concatenate([[0], np.cumsum(x)])
t_idx = np.arange(tau_p - 1, N - tau_f)
M = 252 / tau_p * (cs[t_idx + 1] - cs[t_idx + 1 - tau_p])
F = 252 / tau_f * (cs[t_idx + 1 + tau_f] - cs[t_idx + 1])

# %%
with plt.rc_context(DIAGRAM_RC):
    fig = plt.figure(figsize=(W, 0.8 * W))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], width_ratios=[1.25, 1])

    # (a) Serie y ventanas
    ax = fig.add_subplot(gs[0, :])
    w = 60
    t0, tp, tf = 34, 12, 7  # tamaños ilustrativos
    xs = x[1000:1000 + w] * 1e2
    pos = np.arange(w)
    cols = [C_N] * w
    for i in range(t0 - tp + 1, t0 + 1):
        cols[i] = C_M
    for i in range(t0 + 1, t0 + tf + 1):
        cols[i] = C_F
    ax.bar(pos, xs, color=cols, width=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(t0 + 0.5, color="k", ls="--", lw=0.8)
    ymax, ymin = xs.max(), xs.min()
    ax.set_ylim(ymin * 2.3 - 0.3, ymax * 1.3 + 0.5)

    ya = ymin * 1.3 - 0.1
    ax.annotate("", xy=(t0 - tp + 0.5, ya), xytext=(t0 + 0.5, ya),
                arrowprops=dict(arrowstyle="<->", color=C_M, lw=1.0))
    ax.text(t0 - 0.3, ya - 0.25, r"ventana retrospectiva: $\tau_p$ pasos",
            color=C_M, ha="right", va="top")
    ax.annotate("", xy=(t0 + 0.5, ya), xytext=(t0 + tf + 0.5, ya),
                arrowprops=dict(arrowstyle="<->", color=C_F, lw=1.0))
    ax.text(t0 + 1.3, ya - 0.25, r"horizonte futuro: $\tau_f$ pasos",
            color=C_F, ha="left", va="top")
    ax.text(t0 + 0.9, ymax, r"$t$", ha="left", va="bottom")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel(r"$x_t$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(r"(a) Partición en cada instante $t$: predictor y objetivo", loc="left")

    # (b) Ventana deslizante -> pares
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    ax2.set_title(r"(b) Desplazando $t$: pares $(M_t, F_t)$", loc="left")
    tp2, tf2, L = 8, 5, 22
    for r, tt in enumerate([9, 10, 11]):
        y = 3.4 - r * 1.2
        for c in range(L):
            col = C_M if tt - tp2 < c <= tt else (C_F if tt < c <= tt + tf2 else "#EEF0F2")
            ax2.add_patch(Rectangle((c, y), 0.92, 0.7, fc=col, ec="white", lw=0.5))
        ax2.text(-0.4, y + 0.35, [r"$t$", r"$t+1$", r"$t+2$"][r], ha="right", va="center")
        ax2.text(L + 0.4, y + 0.35,
                 [r"$(M_t,F_t)$", r"$(M_{t+1},F_{t+1})$", r"$(M_{t+2},F_{t+2})$"][r],
                 va="center")
    ax2.text(L / 2, 0.0, r"$\vdots$", ha="center", va="center")
    ax2.add_patch(Rectangle((0, 4.3), 1, 0.35, fc=C_M, ec="none"))
    ax2.text(1.5, 4.47, r"$M$ (predictor)", va="center")
    ax2.add_patch(Rectangle((16, 4.3), 1, 0.35, fc=C_F, ec="none"))
    ax2.text(17.5, 4.47, r"$F$ (objetivo)", va="center")
    ax2.set_xlim(-4, 36)
    ax2.set_ylim(-0.6, 4.9)

    # (c) Dispersión y recta de Pearson
    ax3 = fig.add_subplot(gs[1, 1])
    sub = slice(None, None, 6)
    ax3.scatter(M[sub], F[sub], s=4, alpha=0.45, color="#555", edgecolor="none")
    b, a = np.polyfit(M, F, 1)
    xx = np.array([M.min(), M.max()])
    ax3.plot(xx, a + b * xx, color="crimson", lw=1.5)
    ax3.axhline(F.mean(), color="gray", lw=0.6, ls=":")
    ax3.axvline(M.mean(), color="gray", lw=0.6, ls=":")
    ax3.set_xlabel(r"$M_t(\tau_p)$")
    ax3.set_ylabel(r"$F_t(\tau_f)$")
    ax3.grid(False)
    ax3.set_title("(c) Correlación de Pearson", loc="left")

    fig.savefig(FIG_DIR / "momentum_pearson.pdf")
plt.show()
