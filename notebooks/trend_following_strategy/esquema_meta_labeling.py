# %% [markdown]
# # Esquema del meta-labeling
#
# Diagrama pictórico: el modelo primario (cruce de medias móviles) propone
# operaciones, los eventos se etiquetan como éxito o fracaso, y un modelo
# secundario (random forest) con variables de contexto decide cuáles se
# ejecutan.  Las series son sintéticas (semilla fija).  Se guarda en
# ``figures/esquema_meta_labeling/meta_labeling.pdf``.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/esquema_meta_labeling/``.
FIG_DIR = _ROOT / "figures" / "esquema_meta_labeling"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Rótulos a ``\small`` (10.95 pt en un documento de 12 pt): a tamaño de
#: cuerpo los títulos de los paneles no caben a ancho de texto.
DIAGRAM_RC = {"font.size": 10.95}

GREEN, RED, BLUE, ORANGE, GREY, DARK = (
    "#2E9E5B", "#D1495B", "#2E6F9E", "#E08E2B", "#B8BEC6", "#2B3640",
)

# %%
# Serie sintética, medias móviles causales y eventos de cruce.
rng = np.random.default_rng(54)
N = 160
t = np.arange(N)
price = 100 + 5 * np.sin(t / 11) + 2.5 * np.sin(t / 4.3) + np.cumsum(rng.normal(0, 0.55, N))
ps = pd.Series(price)
ma_f = ps.rolling(5).mean().to_numpy()
ma_s = ps.rolling(14).mean().to_numpy()
valid = slice(14, N)
diff = ma_f - ma_s
cross = [i for i in range(15, N - 10) if diff[i - 1] * diff[i] < 0]
side = [1 if diff[i] > 0 else -1 for i in cross]
# Resultado a futuro (solo ilustrativo): éxito si el precio se mueve a favor.
out = [np.sign(price[min(i + 10, N - 1)] - price[i]) == s for i, s in zip(cross, side)]
keep = np.array(out)
conf = np.where(keep, rng.uniform(0.7, 1.0, len(cross)), rng.uniform(0.1, 0.4, len(cross)))

# %%
# Lienzo en coordenadas de datos con aspecto 1:1 (los círculos no se deforman).
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.0, 18.6, -1.0, 8.9

with plt.rc_context(DIAGRAM_RC):
    fig = plt.figure(figsize=(W, W * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    def frame(x, y, w, h, color="#F6F8FA"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color, ec=DARK, lw=0.8))

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.2,
                                     color=DARK, connectionstyle=f"arc3,rad={rad}"))

    def to_box(x0, y0, w, h):
        """Mapea (t, precio) a coordenadas del lienzo dentro de un marco."""
        lo, hi = price.min() - 2, price.max() + 2
        fx = lambda tt: x0 + 0.2 + (w - 0.4) * np.asarray(tt) / (N - 1)  # noqa: E731
        fy = lambda pp: y0 + 0.2 + (h - 0.4) * (np.asarray(pp) - lo) / (hi - lo)  # noqa: E731
        return fx, fy

    def marker(x, y, s, color, size=1.0):
        ax.scatter([x], [y + (0.22 if s == -1 else -0.22) * size],
                   marker="v" if s == -1 else "^", s=28 * size, color=color,
                   zorder=5, edgecolor="white", linewidth=0.4)

    def tick_mark(cx, cy, ok):
        """Marca blanca dentro del círculo: visto bueno o aspa."""
        r = 0.12
        if ok:
            xs, ys = [cx - r, cx - 0.03, cx + r], [cy, cy - r * 0.8, cy + r]
            ax.plot(xs, ys, color="white", lw=1.1, zorder=5, solid_capstyle="round")
        else:
            for sgn in (1, -1):
                ax.plot([cx - r, cx + r], [cy - sgn * r, cy + sgn * r], color="white",
                        lw=1.1, zorder=5, solid_capstyle="round")

    # Panel 1: medias móviles + cruces
    X1, Y1, W1, H1 = 0.3, 3.3, 5.6, 3.9
    frame(X1, Y1, W1, H1)
    fx, fy = to_box(X1, Y1, W1, H1)
    ax.plot(fx(t), fy(price), color=GREY, lw=0.8)
    ax.plot(fx(t[valid]), fy(ma_f[valid]), color=ORANGE, lw=1.2)
    ax.plot(fx(t[valid]), fy(ma_s[valid]), color=BLUE, lw=1.2)
    for i, s in zip(cross, side):
        marker(fx(i), fy(price[i]), s, DARK)

    # Tira de etiquetas (éxito / fracaso de cada señal)
    XS, YS, WS, HS = 0.3, 0.6, 5.6, 1.5
    frame(XS, YS, WS, HS)
    slots = np.linspace(XS + 0.55, XS + WS - 0.55, len(cross))
    for slot, i, o in zip(slots, cross, out):
        cy = YS + HS / 2
        ax.plot([fx(i), slot], [Y1 + 0.02, YS + HS - 0.45], color=GREY, lw=0.5, ls=":",
                zorder=1)
        ax.add_patch(Circle((slot, cy), 0.25, fc=GREEN if o else RED, ec="white", lw=0.8,
                            zorder=4))
        tick_mark(slot, cy, o)

    # Panel 2: bosque aleatorio
    X2, Y2, W2, H2 = 7.0, 3.3, 5.0, 3.9
    frame(X2, Y2, W2, H2, "#EEF6EE")

    def tree(cx, base_y, seed=0):
        r = np.random.default_rng(seed)
        ax.plot([cx, cx], [base_y - 0.35, base_y], color="#6B4A2B", lw=2.5,
                solid_capstyle="round")
        levels = [[(cx, base_y)]]
        dxs = [0.42, 0.20, 0.10]
        for d in range(3):
            cur = []
            for (px, py) in levels[-1]:
                if d == 2 and r.random() < 0.35:  # poda: árboles asimétricos
                    continue
                for k in (-1, 1):
                    nx, ny = px + k * dxs[d], py + 0.85
                    ax.plot([px, nx], [py, ny], color="#6B4A2B", lw=1.0, zorder=2)
                    cur.append((nx, ny))
            levels.append(cur)
        for lv, pts in enumerate(levels):
            for (px, py) in pts:
                if lv == 3:
                    ax.add_patch(Circle((px, py), 0.11,
                                        fc=GREEN if r.random() > 0.45 else RED,
                                        ec="white", lw=0.4, zorder=4))
                else:
                    ax.add_patch(Circle((px, py), 0.10, fc=DARK, ec="white", lw=0.4,
                                        zorder=3))

    tree(7.9, 3.95, 1)
    tree(9.5, 3.95, 2)
    tree(11.1, 3.95, 3)

    # Variables de contexto (volatilidad / indicadores)
    XC, YC, WC, HC = 7.0, 0.6, 5.0, 1.5
    frame(XC, YC, WC, HC)
    tt = np.linspace(0, 1, 120)
    vol = (0.15 + 0.35 * np.abs(np.sin(7 * tt)) * np.exp(-((tt - 0.65) ** 2) / 0.08)
           + 0.04 * rng.random(120))
    xv = XC + 0.3 + (WC - 0.6) * tt * 0.62
    ax.fill_between(xv, YC + 0.25, YC + 0.25 + vol * 2.2, color=ORANGE, alpha=0.55, lw=0)
    ax.plot(xv, YC + 0.25 + vol * 2.2, color=ORANGE, lw=0.9)
    for k, h in enumerate([0.35, 0.7, 0.5, 0.9]):
        ax.add_patch(Rectangle((XC + 3.55 + k * 0.33, YC + 0.25), 0.22, h, fc=BLUE,
                               alpha=0.8, ec="none"))

    # Panel 3: operaciones filtradas
    X3, Y3, W3, H3 = 13.1, 3.3, 5.2, 3.9
    frame(X3, Y3, W3, H3)
    fx3, fy3 = to_box(X3, Y3, W3, H3)
    ax.plot(fx3(t), fy3(price), color=GREY, lw=0.8)
    for i, s, k, c in zip(cross, side, keep, conf):
        if k:
            marker(fx3(i), fy3(price[i]), s, GREEN, size=0.7 + 0.9 * c)
        else:
            ax.scatter([fx3(i)], [fy3(price[i])], marker="x", s=18, color=RED, alpha=0.8,
                       zorder=5, linewidths=1.0)

    # Flechas de conexión
    arrow((X1 + W1 + 0.05, 5.25), (X2 - 0.05, 5.25))              # señales -> bosque
    arrow((XS + WS + 0.05, 1.35), (X2 + 0.6, Y2 - 0.02), rad=0.25)  # etiquetas -> bosque
    arrow((XC + WC / 2, YC + HC + 0.02), (XC + WC / 2, Y2 - 0.02))  # contexto -> bosque
    arrow((X2 + W2 + 0.05, 5.25), (X3 - 0.05, 5.25))              # bosque -> operaciones

    # Títulos de los paneles
    kw = dict(ha="center", color=DARK, linespacing=1.1)
    ax.text(X1 + W1 / 2, Y1 + H1 + 0.15, "Modelo primario:\nmedias móviles", va="bottom", **kw)
    ax.text(X2 + W2 / 2, Y2 + H2 + 0.15, "Modelo secundario:\nrandom forest", va="bottom", **kw)
    ax.text(X3 + W3 / 2, Y3 + H3 + 0.15, "Operaciones\nejecutadas", va="bottom", **kw)
    ax.text(XS + WS / 2, YS - 0.15, "Eventos etiquetados", va="top", **kw)
    ax.text(XC + WC / 2, YC - 0.15, "Variables de contexto", va="top", **kw)

    fig.savefig(FIG_DIR / "meta_labeling.pdf")
plt.show()
