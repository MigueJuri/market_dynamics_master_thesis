# %% [markdown]
# # División cronológica de los datos
#
# Esquema de la partición en entrenamiento, validación y prueba (70/15/15 %)
# usada en ``Volatilidad_LSTM.py``.  Reemplaza a ``data_division.svg``.
# Se guarda en ``figures/esquema_division_datos/division_datos.pdf``.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/esquema_division_datos/``.
FIG_DIR = _ROOT / "figures" / "esquema_division_datos"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Particiones en orden cronológico: (nombre, fracción, color de relleno).
SPLITS = [
    ("Entrenamiento", 0.70, "#C6D9EC"),
    ("Validación", 0.15, "#F2E8B8"),
    ("Prueba", 0.15, "#F6C8CF"),
]
EDGE = "#2B303A"


# %%
def draw_division() -> plt.Figure:
    """Barra de particiones; 1 unidad del eje = 1 pulgada de la figura."""
    x_max, y_max = W, 1.25
    fig = plt.figure(figsize=(x_max, y_max), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    x0, x1 = 0.05, x_max - 0.05
    y_bar, h_bar = 0.62, 0.56
    total = x1 - x0

    x = x0
    for nombre, frac, color in SPLITS:
        w = frac * total
        ax.add_patch(Rectangle((x, y_bar), w, h_bar, facecolor=color,
                               edgecolor=EDGE, linewidth=0.8, zorder=2))
        ax.text(x + w / 2, y_bar + h_bar / 2,
                f"{nombre}\n({round(100 * frac)}\\,\\%)",
                ha="center", va="center", multialignment="center", zorder=3)
        x += w

    # Eje temporal bajo la barra: la división es cronológica.
    y_arrow = 0.40
    ax.add_patch(FancyArrowPatch(
        (x0, y_arrow), (x1, y_arrow),
        arrowstyle="-|>,head_length=3.8,head_width=2.3",
        color=EDGE, lw=0.9, mutation_scale=1.0, zorder=2,
    ))
    ax.text((x0 + x1) / 2, 0.17,
            r"Conjunto de datos completo (100\,\%), en orden cronológico",
            ha="center", va="center")
    return fig


fig = draw_division()
fig.savefig(FIG_DIR / "division_datos.pdf")
plt.show()
