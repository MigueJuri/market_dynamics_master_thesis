# %% [markdown]
# # Celda LSTM
#
# Esquema de una celda LSTM con sus tres puertas (olvido, entrada y salida),
# con la notación del apéndice de la tesis: entrada $u_t$, estado de la celda
# $c_t$, estado oculto $h_t$, puertas $f_t$, $i_t$, $o_t$ e información
# candidata $\tilde c_t$; ``sigm`` es la sigmoide.  Reemplaza a
# ``LSTMcell.svg``.  Se guarda en ``figures/esquema_celda_lstm/celda_lstm.pdf``.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/esquema_celda_lstm/``.
FIG_DIR = _ROOT / "figures" / "esquema_celda_lstm"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Texto a ``\small`` (10.95 pt), como en ``esquema_lstm.py``.
DIAGRAM_RC = {"font.size": 10.95}

#: Paleta de ``esquema_lstm.py`` más tintes suaves para cada puerta.
C = {
    "cell_bg": "#F8FAFC",
    "cell_edge": "#BDC3C7",
    "forget_bg": "#FFF4DC",
    "input_bg": "#E6F2E6",
    "output_bg": "#E3EEF9",
    "sigm": "#0056B3",
    "tanh": "#D97724",
    "op": "#2B303A",
    "state": "#1C3B5E",
    "hidden": "#228833",
    "arrow": "#2B303A",
}


# %%
def draw_cell() -> plt.Figure:
    """Dibuja la celda; 1 unidad del eje = 1 pulgada de la figura."""
    x_max, y_max = W, 2.85
    fig = plt.figure(figsize=(x_max, y_max), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    # Geometría (pulgadas)
    x_in, x_out = 0.72, 5.80          # extremos de las líneas de estado
    y_c, y_h, y_gate, y_bus = 2.55, 1.85, 1.30, 0.68
    x_f, x_i, x_g, x_o = 1.65, 2.65, 3.55, 4.65
    w_box, h_box, r_op = 0.56, 0.28, 0.11

    def arrow(p1, p2, color=C["arrow"], lw=1.1):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>,head_length=3.8,head_width=2.3",
            color=color, lw=lw, mutation_scale=1.0, zorder=2,
            shrinkA=0, shrinkB=0,
        ))

    def line(xs, ys, color=C["arrow"], lw=1.1):
        ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="butt")

    def box(cx, cy, text, fc, w=w_box, h=h_box):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            facecolor=fc, edgecolor="none", zorder=3,
        ))
        ax.text(cx, cy, text, ha="center", va="center", color="white", zorder=4)

    def op(cx, cy, text):
        ax.add_patch(Circle((cx, cy), r_op, facecolor=C["op"],
                            edgecolor="none", zorder=3))
        ax.text(cx, cy, text, ha="center", va="center", color="white", zorder=4)

    def panel(x0, x1, color, caption):
        ax.add_patch(FancyBboxPatch(
            (x0, 0.10), x1 - x0, 2.30,
            boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=color, edgecolor="none", zorder=1,
        ))
        ax.text((x0 + x1) / 2, 0.33, caption, ha="center", va="center",
                multialignment="center", zorder=4)

    # Celda y paneles de las puertas
    ax.add_patch(FancyBboxPatch(
        (1.00, 0.02), 4.30, 2.76,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=C["cell_bg"], edgecolor=C["cell_edge"], lw=1.0, zorder=0,
    ))
    panel(1.15, 2.15, C["forget_bg"], "Puerta\nde olvido")
    panel(2.30, 4.00, C["input_bg"], "Puerta\nde entrada")
    panel(4.15, 5.15, C["output_bg"], "Puerta\nde salida")

    # Estado de la celda: c_{t-1} -> (x f_t) -> (+ i_t c~_t) -> c_t
    line([x_in, x_f - r_op], [y_c, y_c], C["state"], 1.8)
    line([x_f + r_op, x_g - r_op], [y_c, y_c], C["state"], 1.8)
    line([x_g + r_op, x_out - 0.05], [y_c, y_c], C["state"], 1.8)
    arrow((x_out - 0.06, y_c), (x_out, y_c), C["state"], 1.8)
    ax.text(x_in - 0.05, y_c, r"$c_{t-1}$", ha="right", va="center")
    ax.text(x_out + 0.05, y_c, r"$c_t$", ha="left", va="center")
    op(x_f, y_c, r"$\times$")
    op(x_g, y_c, r"$+$")

    # Entradas: bus con h_{t-1} y u_t hacia las cuatro capas
    line([x_in, x_o], [y_bus, y_bus])
    ax.text(x_in - 0.05, y_bus, r"$h_{t-1},\,u_t$", ha="right", va="center")
    for x in (x_f, x_i, x_g, x_o):
        arrow((x, y_bus), (x, y_gate - h_box / 2))

    # Puerta de olvido
    box(x_f, y_gate, r"sigm", C["sigm"])
    arrow((x_f, y_gate + h_box / 2), (x_f, y_c - r_op))
    ax.text(x_f + 0.07, (y_gate + y_c) / 2, r"$f_t$", ha="left", va="center")

    # Puerta de entrada e información candidata
    box(x_i, y_gate, r"sigm", C["sigm"])
    box(x_g, y_gate, r"$\tanh$", C["tanh"])
    op(x_g, y_h, r"$\times$")
    line([x_i, x_i], [y_gate + h_box / 2, y_h])
    arrow((x_i, y_h), (x_g - r_op, y_h))
    arrow((x_g, y_gate + h_box / 2), (x_g, y_h - r_op))
    arrow((x_g, y_h + r_op), (x_g, y_c - r_op))
    ax.text(x_i + 0.07, (y_gate + y_h) / 2 + 0.03, r"$i_t$", ha="left", va="center")
    ax.text(x_g + 0.07, (y_gate + y_h) / 2 + 0.03, r"$\tilde c_t$",
            ha="left", va="center")

    # Puerta de salida: h_t = o_t * tanh(c_t)
    y_tanh_c = 2.20
    h_small = 0.24
    line([x_o, x_o], [y_c, y_tanh_c + h_small / 2], C["state"], 1.1)
    box(x_o, y_tanh_c, r"$\tanh$", C["tanh"], h=h_small)
    box(x_o, y_gate, r"sigm", C["sigm"])
    op(x_o, y_h, r"$\times$")
    arrow((x_o, y_tanh_c - h_small / 2), (x_o, y_h + r_op))
    arrow((x_o, y_gate + h_box / 2), (x_o, y_h - r_op))
    ax.text(x_o + 0.07, (y_gate + y_h) / 2 + 0.03, r"$o_t$", ha="left", va="center")
    line([x_o + r_op, x_out - 0.05], [y_h, y_h], C["hidden"], 1.8)
    arrow((x_out - 0.06, y_h), (x_out, y_h), C["hidden"], 1.8)
    ax.text(x_out + 0.05, y_h, r"$h_t$", ha="left", va="center")
    return fig


with plt.rc_context(DIAGRAM_RC):
    fig = draw_cell()
    fig.savefig(FIG_DIR / "celda_lstm.pdf")
plt.show()
