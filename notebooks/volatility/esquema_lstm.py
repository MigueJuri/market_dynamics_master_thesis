# %% [markdown]
# # Arquitectura de la red LSTM
#
# Diagrama del modelo de ``Volatilidad_LSTM.py``: bloque residual LSTM
# (izquierda) y cabezal MLP (derecha), con las dimensiones de cada tensor.
# Se guarda en ``figures/esquema_lstm/arquitectura_lstm.pdf``.

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

#: Figuras de este cuaderno: ``figures/esquema_lstm/`` en la raíz del repositorio.
FIG_DIR = _ROOT / "figures" / "esquema_lstm"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Texto de las cajas a ``\small`` (10.95 pt en un documento de 12 pt): a
#: tamaño de cuerpo los rótulos no caben en las cajas a ancho de texto.
DIAGRAM_RC = {"font.size": 10.95}

# Paleta sobria (azules + acento cálido para la activación)
C = {
    "io": "#1C3B5E",
    "extract": "#10253B",
    "norm_lin": "#52657A",
    "lstm_bg": "#E3EEF9",
    "lstm_edge": "#1C3B5E",
    "dropout": "#495057",
    "fc": "#0056B3",
    "act": "#D97724",
    "head_bg": "#F8FAFC",
    "head_edge": "#4A90E2",
    "arrow": "#2B303A",
    "dim": "#495057",
}


# %%
def draw_lstm() -> plt.Figure:
    """Dibuja la arquitectura; 1 unidad del eje = 1 pulgada de la figura."""
    # Lienzo en pulgadas: ancho W, alto proporcional al contenido.
    x_max, y_max = W, 4.0
    fig = plt.figure(figsize=(x_max, y_max), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    w_l, w_r = 2.0, 2.45
    h_std, h_lstm = 0.34, 0.42
    x_l, x_r = 1.35, 4.68

    def add_box(cx, cy, w, h, text, fc, ec="none", tc="white", lw=1.0):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
        ))
        ax.text(cx, cy, text, ha="center", va="center", color=tc, zorder=4)

    def add_arrow(p1, p2):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>,head_length=3.8,head_width=2.3",
            color=C["arrow"], lw=1.1, mutation_scale=1.0, zorder=2,
        ))

    def add_dim(x, y, text, ha="left"):
        ax.text(x, y, text, ha=ha, va="center", color=C["dim"], zorder=4)

    # Columna izquierda: bloque residual LSTM
    y_in, y_lin, y_ln = 3.65, 3.00, 2.32
    y_lstm, y_do1, y_add = 1.68, 1.06, 0.45

    add_box(x_l, y_in, w_l, h_std, "Entrada", C["io"])
    add_box(x_l, y_lin, w_l, h_std, "Lineal", C["norm_lin"])
    add_box(x_l, y_ln, w_l, h_std, "Pre-LayerNorm", C["norm_lin"])
    add_box(x_l, y_lstm, w_l, h_lstm, "LSTM (hidden $=16$)",
            C["lstm_bg"], ec=C["lstm_edge"], tc=C["lstm_edge"], lw=1.3)
    add_box(x_l, y_do1, w_l, h_std, "Dropout 1", C["dropout"])

    # Nodo de suma residual
    r_add = 0.14
    ax.add_patch(Circle((x_l, y_add), r_add, facecolor="white",
                        edgecolor=C["arrow"], lw=1.2, zorder=3))
    ax.text(x_l, y_add, "$+$", ha="center", va="center", color=C["arrow"], zorder=4)

    add_arrow((x_l, y_in - h_std / 2), (x_l, y_lin + h_std / 2))
    add_arrow((x_l, y_lin - h_std / 2), (x_l, y_ln + h_std / 2))
    add_arrow((x_l, y_ln - h_std / 2), (x_l, y_lstm + h_lstm / 2))
    add_arrow((x_l, y_lstm - h_lstm / 2), (x_l, y_do1 + h_std / 2))
    add_arrow((x_l, y_do1 - h_std / 2), (x_l, y_add + r_add))

    # Conexión residual (skip connection)
    y_branch = (y_lin - h_std / 2 + y_ln + h_std / 2) / 2
    x_skip = 0.22
    ax.plot([x_l, x_skip, x_skip], [y_branch, y_branch, y_add],
            color=C["arrow"], lw=1.1, zorder=2)
    add_arrow((x_skip, y_add), (x_l - r_add, y_add))

    # Dimensiones izquierda (a la derecha de la flecha vertical)
    add_dim(x_l + 0.08, (y_in + y_lin) / 2, r"[batch, lookback, $n_f$]")
    add_dim(x_l + 0.08, y_branch, "[batch, lookback, hidden]")

    # Conexión entre columnas
    x_mid = 3.3
    y_ext = y_in
    ax.plot([x_l + r_add, x_mid, x_mid], [y_add, y_add, y_ext],
            color=C["arrow"], lw=1.1, zorder=2)
    add_arrow((x_mid, y_ext), (x_r - w_r / 2, y_ext))

    # Columna derecha: cabezal MLP y salida
    y_fc1, y_elu, y_do2, y_fc2, y_out = 2.86, 2.30, 1.74, 1.18, y_add

    pad_x, pad_y = 0.12, 0.13
    ax.add_patch(FancyBboxPatch(
        (x_r - w_r / 2 - pad_x, y_fc2 - h_std / 2 - pad_y),
        w_r + 2 * pad_x,
        (y_fc1 - y_fc2) + h_std + 2 * pad_y,
        boxstyle="round,pad=0.015,rounding_size=0.09",
        facecolor=C["head_bg"], edgecolor=C["head_edge"],
        linestyle=(0, (3, 2)), linewidth=1.1, zorder=1,
    ))

    add_box(x_r, y_ext, w_r, h_std, r"Último paso (día $t$)", C["extract"])
    add_box(x_r, y_fc1, w_r, h_std, "Lineal (FC1)", C["fc"])
    add_box(x_r, y_elu, w_r, h_std, "Activación ELU", C["act"])
    add_box(x_r, y_do2, w_r, h_std, "Dropout 2", C["dropout"])
    add_box(x_r, y_fc2, w_r, h_std, "Lineal (FC2)", C["fc"])
    add_box(x_r, y_out, w_r, h_std, "Salida", C["io"])

    add_arrow((x_r, y_ext - h_std / 2), (x_r, y_fc1 + h_std / 2))
    add_arrow((x_r, y_fc1 - h_std / 2), (x_r, y_elu + h_std / 2))
    add_arrow((x_r, y_elu - h_std / 2), (x_r, y_do2 + h_std / 2))
    add_arrow((x_r, y_do2 - h_std / 2), (x_r, y_fc2 + h_std / 2))
    add_arrow((x_r, y_fc2 - h_std / 2), (x_r, y_out + h_std / 2))

    add_dim(x_r + 0.08, (y_ext + y_fc1) / 2, "[batch, hidden]")
    add_dim(x_r + 0.08, (y_fc2 + y_out) / 2, "[batch]")
    return fig


with plt.rc_context(DIAGRAM_RC):
    fig = draw_lstm()
    fig.savefig(FIG_DIR / "arquitectura_lstm.pdf")
plt.show()
