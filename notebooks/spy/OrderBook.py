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
# # Libro de órdenes (LOB) del SPY
#
# Este cuaderno estudia la **estructura de liquidez** del libro de órdenes
# del SPY a partir de los ficheros LOBSTER (`message` + `orderbook`):
#
# 1. **distribución relativa**: perfil de volumen pasivo en bins de distancia
#    al precio medio (perfil de liquidez alrededor del punto medio);
# 2. **distribución absoluta**: perfil de volumen pasivo en bins de precio
#    (dónde se acumula la liquidez en valor);
# 3. **snapshot único**: estado del libro en un instante concreto;
# 4. **precio medio promediado**: evolución del precio medio del libro en
#    buckets de tiempo.
#
# La lógica reutilizable vive en `src/thesis/orderbook.py` (carga,
# validación y extracción); aquí el cuaderno la coordina, **por etapas**, con
# una celda por sección del pipeline.  El **código de dibujo de las figuras
# está en el cuaderno** (no en el paquete), para poder retocar etiquetas,
# escalas y colores sin tocar `src/thesis/`.
#
# .. warning::
#    Las extracciones son **descriptivas**, no señales: no hay train/test que
#    contaminar.  El snapshot es **retrospectivo** por diseño (elige la
#    primera fila con ``Time >= target_time``), así que no debe usarse como
#    variable de entrada de un modelo.
#
# .. note::
#    Los ficheros LOBSTER **no** están en el repositorio.  Hay que apuntar
#    ``THESIS_LOB_DIR`` al directorio que los contiene (por defecto
#    ``data/OrderBook``):
#
#    .. code-block:: bash
#
#       export THESIS_LOB_DIR=/ruta/a/OrderBook

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _find_repo_root(start: Path) -> Path:
    """Raíz del repositorio: el directorio que contiene ``src/thesis``."""
    for candidate in (start, *start.parents):
        if (candidate / "src" / "thesis").is_dir():
            return candidate
    return start


# Permite ejecutar el archivo desde ``notebooks/`` o desde ``notebooks/spy/``.
# (``__file__`` no existe dentro del kernel de Jupyter: ahí se usa el
# directorio de trabajo actual.)
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _find_repo_root(_HERE)  # raíz del repositorio (``thesis/``)
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.config import DATA_DIR
from thesis.orderbook import (
    aggregate_volume_bins,
    extract_absolute_lob_distribution,
    extract_averaged_mid_price,
    extract_relative_lob_distribution,
    extract_single_snapshot,
    lob_file_paths,
)

#: Directorio de figuras (anclado a la raíz del repositorio, no al CWD).
OUTPUT_DIR = _ROOT / "outputs"

# %% [markdown]
# ## Configuración
#
# Todos los parámetros del experimento viven aquí: se cambian **en sitio** y
# se re-ejecuta la celda de la etapa correspondiente.
#
# **Unidades**: los precios de LOBSTER están en *ticks* (enteros) y se
# dividen entre 10 000 para obtener USD; el tiempo va en segundos desde
# medianoche (puede superar 86 400 en *after-hours*); el volumen está en
# acciones.

# %%
CONFIG: dict[str, Any] = {
    # --- Ficheros LOBSTER ---
    "symbol": "SPY",
    "date": "2012-06-21",
    "start": "34200000",  # 09:30:00
    "end": "37800000",  # 10:30:00
    "levels": 50,
    # Directorio con los ficheros; ``THESIS_LOB_DIR`` tiene prioridad.
    "directory": Path(os.environ.get("THESIS_LOB_DIR", Path(DATA_DIR) / "OrderBook")),
    # --- Ventana de análisis (segundos desde medianoche) ---
    "start_time": 34200.0,
    "end_time": 37799.0,
    # --- Agregación ---
    "bin_width": 0.02,  # USD (distribución relativa)
    "absolute_bin_width": 0.05,  # USD (distribución absoluta, re-agregada)
    "snapshot_bin_width": 0.05,  # USD (snapshot)
    "mid_interval": 5.0,  # segundos (precio medio promediado)
    # --- Ejes (None = automático) ---
    "relative_xlim": (-0.6, 0.6),  # USD alrededor del precio medio
    "absolute_xlim": (134.0, 136.5),  # USD (rango de precio del activo)
    "snapshot_half_width": 0.6,  # USD alrededor del precio medio
    # --- Figuras ---
    "show_figure": True,
    "save_figures": True,
}

MESSAGE_PATH, ORDERBOOK_PATH = lob_file_paths(
    symbol=CONFIG["symbol"],
    date=CONFIG["date"],
    start=CONFIG["start"],
    end=CONFIG["end"],
    levels=CONFIG["levels"],
    directory=CONFIG["directory"],
)

print(f"Mensajes:  {MESSAGE_PATH}")
print(f"Orderbook: {ORDERBOOK_PATH}")
for path in (MESSAGE_PATH, ORDERBOOK_PATH):
    if not path.exists():
        print(
            f"AVISO: no existe {path}. Define THESIS_LOB_DIR o ajusta "
            "CONFIG['directory'] antes de ejecutar las etapas."
        )

# %% [markdown]
# ## Etapa 1: distribución relativa del LOB
#
# Perfil de liquidez **relativo al precio medio**: el volumen pasivo se
# acumula en bins de distancia $\Delta P = P - P_{mid}$ y se promedia por
# snapshot (no por mensaje).  Los niveles vacíos (centinelas de LOBSTER) se
# descartan antes de escalar los precios.

# %%
if __name__ == "__main__":
    bids_binned, asks_binned, max_range = extract_relative_lob_distribution(
        MESSAGE_PATH,
        ORDERBOOK_PATH,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        levels=CONFIG["levels"],
        bin_width=CONFIG["bin_width"],
    )

    print(f"Rango relativo: ±{max_range:.4f} USD")
    print(f"Bins con volumen (bid): {len(bids_binned)}")
    print(f"Bins con volumen (ask): {len(asks_binned)}")

    relative_fig, relative_ax = plt.subplots(figsize=(6, 3), dpi=300)

    bar_width = CONFIG["bin_width"] * 0.85
    relative_ax.bar(
        bids_binned["BinMid"].astype(float),
        bids_binned["Volume"],
        width=bar_width,
        color="forestgreen",
        alpha=0.75,
        edgecolor="black",
        label="Compra",
    )
    relative_ax.bar(
        asks_binned["BinMid"].astype(float),
        asks_binned["Volume"],
        width=bar_width,
        color="firebrick",
        alpha=0.75,
        edgecolor="black",
        label="Venta",
    )
    relative_ax.axvline(
        0, color="black", linestyle="--", linewidth=1.2, label="Precio Medio"
    )
    relative_ax.set_xlabel(r"Distancia al Precio Medio $\Delta P$ (USD)")
    relative_ax.set_ylabel("Volumen pasivo promedio (Acciones)")
    if CONFIG["relative_xlim"] is not None:
        relative_ax.set_xlim(*CONFIG["relative_xlim"])
    relative_ax.legend()
    relative_fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        relative_fig.savefig(
            OUTPUT_DIR / "spy_lob_relative_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ## Etapa 2: distribución absoluta del LOB
#
# El mismo perfil, pero en **precio absoluto**: dónde se acumula la
# liquidez en valor.  Los bins se re-agregan a `absolute_bin_width` para que
# la figura no tenga una barra por píxel.

# %%
if __name__ == "__main__":
    absolute_bids, absolute_asks, price_range = extract_absolute_lob_distribution(
        MESSAGE_PATH,
        ORDERBOOK_PATH,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        levels=CONFIG["levels"],
        bin_width=CONFIG["bin_width"],
    )

    print(f"Rango de precio: {price_range[0]:.4f} — {price_range[1]:.4f} USD")

    # Re-agregación a un bin más ancho (mismo helper que el snapshot).
    absolute_bids_agg = aggregate_volume_bins(
        absolute_bids["BinMid"], absolute_bids["Volume"], CONFIG["absolute_bin_width"]
    )
    absolute_asks_agg = aggregate_volume_bins(
        absolute_asks["BinMid"], absolute_asks["Volume"], CONFIG["absolute_bin_width"]
    )

    absolute_fig, absolute_ax = plt.subplots(figsize=(6, 3), dpi=300)

    bar_width = CONFIG["absolute_bin_width"] * 0.85
    if not absolute_bids_agg.empty:
        absolute_ax.bar(
            absolute_bids_agg["BinMid"],
            absolute_bids_agg["Volume"],
            width=bar_width,
            color="forestgreen",
            alpha=0.75,
            edgecolor="black",
            label="Compra",
        )
    if not absolute_asks_agg.empty:
        absolute_ax.bar(
            absolute_asks_agg["BinMid"],
            absolute_asks_agg["Volume"],
            width=bar_width,
            color="firebrick",
            alpha=0.75,
            edgecolor="black",
            label="Venta",
        )
    absolute_ax.set_xlabel(f"Precio (USD) - Bins de ${CONFIG['absolute_bin_width']}")
    absolute_ax.set_ylabel("Volumen pasivo promedio (Acciones)")
    if CONFIG["absolute_xlim"] is not None:
        absolute_ax.set_xlim(*CONFIG["absolute_xlim"])
    absolute_ax.legend()
    absolute_fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        absolute_fig.savefig(
            OUTPUT_DIR / "spy_lob_absolute_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ## Etapa 3: snapshot único
#
# Estado del libro en el primer instante con `Time >= target_time`.  Es
# **retrospectivo**: el snapshot corresponde a un momento *posterior* al
# objetivo (la fila 34202.0 del fichero sintético es exactamente el primer
# mensaje en o después del objetivo).

# %%
if __name__ == "__main__":
    target_time = CONFIG["start_time"]
    snap_bids, snap_asks, actual_time = extract_single_snapshot(
        MESSAGE_PATH,
        ORDERBOOK_PATH,
        target_time=target_time,
        levels=CONFIG["levels"],
    )

    print(f"t objetivo: {target_time:.1f} s -> t real: {actual_time:.1f} s")
    print(f"Niveles con precio válido: {len(snap_bids)} bid / {len(snap_asks)} ask")

    snapshot_fig, snapshot_ax = plt.subplots(figsize=(6, 3), dpi=300)

    snapshot_bin_width = CONFIG["snapshot_bin_width"]
    snap_bids_agg = aggregate_volume_bins(
        snap_bids["Price"], snap_bids["Volume"], snapshot_bin_width
    )
    snap_asks_agg = aggregate_volume_bins(
        snap_asks["Price"], snap_asks["Volume"], snapshot_bin_width
    )

    bar_width = snapshot_bin_width * 0.85
    if not snap_bids_agg.empty:
        snapshot_ax.bar(
            snap_bids_agg["BinMid"],
            snap_bids_agg["Volume"],
            width=bar_width,
            color="forestgreen",
            alpha=0.75,
            edgecolor="black",
            label="Compra",
        )
    if not snap_asks_agg.empty:
        snapshot_ax.bar(
            snap_asks_agg["BinMid"],
            snap_asks_agg["Volume"],
            width=bar_width,
            color="firebrick",
            alpha=0.75,
            edgecolor="black",
            label="Venta",
        )

    # Precio medio a partir del mejor bid y del mejor ask.
    if not snap_bids.empty and not snap_asks.empty:
        best_bid = snap_bids["Price"].max()
        best_ask = snap_asks["Price"].min()
        mid_price = (best_bid + best_ask) / 2.0
        snapshot_ax.axvline(
            mid_price,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=rf"$P_{{mid}}$ = {mid_price:.2f}",
        )
        half_width = CONFIG["snapshot_half_width"]
        if half_width is not None:
            snapshot_ax.set_xlim(mid_price - half_width, mid_price + half_width)

    snapshot_ax.set_xlabel("Precio (USD)")
    snapshot_ax.set_ylabel("Volumen Pasivo (Acciones)")
    snapshot_ax.legend()
    snapshot_fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_fig.savefig(
            OUTPUT_DIR / "spy_lob_snapshot.png", dpi=300, bbox_inches="tight"
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ## Etapa 4: precio medio promediado
#
# El precio medio del libro (mejor bid y mejor ask) se promedia en buckets
# de `mid_interval` segundos.  El promedio es por **snapshots**, no por
# duración de la ventana.

# %%
if __name__ == "__main__":
    df_avg = extract_averaged_mid_price(
        MESSAGE_PATH,
        ORDERBOOK_PATH,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        interval=CONFIG["mid_interval"],
    )

    print(f"Buckets: {len(df_avg)} (cada {CONFIG['mid_interval']:.0f} s)")
    print(
        f"Precio medio: {df_avg['MidPrice'].min():.4f} — "
        f"{df_avg['MidPrice'].max():.4f} USD"
    )

    mid_fig, mid_ax = plt.subplots(figsize=(6, 3), dpi=300)

    mid_ax.plot(
        df_avg["TimeBin"] - CONFIG["start_time"],
        df_avg["MidPrice"],
        color="navy",
        linewidth=1.0,
    )
    mid_ax.set_xlabel("Tiempo (s)")
    mid_ax.set_ylabel("Precio Medio (USD)")

    # Márgenes para un eje Y legible (2 % del rango; mínimo de 0,05 USD).
    p_min, p_max = df_avg["MidPrice"].min(), df_avg["MidPrice"].max()
    margin = (p_max - p_min) * 0.02
    if margin == 0:
        margin = 0.05
    mid_ax.set_ylim(p_min - margin, p_max + margin)
    mid_fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        mid_fig.savefig(
            OUTPUT_DIR / "spy_lob_mid_price.png", dpi=300, bbox_inches="tight"
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %%
# %%
# ## Publicación de métricas en el ledger
#
# Celda final que envía las métricas disponibles al ledger estandarizado de la
# tesis.  No recalcula resultados; captura las variables del notebook.

if __name__ == "__main__":
    from thesis.results import publish

    _payload = {}
    for _key in ("CONFIG", "config", "metrics", "results", "summary", "backtest"):
        if _key in globals():
            _payload[_key.lower()] = globals()[_key]

    publish(
        experiment_id="spy/orderbook",
        payload=_payload,
        chapter="SPY",
    )

# %%
