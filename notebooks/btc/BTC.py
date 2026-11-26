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
# # Análisis de BTC: power-law y ciclos de mercado
#
# Este cuaderno estudia el precio de BTC con dos herramientas descriptivas:
# una **regresión cuantílica power-law** sobre el tiempo desde la génesis y
# una **detección de ciclos** (picos y valles macro) sobre el precio
# logarítmico.
#
# La lógica reutilizable vive en `src/thesis/btc.py` (datos, power-law y
# ciclos); aquí el cuaderno la coordina, **por etapas**, con una celda por
# sección del pipeline:
#
# 1. **datos**: descarga (o caché) del OHLCV diario de Bitstamp;
# 2. **power-law**: preparación del frame y regresión cuantílica;
# 3. **figura**: el código de dibujo va **escrito en el cuaderno**, sin
#    funciones de apoyo, para poder retocar etiquetas, escalas y colores;
#    además, una **proyección semilog** de los cuantiles hacia el futuro
#    (exploratoria, no es una señal de trading);
# 4. **ciclos**: picos y valles macro;
# 5. **curva suavizada y picos**: precio dettrendizado por la banda del
#    cuantil 10, suavizado con un promedio móvil y marcado de los máximos
#    locales detectados sobre la curva.
#
# Cada etapa tiene su propia celda: se puede cambiar un parámetro o una
# figura y re-ejecutar **sólo** la celda afectada, sin volver a descargar los
# datos.  El archivo también se ejecuta como script
# (`python notebooks/btc/BTC.py`), en cuyo caso las celdas corren en orden.

# %%
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Permite ejecutar el archivo desde ``notebooks/`` o desde ``notebooks/btc/``.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # raíz del repositorio (``thesis/``)
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.btc import (
    BTC_BOTTOMS,
    BTC_GENESIS,
    BTC_HALVINGS,
    BTC_PEAKS,
    BTC_QUANTILES,
    detect_market_cycles,
    detrend_power_law_quantile,
    fetch_btc_ohlcv,
    find_local_extrema,
    fit_power_law_quantiles,
    prepare_power_law_frame,
    select_extrema,
    smooth_cycle_residual,
)
from thesis.config import DATA_DIR

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

#: Figuras de este cuaderno: ``figures/BTC/`` en la raíz del repositorio.
FIG_DIR = _ROOT / "figures" / "BTC"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Configuración
#
# Todos los parámetros del experimento viven aquí: se cambian **en sitio** y
# se re-ejecuta la celda de la etapa correspondiente.

# %%
CONFIG: dict[str, Any] = {
    # Datos
    "symbol": "btcusd",
    "cache_dir": DATA_DIR,
    "refresh": False,
    # Power-law
    "quantiles": BTC_QUANTILES,
    # Anotaciones de la figura (líneas de referencia, no señales)
    "halvings": BTC_HALVINGS,
    "peaks": BTC_PEAKS,
    "bottoms": BTC_BOTTOMS,
    # Ciclos
    "distance": 200,
    "prominence": 0.3,
    # Curva suavizada y picos (Etapa 5)
    "peak_selection": {
        # Cuantil de la banda que se resta al precio (el "suelo" del power-law).
        "quantile": 0.10,
        # Ventana (días) del promedio móvil aplicado al precio dettrendizado.
        # La ventana es centrada; los picos se detectan sobre esa media.
        "smooth_window": 365,
        # Filtro de extremos locales candidatos (días y log10 de prominencia).
        "distance": 200,
        "prominence": 0.05,
        # Picos a destacar: id de la tabla de candidatos ("P1", "P2", ...).
        # Vacío = no se marca ningún pico.
        "selected_peaks": ["P1", "P2", "P3", "P4"],
    },
    # Proyección futura (Etapa 6): escala semilog con extrapolación.
    "prediction": {
        # Fecha de inicio de la ventana histórica mostrada en la figura.
        "start_date": "2018-01-01",
        # Horizonte de la proyección (~2 años hacia delante).
        "horizon_days": 730,
        # Días desde el génesis para imprimir los valores de los cuantiles
        # en la Etapa 2. None = no imprimir (la predicción de la Etapa 6 se
        # calcula sola a partir de los picos).
        "query_days_since_genesis": None,
        # Cuantiles que delimitan los sectores del gráfico arcoíris (de abajo
        # arriba); cada par consecutivo define un sector.
        "rainbow_quantiles": (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    },
    # Regresión sobre la tendencia reciente (Etapa 7): mismas regresiones
    # cuantílicas, ajustadas sólo con los datos desde esta fecha.
    "recent_fit": {
        "start_date": "2018-01-01",
        # Cuantiles cuyas pendientes se comparan con las del ajuste completo.
        "quantiles": (0.10, 0.50, 0.90),
    },
    # Figura
    "show_figure": True,
}

# %% [markdown]
# ## Etapa 1: datos
#
# Descarga el OHLCV diario de Bitstamp o lo lee de la caché
# `data/btc_daily_ohlcv.csv`.  Es la única celda que toca la red: al
# re-ejecutar las etapas siguientes no se vuelve a descargar nada.


# %%
def load_ohlcv(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Descarga (o lee de caché) el OHLCV diario de BTC.

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.

    Returns
    -------
    pd.DataFrame
        OHLCV diario validado, indexado por fecha.
    """
    cfg = {**CONFIG, **(config or {})}
    btc_df = fetch_btc_ohlcv(
        symbol=cfg["symbol"],
        cache_dir=Path(cfg["cache_dir"]),
        refresh=cfg["refresh"],
    )

    print(btc_df.head())
    print("\nTotal days fetched:", len(btc_df))
    return btc_df


# %% [markdown]
# ## Etapa 2: power-law
#
# Prepara `days`, `log_days` y `log_price` y ajusta la regresión cuantílica
# `log_price ~ log_days`.  Re-ejecutar esta celda con otros cuantiles
# actualiza la tabla de exponentes sin volver a descargar datos.


# %%
def print_quantile_values_at_days(
    fit: dict[float, dict[str, Any]],
    days_since_genesis: int,
) -> None:
    """Imprime la fecha y los valores ajustados para un día desde el génesis."""
    if isinstance(days_since_genesis, bool) or days_since_genesis <= 0:
        raise ValueError("days_since_genesis debe ser un entero positivo.")

    date = pd.Timestamp(BTC_GENESIS) + pd.Timedelta(days=days_since_genesis)
    print(
        f"\nValores de la regresión cuantílica para "
        f"{days_since_genesis} días desde el génesis ({date.date()}):"
    )
    for q, res in fit.items():
        params = res["params"]
        log_price = params["Intercept"] + params["log_days"] * np.log10(
            days_since_genesis
        )
        price = 10**log_price
        print(f"q={q:.2f} | Fecha: {date.date()} | Valor: {price:.4f}")


def fit_power_law(
    btc_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]]]:
    """Ajusta la regresión cuantílica power-law sobre el OHLCV.

    Parameters
    ----------
    btc_df : pd.DataFrame
        OHLCV diario de BTC.
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        El frame preparado y el diccionario de cuantiles
        ``{q: {"params": ..., "prediction": ...}}``.
    """
    cfg = {**CONFIG, **(config or {})}
    reg_df = prepare_power_law_frame(btc_df)
    fit = fit_power_law_quantiles(reg_df, quantiles=cfg["quantiles"])

    print("\nPendientes de las regresiones cuantílicas (power-law):")
    for q, res in fit.items():
        params = res["params"]
        print(
            f"q={q:.2f} | Pendiente (Beta): {params['log_days']:.4f} | "
            f"Intercepto (Alpha): {params['Intercept']:.4f}"
        )
    query_days = cfg["prediction"]["query_days_since_genesis"]
    if query_days is not None:
        print_quantile_values_at_days(fit, query_days)
    return reg_df, fit


# %% [markdown]
# ## Etapa 3: figura
#
# El código de dibujo está **escrito en el cuaderno**, sin funciones de
# apoyo: no se llama a `thesis.btc.plot_btc_quantreg` ni a ningún helper
# propio.  Así, la figura (etiquetas, rangos, colores, anotaciones) se
# edita y se re-ejecuta aquí mismo, sin tocar `src/thesis/btc.py`.
#
# Se dibujan tres gráficos independientes: la evolución temporal en escala
# lineal, el eje temporal en escala logarítmica (con las rectas de los
# cuantiles y las anotaciones de halvings, picos y valles) y los días desde
# la génesis en escala log-log.  El código está en la celda de ejecución
# "Etapa 3: figura", al final del cuaderno, para que en modo script las
# etapas se ejecuten en orden.


# %% [markdown]
# ## Etapa 4: ciclos
#
# Detecta picos y valles macro sobre `log10(close)` con `find_peaks`.  El
# pico de Mt. Gox (2011-06-08), anterior a los datos de Bitstamp, se inyecta
# a mano.


# %%
def find_cycles(
    btc_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detecta los picos y valles macro del precio de BTC.

    Parameters
    ----------
    btc_df : pd.DataFrame
        OHLCV diario de BTC.
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(peaks, bottoms)`` con las distancias al pico y al valle previos.
    """
    cfg = {**CONFIG, **(config or {})}
    return detect_market_cycles(
        btc_df,
        distance=cfg["distance"],
        prominence=cfg["prominence"],
    )


# %% [markdown]
# ## Ejecución por etapas
#
# Cada etapa se invoca en su propia celda y deja su resultado en el espacio de
# nombres, de modo que se puede re-ejecutar **sólo** la celda que interesa.
#
# Las llamadas van bajo `if __name__ == "__main__":` para que el archivo pueda
# importarse como módulo sin descargar nada; en Jupyter la condición es
# verdadera y las celdas se ejecutan de forma interactiva.

# %% [markdown]
# ### Etapa 1: datos

# %%
if __name__ == "__main__":
    btc_df = load_ohlcv(CONFIG)

# %% [markdown]
# ### Etapa 2: power-law
#
# Cambiar `CONFIG["quantiles"]` y re-ejecutar **esta celda** actualiza los
# exponentes y las rectas sin volver a descargar los datos.

# %%
if __name__ == "__main__":
    reg_df, fit = fit_power_law(btc_df, CONFIG)

# %% [markdown]
# ### Etapa 3: figura
#
# Código de dibujo **directo**: no hay funciones intermedias, así que cada
# línea se puede retocar y volver a ejecutar.  Usa `reg_df` y `fit` de la
# etapa 2 y `CONFIG` para las anotaciones.

# %%
if __name__ == "__main__":
    # Estilos de las rectas de cuantiles (editables aquí).
    quantile_colors: dict[float, str] = {0.10: "green", 0.50: "blue", 0.90: "red"}
    quantile_styles: dict[float, str] = {0.10: "--", 0.50: "-.", 0.90: "--"}

    # Anotaciones: fechas -> timestamp (líneas de referencia, no señales).
    halving_dates = [pd.Timestamp(str(d).strip()) for d in CONFIG["halvings"]]
    peak_dates = [pd.Timestamp(str(d).strip()) for d in CONFIG["peaks"]]
    bottom_dates = [pd.Timestamp(str(d).strip()) for d in CONFIG["bottoms"]]

    # --- Figura 1: evolución temporal en escala lineal ---
    linear_fig, ax0 = plt.subplots()
    ax0.plot(
        reg_df.index,
        reg_df["close"],
        # label="Cierre de BTC",
        color="black",
        alpha=0.7,
        linewidth=0.8,
    )
    # ax0.set_title("Evolución del precio de Bitcoin")
    ax0.set_xlabel("Fecha")
    ax0.set_ylabel("Precio (USD)")
    ax0.legend()
    ax0.grid(True, ls="--", alpha=0.4)

    # --- Figura 2: eje temporal logarítmico con anotaciones ---
    temporal_fig, ax1 = plt.subplots()
    ax1.plot(
        reg_df.index,
        reg_df["close"],
        label="Cierre de BTC",
        color="black",
        alpha=0.6,
        linewidth=1,
    )
    for q, res in fit.items():
        ax1.plot(
            reg_df.index,
            res["prediction"],
            label=rf"Reg. $p\cdot 100={int(q * 100)}$",
            color=quantile_colors.get(q, "gray"),
            linestyle=quantile_styles.get(q, "--"),
            linewidth=2,
        )

    for i, halving_date in enumerate(halving_dates):
        ax1.axvline(
            halving_date,
            color="purple",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
            label="Halving" if i == 0 else None,
        )

    for i, peak_date in enumerate(peak_dates):
        ax1.axvline(
            peak_date,
            color="green",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label="Pico" if i == 0 else None,
        )

    for i, bottom_date in enumerate(bottom_dates):
        ax1.axvline(
            bottom_date,
            color="red",
            linestyle="-.",
            alpha=0.5,
            linewidth=1.5,
            label="Valle" if i == 0 else None,
        )

    ax1.set_yscale("log")
    # ax1.set_title("Dinámica del precio de BTC — regresión cuantílica power-law")
    ax1.set_xlabel("Días desde la génesis")
    ax1.set_ylabel("Precio (USD)")
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # --- Figura 3: power-law log-log (días desde la génesis) ---
    power_law_fig, ax2 = plt.subplots()
    ax2.plot(
        reg_df["days"],
        reg_df["close"],
        label="BTC",
        color="black",
        alpha=0.7,
        linewidth=1,
    )
    for q, res in fit.items():
        ax2.plot(
            reg_df["days"],
            res["prediction"],
            label=rf"Reg. $p\cdot 100={int(q * 100)}$",
            color=quantile_colors.get(q, "gray"),
            linestyle=quantile_styles.get(q, "--"),
            linewidth=1,
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    # ax2.set_title("Dinámica del precio de BTC — regresión cuantílica power-law")
    ax2.set_xlabel("Días desde el primer bloque")
    ax2.set_ylabel("Precio (USD)")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    figures = (linear_fig, temporal_fig, power_law_fig)
    linear_fig.savefig(FIG_DIR / "btc_price_linear.pdf")
    temporal_fig.savefig(FIG_DIR / "btc_quantreg_temporal_log.pdf")
    power_law_fig.savefig(FIG_DIR / "btc_quantreg_power_law.pdf")

    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ### Etapa 4: ciclos
#
# Cambiar `CONFIG["distance"]` o `CONFIG["prominence"]` y re-ejecutar **esta
# celda** recalcula los ciclos macro.

# %%
if __name__ == "__main__":
    peaks_df, bottoms_df = find_cycles(btc_df, CONFIG)


# %% [markdown]
# ### Etapa 5: curva suavizada y picos
#
# 1. **banda q10**: se toma el precio en escala log-log y se ajusta (o se
#    reutiliza) la regresión cuantílica del 10;
# 2. **precio dettrendizado**: se resta esa banda, de modo que los **mínimos
#    locales quedan cerca de cero** (`resid_log`);
# 3. **media móvil**: se aplica un promedio con ventana (`smooth_window`
#    días) al precio dettrendizado (`resid_ma`): los máximos locales se
#    detectan sobre esa serie suavizada, no sobre el ruido diario.  La
#    ventana es **centrada** (usa datos posteriores), de modo que los
#    extremos quedan alineados con los giros reales del precio; es un uso
#    descriptivo, no una señal causal;
# 4. **candidatos**: se listan los máximos locales con un `id` numerado
#    (`P1`, `P2`, ...);
# 5. **elección**: se escriben los ids a destacar en
#    `CONFIG["peak_selection"]["selected_peaks"]`;
# 6. **figura**: la curva suavizada con los picos marcados **en su valor
#    real** (`resid_ma` en la fecha del pico).
#
# El código de dibujo va **en línea**, como en la Etapa 3.

# %%
if __name__ == "__main__":
    peak_cfg = CONFIG["peak_selection"]

    # --- 1-2: banda del cuantil 10 y precio dettrendizado ---
    cycle_df = detrend_power_law_quantile(
        reg_df,
        fit,
        quantile=peak_cfg["quantile"],
    )

    # --- 3: promedio móvil centrado del precio dettrendizado ---
    # La ventana se centra en cada fecha (usa datos posteriores): los picos
    # quedan alineados con los giros reales del precio.  Con
    # `min_periods` reducido a la mitad de la ventana, la serie suavizada
    # cubre todo el histórico, incluidos los bordes.
    smooth_window = int(peak_cfg["smooth_window"])
    cycle_df["resid_ma"] = smooth_cycle_residual(
        cycle_df["resid_log"],
        window=smooth_window,
        min_periods=max(1, smooth_window // 2),
        centered=True,
    )

    # --- 4: máximos locales candidatos (sobre la serie suavizada) ---
    _, peaks_cand = find_local_extrema(
        cycle_df["resid_ma"],
        distance=peak_cfg["distance"],
        prominence=peak_cfg["prominence"],
    )
    peaks_cand = peaks_cand.join(cycle_df[["close", "days"]])

    # --- tabla numerada con los máximos detectados ---
    print(
        f"\n=== MÁXIMOS LOCALES DETECTADOS (media móvil centrada de "
        f"{smooth_window} días sobre el residuo frente a la banda q10) ==="
    )
    print(f"{'id':>4}  {'fecha':<12} {'resid_ma':>10} {'precio':>12} {'días':>7}")
    for _, row in peaks_cand.iterrows():
        print(
            f"{row['id']:>4}  {row.name.strftime('%Y-%m-%d'):<12} "
            f"{row['value']:>10.3f} {row['close']:>12,.0f} {row['days']:>7.0f}"
        )

    print(
        "\nLos ids a destacar van en\n"
        "  CONFIG['peak_selection']['selected_peaks']\n"
        "y vuelve a ejecutar esta celda."
    )

    # --- 5: picos a destacar (ids de la tabla anterior) ---
    peak_sel = select_extrema(peaks_cand, peak_cfg["selected_peaks"])

    print("\nPicos destacados:")
    for _, row in peak_sel.iterrows():
        print(
            f"  {row['id']}  {row.name.strftime('%Y-%m-%d')}  "
            f"resid_ma={row['value']:.3f}  close={row['close']:,.0f}"
        )

    # --- Figura 4: precio log-log con banda q10 y precio suavizado con sus picos ---
    cycle_fig, (ax3, ax4) = plt.subplots(
        2, 1, figsize=(W, 0.8 * W), gridspec_kw={"height_ratios": [1, 1.5]}
    )

    # Panel A: precio en escala log-log con la banda q10 destacada.
    ax3.plot(
        cycle_df["days"],
        cycle_df["close"],
        label="BTC",
        color="black",
        alpha=0.6,
        linewidth=1,
    )
    ax3.plot(
        cycle_df["days"],
        cycle_df["band"],
        label=r"$Q_{10}$ Reg",
        color="green",
        linewidth=2,
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    # ax3.set_title("BTC en escala log-log con la banda del cuantil 10")
    ax3.set_xlabel("Días desde el primer bloque")
    ax3.set_ylabel("Precio (USD)")
    ax3.text(0.02, 0.95, "a", transform=ax3.transAxes, va="top")
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.4)

    # Panel B: precio dettrendizado, su media móvil y los picos marcados.
    ax4.plot(
        cycle_df.index,
        cycle_df["resid_log"],
        label="Precio dettrendizado (log10)",
        color="black",
        alpha=0.3,
        linewidth=0.8,
    )
    ax4.plot(
        cycle_df.index,
        cycle_df["resid_ma"],
        label=f"Media móvil centrada ({smooth_window} días)",
        color="blue",
        alpha=0.9,
        linewidth=1.6,
    )
    for i, halving_date in enumerate(halving_dates):
        ax4.axvline(
            halving_date,
            color="purple",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
            label="Halving" if i == 0 else None,
        )
    ax4.axhline(0.0, color="gray", linestyle=":", linewidth=1)

    # # Etiquetas de los picos, en el valor real de la curva suavizada.

    # for stamp, value in peak_sel["value"].items():
    #     ax4.annotate(
    #         stamp.strftime("%Y-%m"),
    #         (stamp, value),
    #         textcoords="offset points",
    #         xytext=(0, 8),
    #         ha="center",
    #         fontsize=7,
    #         color="green",
    #     )

    # ax4.set_title(
    #     f"Precio dettrendizado y media móvil centrada de {smooth_window} días "
    #     "con los picos detectados"
    # )
    ax4.set_xlabel("Fecha")
    ax4.set_ylabel(r"log(BTC/$Q_{10}$)")
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax4.text(0.02, 0.95, "b", transform=ax4.transAxes, va="top")
    ax4.grid(True, ls="--", alpha=0.4)
    cycle_fig.savefig(FIG_DIR / "btc_market_cycles.pdf")

    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")


# %% [markdown]
# ### Etapa 6: proyección a futuro (gráfico arcoíris)
#
# Figura de BTC desde `CONFIG["prediction"]["start_date"]`
# en escala **semilog** (eje Y logarítmico, eje X lineal en fechas) con un
# **gráfico arcoíris**: los sectores entre rectas consecutivas de la
# regresión cuantílica power-law (`log10(price) = α + β·log10(days)`,
# cuantiles `CONFIG["prediction"]["rainbow_quantiles"]`) se pintan con una
# escala de color progresiva, por debajo del precio y semitransparentes, de
# modo que se lee en qué sector está el precio en cada momento.
#
# Los sectores se extienden hasta pasada la fecha de la predicción del
# capítulo, que se marca con el rango de precio previsto.  Todo se calcula a
# partir de los picos destacados en la Etapa 5: la fecha es el último pico más
# el promedio de las tres últimas distancias entre picos; el extremo inferior
# es el valor de `Q10` (piso) en esa fecha; el superior, el último pico
# multiplicado por su aumento respecto del pico anterior (el aumento de
# pico a pico se desacelera, así que el próximo sería menor).  Una
# línea vertical separa los datos observados de la extrapolación, que es
# **descriptiva**: **no es una señal de trading** ni un pronóstico riguroso.
#
# Depende de `reg_df` (Etapa 2) y `peak_sel` (Etapa 5); puede re-ejecutarse **sólo** esta celda.

# %%
if __name__ == "__main__":
    pred_cfg = CONFIG["prediction"]
    rainbow_q = tuple(sorted(pred_cfg["rainbow_quantiles"]))

    # --- Fechas y días de corte ---
    last_date = reg_df.index.max()
    last_days = int(reg_df["days"].max())
    cutoff = pd.Timestamp(pred_cfg["start_date"])
    # --- Predicción del próximo máximo, calculada a partir de los picos ---
    peak_dates_sel = peak_sel.index
    peak_closes = peak_sel["close"].to_numpy(dtype=float)
    mean_gap = float(np.mean(np.diff(peak_dates_sel[-4:]).astype("timedelta64[D]").astype(int)))
    target_date = peak_dates_sel[-1] + pd.Timedelta(days=round(mean_gap))
    target_days = int((target_date - pd.Timestamp(BTC_GENESIS)).days)
    last_ratio = peak_closes[-1] / peak_closes[-2]
    p10 = fit[0.10]["params"]
    t_min = float(10 ** (p10["Intercept"] + p10["log_days"] * np.log10(target_days)))
    t_max = float(last_ratio * peak_closes[-1])
    print(
        f"\nPredicción del próximo máximo: {target_date:%Y-%m-%d} "
        f"(día {target_days}; distancia media {mean_gap:.0f} días)\n"
        f"  mínimo (Q10 en esa fecha): {t_min:,.0f} USD\n"
        f"  máximo (último pico x{last_ratio:.2f}): {t_max:,.0f} USD"
    )

    # Eje temporal completo (histórico + futuro) en días enteros desde el
    # génesis: el futuro llega al horizonte configurado o, si es mayor, hasta
    # medio año después de la fecha de la predicción.
    end_days = max(
        last_days + int(pred_cfg["horizon_days"]),
        target_days + 180,
    )
    start_days = int((cutoff - pd.Timestamp(BTC_GENESIS)).days)
    grid_days = np.arange(start_days, end_days + 1)  # paso entero
    grid_dates = pd.Timestamp(BTC_GENESIS) + pd.to_timedelta(grid_days, unit="D")

    # --- Ventana histórica (in-sample) ---
    hist = reg_df[reg_df.index >= cutoff]

    # --- Rectas de los cuantiles del arcoíris, evaluadas en todo el eje ---
    # El ajuste es in-sample sobre toda la historia (figura descriptiva).
    rainbow_fit = fit_power_law_quantiles(reg_df, quantiles=rainbow_q)
    band_prices = {
        q: 10
        ** (
            res["params"]["Intercept"] + res["params"]["log_days"] * np.log10(grid_days)
        )
        for q, res in rainbow_fit.items()
    }

    # --- Figura 4: gráfico arcoíris con la predicción del capítulo ---
    projection_fig, ax5 = plt.subplots(figsize=(W, 0.62 * W))
    cmap = plt.get_cmap("turbo")
    n_bands = len(rainbow_q) - 1
    band_colors = cmap(np.linspace(0.08, 0.92, n_bands))
    for k, (q_lo, q_hi) in enumerate(itertools.pairwise(rainbow_q)):
        ax5.fill_between(
            grid_dates,
            band_prices[q_lo],
            band_prices[q_hi],
            color=band_colors[k],
            alpha=0.35,
            linewidth=0,
            zorder=1,
            label=rf"$Q_{{{int(q_lo * 100)}}}$--$Q_{{{int(q_hi * 100)}}}$",
        )

    # Precio por encima de los sectores.
    ax5.plot(
        hist.index,
        hist["close"],
        label="BTC",
        color="black",
        linewidth=1,
        zorder=3,
    )

    # Separador entre datos observados y proyección.
    ax5.axvline(last_date, color="gray", linestyle=":", linewidth=1.2, zorder=2)

    # Predicción del próximo máximo (texto del capítulo): rango de precio.
    ax5.vlines(target_date, t_min, t_max, color="black", linewidth=1.5, zorder=4)
    ax5.scatter(
        [target_date, target_date],
        [t_min, t_max],
        marker="_",
        s=120,
        color="black",
        linewidths=1.5,
        zorder=4,
        label=f"Predicción ({target_date:%m-%Y})",
    )
    for value in (t_min, t_max):
        ax5.annotate(
            rf"${value / 1e3:.0f}$ mil",
            xy=(target_date, value),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            zorder=5,
        )

    ax5.set_yscale("log")
    ax5.set_xlim(cutoff, grid_dates[-1])
    ax5.set_ylim(
        band_prices[rainbow_q[0]][0] * 0.8, band_prices[rainbow_q[-1]][-1] * 1.2
    )
    # Una marca por año (enero), con etiqueta del año.
    year_ticks = pd.date_range(cutoff, grid_dates[-1], freq="YS")
    ax5.set_xticks(year_ticks)
    ax5.set_xticklabels([f"{d.year}" for d in year_ticks])
    ax5.set_xlabel("Fecha")
    ax5.set_ylabel("Precio (USD)")
    # Sectores de arriba abajo, en el mismo orden que en el gráfico.
    handles, labels = ax5.get_legend_handles_labels()
    order = list(range(n_bands))[::-1] + list(range(n_bands, len(handles)))
    with plt.rc_context({"legend.fontsize": "small"}):
        ax5.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="upper left",
            ncols=2,
            framealpha=0.9,
        )
    ax5.grid(True, which="both", ls="--", alpha=0.4)
    projection_fig.savefig(FIG_DIR / "btc_quantreg_projection.pdf")

    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ### Etapa 7: pendientes desde otra fecha y piso alternativo
#
# Repite la regresión cuantílica `log10(price) ~ log10(days)` usando sólo los
# datos desde `CONFIG["recent_fit"]["start_date"]` (los días siguen contados
# desde el génesis, así que los parámetros son comparables con los del ajuste
# completo).  Se imprimen las pendientes de ambos ajustes y el piso (`Q10`
# reciente) en la fecha de la predicción de la Etapa 6, junto al piso del
# ajuste completo.  Si la pendiente reciente es menor, el piso alternativo
# queda por debajo del de la Etapa 6.  Es descriptivo, como el resto del
# cuaderno: con pocos años de datos el ajuste es poco robusto.
#
# Depende de `reg_df`, `fit` (Etapa 2) y `target_days`, `target_date`, `t_min`
# (Etapa 6).

# %%
if __name__ == "__main__":
    recent_cfg = CONFIG["recent_fit"]
    recent_start = pd.Timestamp(recent_cfg["start_date"])
    recent_df = reg_df[reg_df.index >= recent_start]
    recent_fit = fit_power_law_quantiles(
        recent_df, quantiles=tuple(recent_cfg["quantiles"])
    )

    print(
        f"\nPendientes desde {recent_start:%Y-%m-%d} "
        f"({len(recent_df)} días) frente al ajuste completo:"
    )
    for q, res in recent_fit.items():
        beta_rec = float(res["params"]["log_days"])
        beta_full = float(fit[q]["params"]["log_days"]) if q in fit else float("nan")
        print(
            f"q={q:.2f} | Beta reciente: {beta_rec:.4f} | "
            f"Beta completa: {beta_full:.4f}"
        )

    p10_rec = recent_fit[0.10]["params"]
    t_min_recent = float(
        10 ** (p10_rec["Intercept"] + p10_rec["log_days"] * np.log10(target_days))
    )
    print(
        f"\nPiso alternativo para {target_date:%Y-%m-%d} (día {target_days}):\n"
        f"  Q10 ajuste completo : {t_min:,.0f} USD\n"
        f"  Q10 desde {recent_start:%Y-%m-%d}: {t_min_recent:,.0f} USD "
        f"({t_min_recent / t_min:.2f} veces el piso completo)"
    )

# %%
# ## Métricas del experimento
#
# ``metrics_df`` recoge las magnitudes ya calculadas en las etapas
# anteriores (no se recalcula nada): los parámetros de la regresión
# cuantílica power-law por cuantil (α, β y R²) y los picos destacados con su
# fecha y su residuo suavizado.
#
# Las dos familias de magnitudes no comparten columnas, así que el frame va
# en formato largo: una fila por magnitud, con ``model`` como etiqueta de la
# magnitud y ``value`` como su valor.

if __name__ == "__main__":
    _btc_rows = []
    for _q, _res in fit.items():
        _params = _res["params"]
        _obs_log = reg_df["log_price"].to_numpy()
        _fit_log = np.log10(_res["prediction"].to_numpy())
        _ss_res = float(np.sum((_obs_log - _fit_log) ** 2))
        _ss_tot = float(np.sum((_obs_log - _obs_log.mean()) ** 2))
        _btc_rows.append(
            {"model": f"alpha (q={_q:.2f})", "value": float(_params["Intercept"])}
        )
        _btc_rows.append(
            {"model": f"beta (q={_q:.2f})", "value": float(_params["log_days"])}
        )
        _btc_rows.append(
            {"model": f"r2 (q={_q:.2f})", "value": 1.0 - _ss_res / _ss_tot}
        )

    for _date, _row in peak_sel.iterrows():
        _label = f"Pico {_row['id']} ({_date:%Y-%m-%d})"
        _btc_rows.append(
            {"model": f"resid_ma ({_label})", "value": float(_row["value"])}
        )
        _btc_rows.append({"model": f"close ({_label})", "value": float(_row["close"])})

    for _q, _res in recent_fit.items():
        _btc_rows.append(
            {
                "model": f"beta desde {recent_start:%Y} (q={_q:.2f})",
                "value": float(_res["params"]["log_days"]),
            }
        )
    _btc_rows.append({"model": "piso Q10 completo (USD)", "value": t_min})
    _btc_rows.append(
        {"model": f"piso Q10 desde {recent_start:%Y} (USD)", "value": t_min_recent}
    )

    metrics_df = pd.DataFrame(_btc_rows)

# %%

# %%
# ## Publicación de métricas en el ledger
#
# Celda final que envía las métricas disponibles al ledger estandarizado de la
# tesis.  No recalcula resultados; captura las variables del notebook.

if __name__ == "__main__":
    from thesis.results import publish

    publish(
        experiment_id="btc/btc",
        payload={"config": CONFIG, "metrics_df": metrics_df},
        chapter="Bitcoin",
    )

# %%
