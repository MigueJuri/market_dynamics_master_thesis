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
# La lógica reutilizable vive en `src/thesis/btc.py`; aquí el cuaderno sólo la
# coordina, **por etapas**, con una celda por sección del pipeline:
#
# 1. **datos**: descarga (o caché) del OHLCV diario de Bitstamp;
# 2. **power-law**: preparación del frame y regresión cuantílica;
# 3. **figura**: rectas de los cuantiles y anotaciones;
# 4. **ciclos**: picos y valles macro.
#
# Cada etapa tiene su propia celda: se puede cambiar un parámetro o una
# figura y re-ejecutar **sólo** la celda afectada, sin volver a descargar los
# datos.  El archivo también se ejecuta como script
# (`python notebooks/btc/BTC.py`), en cuyo caso las celdas corren en orden.

# %%
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Permite ejecutar el archivo desde ``notebooks/`` o desde ``notebooks/btc/``.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # raíz del repositorio (``thesis/``)
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.btc import (
    BTC_BOTTOMS,
    BTC_HALVINGS,
    BTC_PEAKS,
    BTC_QUANTILES,
    detect_market_cycles,
    fetch_btc_ohlcv,
    fit_power_law_quantiles,
    plot_btc_quantreg,
    prepare_power_law_frame,
)
from thesis.config import DATA_DIR

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
    "distance": 300,
    "prominence": 0.3,
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

    print("\nQuantile Regression Results (Power-Law Exponents):")
    for q, res in fit.items():
        params = res["params"]
        print(
            f"q={q:.2f} | Beta (Slope): {params['log_days']:.4f} | "
            f"Alpha (Intercept): {params['Intercept']:.4f}"
        )
    return reg_df, fit


# %% [markdown]
# ## Etapa 3: figura
#
# Dibuja los dos paneles log-log con las rectas de los cuantiles y las
# anotaciones de halvings, picos y valles.  La figura se personalize aquí
# (etiquetas, rangos, colores) sin tocar `src/thesis/btc.py`.


# %%
def draw_power_law_figure(
    btc_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> plt.Figure:
    """Dibuja el power-law de BTC con sus anotaciones.

    Parameters
    ----------
    btc_df : pd.DataFrame
        OHLCV diario de BTC.
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.

    Returns
    -------
    plt.Figure
        La figura con los dos paneles.
    """
    cfg = {**CONFIG, **(config or {})}
    return plot_btc_quantreg(
        btc_df,
        halvings=cfg["halvings"],
        peaks=cfg["peaks"],
        bottoms=cfg["bottoms"],
        quantiles=cfg["quantiles"],
    )


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

# %%
if __name__ == "__main__":
    fig = draw_power_law_figure(btc_df, CONFIG)
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
# ## Atajo programático
#
# `run_experiment` encadena las cuatro etapas y devuelve los resultados; se usa
# desde tests o scripts.  En el cuaderno no hace falta: cada etapa ya tiene su
# celda.


# %%
def run_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo encadenando las etapas.

    Parameters
    ----------
    config : dict[str, Any] | None
        Sobrescribe claves de :data:`CONFIG`.

    Returns
    -------
    dict
        ``{"ohlcv", "reg_df", "fit", "figure", "peaks", "bottoms"}``.
    """
    cfg = {**CONFIG, **(config or {})}
    btc_df = load_ohlcv(cfg)
    reg_df, fit = fit_power_law(btc_df, cfg)
    figure = draw_power_law_figure(btc_df, cfg)
    peaks, bottoms = find_cycles(btc_df, cfg)
    return {
        "ohlcv": btc_df,
        "reg_df": reg_df,
        "fit": fit,
        "figure": figure,
        "peaks": peaks,
        "bottoms": bottoms,
    }
