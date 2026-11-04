# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: tesis
#     language: python
#     name: python3
# ---

# %%
# # Figuras descriptivas del modelo de volatilidad
#
# Cuaderno paralelo a ``Volatilidad_LSTM.py`` que produce las figuras
# necesarias para *describir* el problema y el modelo en el texto de la
# tesis, sin entrenar la LSTM ni volver a evaluarla.
#
# Contenido:
#
# 1. Autocorrelación de los retornos cuadrados (persistencia de la
#    volatilidad) y de los retornos simples (efecto apalancamiento).
# 2. Separación de precios logarítmicos y volatilidad realizada, con las
#    fechas de división de datos (train / validación / test).
# 3. Volatilidad hacia adelante a 5 sesiones (el objetivo del modelo)
#    frente a la volatilidad causal de 20 sesiones.
#
# Los datos se descargan con la misma función y las mismas fechas que usa
# ``Volatilidad_LSTM.py``, de modo que las figuras describen exactamente la
# serie con la que se entrenó el modelo.
#
# Las figuras se dibujan *in situ*, en celdas independientes, para poder
# personalizarlas sin tocar ``src/thesis`` y sin repetir la descarga: basta
# con re-ejecutar la celda de datos y las celdas de figura.

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

# Permite ejecutar el archivo desde ``notebooks/`` o ``notebooks/volatility/``.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1] if _HERE.name == "volatility" else _HERE.parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.config import TRADING_DAYS
from thesis.timeseries import log_returns, rolling_forward_window
from thesis.volatility import download_ohlcv

# %%
# ## Configuración
CONFIG: dict[str, Any] = {
    # Datos
    "ticker": "SPY",
    "start_date": "1995-01-01",
    # Divisiones temporales: deben coincidir con las de ``Volatilidad_LSTM.py``.
    "train_end": "2017-01-01",
    "val_end": "2022-01-01",
    # Ventana (en sesiones) de la volatilidad realizada histórica.
    "vol_window": 20,
    # Horizonte (en sesiones) de la volatilidad hacia adelante: coincide con
    # ``forecast_horizon`` de ``Volatilidad_LSTM.py``.
    "forward_horizon": 5,
    "smoothing_window": 15,  # Ventana de suavizado de la volatilidad forward
    # Autocorrelación
    "acf_lags": 60,
    "acf_alpha": 0.05,
    # Salidas
    "acf_squared_path": "vol_desc_acf_squared_returns.png",
    "acf_returns_path": "vol_desc_acf_returns.png",
    "prices_vol_path": "vol_desc_logprice_realized_vol.png",
    "forward_vol_path": "vol_desc_forward_vol_5d.png",
}

# %%
# ### Estilo
#
# Fondo blanco, tipografía estándar de matplotlib y cuadrícula muy tenue.
# Todos los ajustes de estilo viven en esta celda: cambiar aquí los aplica a
# todas las figuras.

if __name__ == "__main__":
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
        }
    )

# %%
# ### Datos
#
# Retornos logarítmicos, retornos cuadrados y volatilidad realizada
# histórica (hacia atrás, por tanto sin look-ahead).  El objetivo del modelo
# es la versión *forward* a 5 días de esta última; aquí se usa la versión
# causal para ilustrar el proceso.

if __name__ == "__main__":
    raw = download_ohlcv(CONFIG["ticker"], start=CONFIG["start_date"])

    log_price = np.log(raw["Close"])
    log_ret = log_returns(raw["Close"])
    sq_ret = log_ret**2

    realized_vol = (
        log_ret.rolling(CONFIG["vol_window"], min_periods=CONFIG["vol_window"]).std()
        * np.sqrt(TRADING_DAYS)
        * 100.0
    )

    # Volatilidad hacia adelante a ``H`` sesiones: rolling *forward* de los
    # retornos cuadrados y raíz cuadrada.
    #
    #   σ^H_t = sqrt( Σ_{i=1..H} r_{t+i}^2 )
    #
    # ``rolling_forward_window`` suma los próximos ``H`` valores y los ancla
    # al día actual; por construcción las últimas ``H`` observaciones son
    # ``NaN`` (centinela explícito: requieren datos que aún no existen).
    horizon = CONFIG["forward_horizon"]
    fwd_sq_sum = rolling_forward_window(sq_ret, horizon)
    fwd_vol = np.sqrt(fwd_sq_sum)
    # Misma cantidad, anualizada (factor 252/H) y en porcentaje: es el
    # espacio del objetivo del modelo y el comparable con la serie causal.
    fwd_vol_ann = np.sqrt((TRADING_DAYS / horizon) * fwd_sq_sum) * 100.0

    train_end = pd.Timestamp(CONFIG["train_end"])
    val_end = pd.Timestamp(CONFIG["val_end"])

    print(
        f"[DATOS] {raw.shape[0]} sesiones | {raw.index[0].date()} → {raw.index[-1].date()}"
    )
    print(f"  Train: hasta {train_end.date()} | Val: hasta {val_end.date()}")
    print(
        f"  Vol forward {horizon} sesiones: {fwd_vol.notna().sum()} valores "
        f"({int(fwd_vol.isna().sum())} NaN al final por falta de futuro)"
    )

# %%
# ### 1. Autocorrelación de los retornos cuadrados
#
# La persistencia de la volatilidad se mide con la autocorrelación de
# ``r_t^2``: si el vol clustering es un rasgo del proceso, esta función
# decae lentamente desde un valor alto en el lag 1.  Las líneas discontinuas
# son las bandas de ruido blanco al 95 % (``±z / √n``).

if __name__ == "__main__":
    acf_sq = acf(sq_ret.to_numpy(), nlags=CONFIG["acf_lags"], fft=True)
    n_obs = len(sq_ret)
    z = 1.959963985
    bound = z / np.sqrt(n_obs)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        np.arange(1, len(acf_sq)),
        acf_sq[1:],
        color="C0",
        width=0.8,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(bound, color="gray", ls="--", lw=0.9, label="±1,96/√n (95 %)")
    ax.axhline(-bound, color="gray", ls="--", lw=0.9)
    ax.set_xlabel("Retardo (sesiones)")
    ax.set_ylabel("Autocorrelación")
    ax.set_title("Autocorrelación de los retornos cuadrados")
    ax.set_xticks(np.arange(0, len(acf_sq), 5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONFIG["acf_squared_path"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 2. Autocorrelación de los retornos simples
#
# El efecto apalancamiento se manifiesta como autocorrelación negativa de
# los retornos: las caídas grandes van seguidas de subidas y viceversa.

if __name__ == "__main__":
    acf_ret = acf(log_ret.to_numpy(), nlags=CONFIG["acf_lags"], fft=True)
    n_obs = len(log_ret)
    z = 1.959963985
    bound = z / np.sqrt(n_obs)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        np.arange(1, len(acf_ret)),
        acf_ret[1:],
        color="C1",
        width=0.8,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(bound, color="gray", ls="--", lw=0.9, label="±1,96/√n (95 %)")
    ax.axhline(-bound, color="gray", ls="--", lw=0.9)
    ax.set_xlabel("Retardo (sesiones)")
    ax.set_ylabel("Autocorrelación")
    ax.set_title("Autocorrelación de los retornos logarítmicos")
    ax.set_xticks(np.arange(0, len(acf_ret), 5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONFIG["acf_returns_path"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 3. Precio logarítmico y volatilidad realizada
#
# Dos paneles apilados que comparten el eje temporal; el eje X sólo aparece
# en el panel inferior.  Las líneas verticales marcan el final del
# entrenamiento y el final de la validación, y las etiquetas identifican los
# tres periodos del experimento.

if __name__ == "__main__":
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2},
    )

    ax_top.plot(log_price.index, log_price.to_numpy(), color="black", lw=1.0)
    ax_top.set_ylabel("Precio Log")
    ax_top.set_title("a")
    ax_top.tick_params(labelbottom=False)

    ax_bottom.plot(
        realized_vol.index,
        realized_vol.to_numpy(),
        color="C0",
        lw=1.0,
        label=f"Media Móvil ({CONFIG['vol_window']} días)"
    )
    ax_bottom.set_ylabel("Volatilidad anualizada (%)")
    ax_bottom.set_xlabel("Fecha")
    ax_bottom.set_title("b")
    ax_bottom.legend(loc="upper left", fontsize=9)

    for ax in (ax_top, ax_bottom):
        ax.axvline(train_end, color="gray", ls="--", lw=1.0)
        ax.axvline(val_end, color="gray", ls="--", lw=1.0)

    # Etiquetas de cada partición, en el panel superior.
    inicio = log_price.index[0]
    for izquierda, derecha, etiqueta in (
        (inicio, train_end, "Entrenamiento"),
        (train_end, val_end, "Validación"),
        (val_end, log_price.index[-1], "Test"),
    ):
        ax_top.text(
            izquierda + (derecha - izquierda) / 2,
            0.94,
            etiqueta,
            transform=ax_top.get_xaxis_transform(),
            ha="center",
            va="top",
            color="black",
        )

    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CONFIG["prices_vol_path"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ### 4. Volatilidad hacia adelante a 5 sesiones (objetivo del modelo)
#
# Dos paneles con la misma escala temporal y de volatilidad:
#
# * ``a``: volatilidad realizada **causal** (ventana móvil de 20 sesiones hacia
#   atrás): es la información disponible en ``t``.
# * ``b``: volatilidad realizada **forward** a 5 sesiones, es decir el
#   rolling hacia adelante de los retornos cuadrados y su raíz cuadrada:
#
#   .. math::
#       \sigma^H_t = \sqrt{\sum_{i=1}^{H} r_{t+i}^2},
#       \qquad H = 5
#
#   que es justamente la etiqueta del modelo (en el panel se muestra
#   anualizada, ``sqrt(252/H) * σ^H``, para compartir escala con el panel
#   superior).
#
# La serie forward usa información del futuro por diseño (es el objetivo,
# no una feature), y sus últimas ``H`` observaciones quedan ``NaN`` porque no
# existe futuro para ellas.

if __name__ == "__main__":
    # Volatilidad futura promediada a 5 sesiones 

    vol_fwd_prom = np.exp(np.log(fwd_vol_ann).rolling(
    CONFIG["smoothing_window"],
    min_periods=CONFIG["smoothing_window"],
).mean())
    
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2},
    )

    ax_bottom.plot(
        vol_fwd_prom.index,
        vol_fwd_prom,
        color="0.45",
        lw=1.0,
        label=f"Futura ({CONFIG['forward_horizon']} días) promediada a {CONFIG['smoothing_window']} días",
    )
    ax_bottom.set_ylabel("Volatilidad anualizada (%)")
    ax_top.tick_params(labelbottom=False)
    ax_bottom.legend(loc="upper left", fontsize=9)

    ax_top.plot(
        fwd_vol_ann.index,
        fwd_vol_ann.to_numpy(),
        color="C0",
        lw=1.0,
        label=f"Futura ({CONFIG['forward_horizon']} días)",
    )
    ax_top.set_ylabel("Volatilidad anualizada (%)")
    ax_top.set_xlabel("Fecha")

    ax_top.set_title("a")
    ax_bottom.set_title("b")
    ax_top.legend(loc="upper left", fontsize=9)

    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CONFIG["forward_vol_path"], dpi=150, bbox_inches="tight")
    plt.show()

# %%
