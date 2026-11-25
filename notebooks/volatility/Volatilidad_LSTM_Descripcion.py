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
    "vol_window": 63,
    # Horizonte (en sesiones) de la volatilidad hacia adelante: coincide con
    # ``forecast_horizon`` de ``Volatilidad_LSTM.py``.
    "forward_horizon": 5,
    "smoothing_window": 15,  # Ventana de suavizado de la volatilidad forward
    # Autocorrelación
    "acf_lags": 60,
    "acf_alpha": 0.05,
    # Salidas: nombres de archivo dentro de FIG_DIR (``figures/Volatilidad_LSTM_Descripcion/``)
    "acf_squared_path": "vol_desc_acf_squared_returns.pdf",
    "acf_returns_path": "vol_desc_acf_returns.pdf",
    "prices_vol_path": "vol_desc_logprice_realized_vol.pdf",
    "forward_vol_path": "vol_desc_forward_vol_5d.pdf",
}

# %%
# ### Estilo
#
# Estilo común de la tesis (``styles/thesis.mplstyle``): ancho de texto de
# 155 mm, cuerpo a 12 pt y LaTeX.  Para cambiarlo se edita el ``.mplstyle`` y
# se re-ejecuta esta celda; los ajustes puntuales se hacen in situ.

if __name__ == "__main__":
    # Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
    STYLE = _ROOT / "styles" / "thesis.mplstyle"
    plt.style.use(STYLE)
    W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

    #: Figuras de este cuaderno: ``figures/Volatilidad_LSTM_Descripcion/`` en la raíz del repositorio.
    FIG_DIR = _ROOT / "figures" / "Volatilidad_LSTM_Descripcion"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

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

    fig, ax = plt.subplots(figsize=(W, 0.5 * W))
    ax.bar(
        np.arange(1, len(acf_sq)),
        acf_sq[1:],
        color="C0",
        width=0.8,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(bound, color="gray", ls="--", lw=0.9, label=r"$\pm 1{,}96/\sqrt{n}$ (95\,\%)")
    ax.axhline(-bound, color="gray", ls="--", lw=0.9)
    ax.set_xlabel("Retardo (sesiones)")
    ax.set_ylabel("Autocorrelación")
    ax.set_title("Autocorrelación de los retornos cuadrados")
    ax.set_xticks(np.arange(0, len(acf_sq), 5))
    ax.legend()
    fig.savefig(FIG_DIR / CONFIG["acf_squared_path"])
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

    fig, ax = plt.subplots(figsize=(W, 0.5 * W))
    ax.bar(
        np.arange(1, len(acf_ret)),
        acf_ret[1:],
        color="C1",
        width=0.8,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(bound, color="gray", ls="--", lw=0.9, label=r"$\pm 1{,}96/\sqrt{n}$ (95\,\%)")
    ax.axhline(-bound, color="gray", ls="--", lw=0.9)
    ax.set_xlabel("Retardo (sesiones)")
    ax.set_ylabel("Autocorrelación")
    ax.set_title("Autocorrelación de los retornos logarítmicos")
    ax.set_xticks(np.arange(0, len(acf_ret), 5))
    ax.legend()
    fig.savefig(FIG_DIR / CONFIG["acf_returns_path"])
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
        figsize=(W, 0.7 * W),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2}
    )

    ax_top.plot(log_price.index, log_price.to_numpy(), color="black", lw=0.7)
    ax_top.set_ylabel(r"ln($S_t$)")
    ax_top.text(
        0.02,
        0.85,
        "a",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
    )
    ax_top.tick_params(labelbottom=False)

    ax_bottom.plot(
        realized_vol.index,
        realized_vol.to_numpy(),
        color="C0",
        lw=0.7,
        label=f"Media Móvil ({CONFIG['vol_window']} días)"
    )
    ax_bottom.set_ylabel(r"$\sigma$ (\% anual)")
    ax_bottom.set_xlabel("Fecha")
    ax_bottom.text(
        0.02,
        0.85,
        "b",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
    )
    ax_bottom.legend(loc="upper left")

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
            0.99,
            etiqueta,
            transform=ax_top.get_xaxis_transform(),
            ha="center",
            va="top",
            color="black",
        )

    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / CONFIG["prices_vol_path"])
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
        figsize=(W, 0.8 * W),
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
    ax_bottom.set_ylabel(r"Volatilidad anualizada (\%)")
    ax_top.tick_params(labelbottom=False)
    ax_bottom.legend(loc="upper left")

    ax_top.plot(
        fwd_vol_ann.index,
        fwd_vol_ann.to_numpy(),
        color="C0",
        lw=1.0,
        label=f"Futura ({CONFIG['forward_horizon']} días)",
    )
    ax_top.set_ylabel(r"Volatilidad anualizada (\%)")
    ax_top.set_xlabel("Fecha")

    ax_top.set_title("a")
    ax_bottom.set_title("b")
    ax_top.legend(loc="upper left")

    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / CONFIG["forward_vol_path"])
    plt.show()

# %%
# ## Métricas del experimento
#
# ``metrics_df`` recoge las magnitudes ya calculadas en las celdas
# anteriores (no se recalcula nada): los primeros retardos de la
# autocorrelación de los retornos y de los retornos cuadrados, el R² del
# mejor predictor lineal AR(1) (el cuadrado de la autocorrelación del primer
# retardo) y los momentos de la volatilidad realizada causal y forward.

if __name__ == "__main__":
    metrics_df = pd.DataFrame(
        [
            {
                "model": CONFIG["ticker"],
                "acf_ret_lag1": float(acf_ret[1]),
                "acf_ret_lag5": float(acf_ret[5]),
                "acf_ret_lag10": float(acf_ret[10]),
                "acf_ret_lag20": float(acf_ret[20]),
                "acf_sq_lag1": float(acf_sq[1]),
                "acf_sq_lag5": float(acf_sq[5]),
                "acf_sq_lag10": float(acf_sq[10]),
                "acf_sq_lag20": float(acf_sq[20]),
                "acf_ret_r2": float(acf_ret[1] ** 2),
                "n_obs": int(n_obs),
                "realized_vol_mean": float(realized_vol.mean()),
                "realized_vol_std": float(realized_vol.std()),
                "fwd_vol_mean": float(fwd_vol_ann.mean()),
                "fwd_vol_std": float(fwd_vol_ann.std()),
            }
        ]
    )

# %%
# ## Publicación de métricas en el ledger
#
# Celda final que envía las métricas disponibles al ledger estandarizado de la
# tesis.  No recalcula resultados; captura las variables del notebook.

if __name__ == "__main__":
    from thesis.results import publish

    publish(
        experiment_id="volatility/lstm_description",
        payload={"config": CONFIG, "metrics_df": metrics_df},
        chapter="Volatilidad",
    )

# %%
