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
# # Caracterización estadística del SPY
#
# **Hechos estilizados de los retornos diarios del S&P 500 (ETF `SPY`, 1993–2025)**
#
# Este cuaderno documenta la caracterización empírica de la serie de precios
# del SPY que sirve como punto de partida de la tesis: el objetivo **no** es
# predecir, sino **establecer qué propiedades estadísticas debe respetar
# cualquier modelo** que se ajuste después a esta serie.
#
# Se verifican, uno por uno, los *stylized facts* clásicos de series
# financieras (Mandelbrot 1963; Cont 2001):
#
# | # | Hecho estilizado | Etapa |
# |---|------------------|-------|
# | 1 | Crecimiento aproximadamente exponencial del precio | 3 |
# | 2 | Colas pesadas y leptocurtosis de los retornos | 4 |
# | 3 | Asimetría negativa | 4 |
# | 4 | Incertidumbre de los momentos bajo dependencia serial | 5 |
# | 5 | Ausencia de autocorrelación lineal en los retornos | 6 |
# | 6 | Autocorrelación persistente en los retornos al cuadrado | 6 |
# | 7 | Clustering de volatilidad y regímenes | 8 |
#
# ### Estructura del cuaderno
#
# - **celda 0**: magics (`%autoreload 2`) para recargar `src/` en caliente;
# - **celda 1**: `CONFIG`, con **todos** los parámetros de las figuras;
# - **celda 2**: datos (`load_single_price` + retornos);
# - **celdas 3-10**: una celda por figura, con el **código de dibujo escrito
#   aquí mismo**;
# - **celda 11**: tabla resumen.
#
# El **cómputo** vive en `src/thesis/characterization.py` (importable,
# sin `matplotlib`); el **dibujo** vive aquí, para poder retocar colores,
# bines, spans, cuantiles, `xlim` o tamaños re-ejecutando **sólo** la celda
# afectada, sin reiniciar el kernel.
#
# ### Datos
#
# Precios de cierre **ajustados** (dividendos y splits) del ETF `SPY`,
# descargados con [`yfinance`](https://github.com/ranaroussi/yfinance) desde
# el inicio de la serie (enero de 1993).  La descarga se cachea en `data/`
# para que las corridas posteriores sean reproducibles y offline.
#
# ### Cómo correrlo
#
# ```bash
# uv run jupyter notebook notebooks/spy/SPY_caracterizacion.py
# uv run python notebooks/spy/SPY_caracterizacion.py   # modo script
# ```
#
# > **Nota sobre la definición de retorno.** Salvo aclaración explícita, se
# > trabaja con retornos logarítmicos $x_t = \log S_t - \log S_{t-1}$, que son
# > aditivos en el tiempo.  Para movimientos diarios la diferencia con el
# > retorno simple $S_t/S_{t-1}-1$ es de segundo orden, y en la etapa 4 se
# > muestran ambos para dejarlo documentado.

# %% [markdown]
# ## Celda 0 — recarga en caliente de `src/`
#
# `%autoreload 2` hace que los cambios en `src/thesis/characterization.py` se
# reflejen en la **siguiente celda** sin reiniciar el kernel.  Los magics sólo
# existen en IPython: la guarda `get_ipython()` mantiene el archivo ejecutable
# como script (`uv run python notebooks/spy/SPY_caracterizacion.py`).

# %%
try:
    ip = get_ipython()
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    _INTERACTIVE = True
except NameError:  # pragma: no cover - modo script
    _INTERACTIVE = False  # script mode: sin magics

# %% [markdown]
# ## Celda 1 — imports y `CONFIG`
#
# **Todos** los parámetros del análisis (datos, bootstrap, autocorrelación,
# EWMA, colores, bines, `xlim`, tamaños y rutas de salida) viven en este
# diccionario: cambiar un valor y re-ejecutar la celda de la figura afectada
# basta para rehacerla.

# %%
import sys
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# --- Ruta del repositorio (permite ejecutar el cuaderno desde cualquier CWD) --
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _HERE.parents[1] if (_HERE.parents[1] / "src" / "thesis").is_dir() else _HERE
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.characterization import (
    annualized_stats,
    block_bootstrap_ci,
    bootstrap_acf,
    build_summary_table,
    compute_ewma_volatility,
    describe_returns,
    excess_kurtosis,
    fit_return_distribution,
    fit_squared_return_distributions,
    freedman_diaconis_bins,
    high_volatility_mask,
    unconditional_band_excursions,
)
from thesis.config import DATA_DIR, PLOT_RC
from thesis.market_data import load_single_price
from thesis.timeseries import log_returns, simple_returns

#: Directorio de figuras (anclado a la raíz del repositorio, no al CWD).
OUTPUT_DIR = _ROOT / "outputs"

plt.rcParams.update(PLOT_RC)

CONFIG: dict[str, Any] = {
    # --- Datos ---
    "ticker": "SPY",
    "start": "1993-01-01",
    "end": "2026-01-01",
    "trading_days": 252,
    "seed": 42,
    # --- Bootstrap ---
    # Moving block para los momentos (celda 5); i.i.d. para las bandas de la
    # ACF (celda 6).
    "block_size": 21,
    "n_boot": 1_000,
    "alpha": 0.05,
    # --- Autocorrelación ---
    "max_lags": 50,
    # --- Volatilidad EWMA (convención RiskMetrics) ---
    "lambda_ewma": 0.94,
    "vol_quantile": 0.90,
    # --- Figuras: tamaños ---
    "figsize": (7.0, 3.5),
    "figsize_hist_sq": (6.0, 6.0),
    "figsize_vol": (11.0, 6.0),
    "figsize_bands": (11.0, 4.0),
    # --- Figuras: bines, spans y escalas ---
    "smooth_window": 10,
    "return_ma_years": 5,
    "bins": 250,
    "xlim_sigma": 5.0,
    "xlim_log": 0.10,
    "log_price_ticks": [25, 50, 100, 200, 400, 800],
    "band_ks": (1.0, 2.0, 3.0, 5.0),
    # --- Figuras: colores ---
    "color_price": "C0",
    "color_hist": "skyblue",
    "color_normal": "r",
    "color_student": "darkorange",
    "color_band": "lightgray",
    "color_vol": "C0",
    "color_high_vol": "red",
    "color_sigma1": "gold",
    "color_sigma2": "tomato",
    "color_mu": "navy",
    "color_return_ma": "C3",
    # --- Salida ---
    "save_figures": True,
    "show_figure": True,
}

# %% [markdown]
# ## Celda 2 — datos
#
# Una sola descarga (cacheada en `data/`): la serie de precios, el retorno
# logarítmico y el retorno simple.  Todas las celdas siguientes consumen estas
# variables, de modo que no hay dos secciones trabajando sobre ventanas
# temporales distintas.

# %%
if __name__ == "__main__":
    price = load_single_price(
        ticker=CONFIG["ticker"],
        start=CONFIG["start"],
        end=CONFIG["end"],
        cache_dir=DATA_DIR,
    )
    log_ret = log_returns(price)
    simple_ret = simple_returns(price)
    r2 = log_ret.pow(2.0)

    print(f"Ticker          : {CONFIG['ticker']}")
    print(f"Rango           : {price.index[0]:%Y-%m-%d} a {price.index[-1]:%Y-%m-%d}")
    print(f"Observaciones   : {len(price):,} precios  |  {len(log_ret):,} retornos")
    print(f"Precio inicial  : {price.iloc[0]:,.2f} USD")
    print(f"Precio final    : {price.iloc[-1]:,.2f} USD")
    print(f"Múltiplo total  : x{price.iloc[-1] / price.iloc[0]:,.1f}")

# %% [markdown]
# ## Celda 3 — trayectoria del precio
#
# Un panel en escala lineal exagera visualmente el período reciente: el mismo
# movimiento relativo ocupa más espacio vertical cuanto mayor es el precio.  En
# escala logarítmica, en cambio, **pendientes iguales corresponden a tasas de
# crecimiento iguales**, de modo que una recta equivale a crecimiento
# exponencial constante.  Se grafican los dos paneles juntos para hacer
# explícito el contraste.
#
# El suavizado (media móvil) es puramente cosmético y no se usa en ningún
# cálculo posterior.

# %%
if __name__ == "__main__":
    price_smooth = price.rolling(CONFIG["smooth_window"]).mean()

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=CONFIG["figsize"])

    ax_lin.plot(
        price_smooth.index,
        price_smooth.values,
        linewidth=1.2,
        color=CONFIG["color_price"],
    )
    ax_lin.set_title("Escala lineal")
    ax_lin.set_xlabel("Fecha")
    ax_lin.set_ylabel("Precio ajustado (USD)")

    ax_log.plot(
        price_smooth.index,
        price_smooth.values,
        linewidth=1.2,
        color=CONFIG["color_price"],
    )
    ax_log.set_yscale("log")
    ax_log.set_title("Escala logarítmica")
    ax_log.set_xlabel("Fecha")
    ax_log.set_ylabel("Precio ajustado (USD)")

    # Ticks explícitos: en escala log el formato por defecto es notación
    # científica y el eje menor satura de etiquetas.
    ax_log.set_yticks(CONFIG["log_price_ticks"])
    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax_log.yaxis.set_major_formatter(formatter)
    ax_log.yaxis.set_minor_locator(mticker.NullLocator())

    fig.suptitle(
        f"{CONFIG['ticker']}: precio de cierre ajustado "
        f"(media móvil de {CONFIG['smooth_window']} ruedas)",
        y=1.02,
    )
    fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUTPUT_DIR / "spy_price_trajectory.png", dpi=300, bbox_inches="tight"
        )
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ## Celda 4 — distribución de retornos
#
# Sea $S_t$ el precio del activo al cierre del día $t$.  El retorno logarítmico
# se define como
#
# $$x_t = \log S_t - \log S_{t-1}.$$
#
# Bajo el modelo de referencia (Bachelier–Osborne, movimiento browniano
# geométrico) los $x_t$ serían i.i.d. gaussianos.  La distribución empírica se
# aparta de esa hipótesis en dos direcciones:
#
# - **leptocurtosis**: exceso de curtosis
#   $\kappa = \mathbb{E}[(x-\mu)^4]/\sigma^4 - 3 \gg 0$: hay más masa en el
#   centro y, sobre todo, en las colas;
# - **asimetría negativa**: $\mathrm{skew} < 0$: las caídas grandes son más
#   frecuentes y abruptas que las subas de igual magnitud.
#
# Para hacer visibles las colas se grafica la densidad en **escala
# logarítmica en $y$**: sobre ese eje una gaussiana es una parábola invertida,
# y cualquier desvío de esa forma en los extremos es una cola pesada.  Se
# superpone además una $t$ de Student ajustada por máxima verosimilitud,
# cuyos grados de libertad $\nu$ cuantifican el peso de las colas.
#
# El número de bines del histograma puede fijarse en `CONFIG["bins"]`; el
# número **sugerido** por la **regla de Freedman–Diaconis**
# ($h = 2\,\mathrm{IQR}\,n^{-1/3}$, robusta frente a outliers) se imprime
# como referencia.

# %%
if __name__ == "__main__":
    x = log_ret.values
    params = fit_return_distribution(x)

    # Rejilla del panel lineal: ± CONFIG["xlim_sigma"] desviaciones estándar.
    sigma_daily = float(log_ret.std())
    xlim_lin = CONFIG["xlim_sigma"] * sigma_daily
    grid_lin = np.linspace(-xlim_lin, xlim_lin, 500)
    # Rejilla del panel logarítmico: ventana fija alrededor del cero.
    grid_log = np.linspace(-CONFIG["xlim_log"], CONFIG["xlim_log"], 500)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(9,3.5))

    # --- Panel izquierdo: escala lineal (cuerpo de la distribución) ---
    ax_lin.hist(
        x,
        bins=CONFIG["bins"],
        density=True,
        alpha=0.6,
        color=CONFIG["color_hist"],
        label="Histograma",
        zorder=1,
    )
    ax_lin.plot(
        grid_lin,
        stats.norm.pdf(grid_lin, params["mu"], params["sigma"]),
        color=CONFIG["color_normal"],
        linewidth=1.8,
        label="Normal",
        zorder=4,
    )
    ax_lin.plot(
        grid_lin,
        stats.t.pdf(grid_lin, params["df"], params["loc"], params["scale"]),
        color=CONFIG["color_student"],
        linestyle="--",
        linewidth=1.8,
        label=rf"t-Student ($\nu$={params['df']:.1f})",
        zorder=3,
    )
    ax_lin.text(
        0.1,
        0.95,
        "a",
        transform=ax_lin.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_lin.set_xlim(-xlim_lin, xlim_lin)
    ax_lin.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_lin.set_xlabel("Retorno logarítmico")
    ax_lin.set_ylabel("Densidad")
    # ax_lin.set_title("Cuerpo de la distribución")
    ax_lin.legend(loc="best")

    # --- Panel derecho: escala logarítmica (colas) ---
    ax_log.hist(
        x,
        bins=CONFIG["bins"],
        density=True,
        alpha=0.6,
        color=CONFIG["color_hist"],
        label="Histograma",
        zorder=1,
    )
    ax_log.plot(
        grid_log,
        stats.norm.pdf(grid_log, params["mu"], params["sigma"]),
        color=CONFIG["color_normal"],
        linewidth=1.8,
        label="Normal",
        zorder=4,
    )
    ax_log.plot(
        grid_log,
        stats.t.pdf(grid_log, params["df"], params["loc"], params["scale"]),
        color=CONFIG["color_student"],
        linestyle="--",
        linewidth=1.8,
        label=rf"t-Student ($\nu$={params['df']:.1f})",
        zorder=3,
    )
    ax_log.text(
        0.9,
        0.95,
        "b",
        transform=ax_log.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_log.set_yscale("log")
    ax_log.set_ylim(bottom=1e-3)
    ax_log.set_xlim(-CONFIG["xlim_log"], CONFIG["xlim_log"])
    ax_log.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_log.set_xlabel("Retorno logarítmico")
    ax_log.set_ylabel("Densidad")
    # ax_log.set_title("Colas (densidad en escala log)")
    ax_log.legend(loc="best")

    fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / "spy_return_density.png", dpi=300, bbox_inches="tight")
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close("all")

    print(
        f"Bines sugeridos (Freedman-Diaconis): {freedman_diaconis_bins(log_ret.values)}"
    )
    print(f"Normal    : mu={params['mu']:.5f}, sigma={params['sigma']:.5f}")
    print(
        f"t-Student : nu={params['df']:.2f}, loc={params['loc']:.5f}, "
        f"scale={params['scale']:.5f}"
    )

    # Momentos bajo las dos definiciones de retorno: la asimetría y la
    # curtosis son cualitativamente idénticas, luego la elección de la
    # definición no genera el hecho estilizado, sólo lo reexpresa.
    moments = pd.concat(
        [
            describe_returns(simple_ret.values, "Retorno simple"),
            describe_returns(log_ret.values, "Retorno logarítmico"),
        ],
        axis=1,
    )
    print()
    # ``DataFrame.map`` no existe en pandas 2.0: se usa ``apply`` por columna.
    print(moments.apply(lambda col: col.map(lambda v: f"{v:.4g}")).to_string())

# %% [markdown]
# ## Celda 5 — momentos con incertidumbre (*moving block bootstrap*)
#
# Reportar una curtosis puntual sin barras de error es engañoso: se trata de un
# estimador de momento de orden 4, dominado por un puñado de observaciones
# extremas, y por lo tanto de varianza muy alta.
#
# El bootstrap i.i.d. clásico tampoco sirve acá, porque **destruye la
# dependencia temporal**: al remuestrear observaciones sueltas se rompe el
# clustering de volatilidad (etapa 8) y los intervalos resultan
# artificialmente angostos.  Se usa entonces un *moving block bootstrap*, que
# remuestrea bloques contiguos de longitud $b$ y preserva la estructura de
# dependencia dentro de cada bloque.
#
# La elección $b \approx 10$ días hábiles es un compromiso: suficientemente
# largo para contener un episodio típico de volatilidad, suficientemente corto
# para que el número de bloques efectivos $T/b$ mantenga variabilidad en las
# réplicas.  La semilla es **explícita**: el bootstrap es reproducible sin
# tocar el estado global de numpy.
#
# En la celda 6, en cambio, las bandas de la ACF se calculan con el bootstrap
# i.i.d. simple: el contraste que interesa allí es contra la hipótesis nula
# de independencia, y no contra la varianza muestral de un estimador de
# momento con fuerte dependencia.

# %%
if __name__ == "__main__":

    def _annualized(x_boot: np.ndarray) -> np.ndarray:
        """Estadístico de bootstrap: media y volatilidad anualizadas."""
        return annualized_stats(x_boot, trading_days=CONFIG["trading_days"])

    k_hat, k_ci = block_bootstrap_ci(
        log_ret.values,
        excess_kurtosis,
        block_size=CONFIG["block_size"],
        reps=CONFIG["n_boot"],
        alpha=CONFIG["alpha"],
        seed=CONFIG["seed"],
    )
    ann_hat, ann_ci = block_bootstrap_ci(
        log_ret.values,
        _annualized,
        block_size=CONFIG["block_size"],
        reps=CONFIG["n_boot"],
        alpha=CONFIG["alpha"],
        seed=CONFIG["seed"],
    )

    print(
        f"Exceso de curtosis      : {k_hat[0]:7.2f}   "
        f"(IC {1 - CONFIG['alpha']:.0%}: [{k_ci[0, 0]:.2f}, {k_ci[1, 0]:.2f}])"
    )
    print(
        f"Media anualizada        : {ann_hat[0]:7.2%}   "
        f"(IC {1 - CONFIG['alpha']:.0%}: [{ann_ci[0, 0]:.2%}, {ann_ci[1, 0]:.2%}])"
    )
    print(
        f"Volatilidad anualizada  : {ann_hat[1]:7.2%}   "
        f"(IC {1 - CONFIG['alpha']:.0%}: [{ann_ci[0, 1]:.2%}, {ann_ci[1, 1]:.2%}])"
    )

# %% [markdown]
# ## Celda 6 — autocorrelación: retornos vs. retornos al cuadrado
#
# El par de resultados más citado sobre series financieras diarias:
#
# 1. $\rho(x_t, x_{t-k}) \approx 0$ para $k \ge 1$ — no hay estructura
#    **lineal** predecible;
# 2. $\rho(x_t^2, x_{t-k}^2) > 0$ y con decaimiento lento — la **magnitud** de
#    los retornos sí es fuertemente predecible.
#
# Es decir: los retornos son *no correlacionados* pero no *independientes*.
#
# **Sobre las bandas de confianza.**  Las bandas estándar $\pm z/\sqrt{T}$
# (ruido blanco de Bartlett) suponen independencia y homocedasticidad: bajo
# heterocedasticidad condicional —exactamente lo que caracteriza a esta
# serie— subestiman la varianza de $\hat\rho_k$ y producen rechazos
# espurios.  Las bandas de esta celda se calculan por **bootstrap i.i.d.**
# (remuestreo con reemplazo de observaciones individuales): la ACF se
# recalcula sobre cada réplica y la banda percentil $[\alpha/2, 1-\alpha/2]$
# mide su dispersión.
#
# El punto clave del remuestreo i.i.d. es que **destruye la dependencia
# temporal**: cada réplica es una permutación de las observaciones, de modo
# que su autocorrelación es nula por construcción.  La banda resultante es
# entonces la distribución *nula* del estimador (el equivalente empírico de
# la banda de Bartlett, pero calculado por simulación sobre la distribución
# empírica de la serie), y el contraste $\lvert\hat\rho_k\rvert > h_k$ es un
# test contra la hipótesis nula de independencia.
#
# El precio de esta simplicidad es que la banda **subestima la incertidumbre
# real** del estimador en el panel de $x_t^2$: la autocorrelación verdadera
# de los cuadrados es persistente y su varianza muestral es más ancha que la
# nula.  En ese panel el conteo de lags "significativos" debe leerse como
# evidencia *contra la independencia*, no como una medida de la magnitud de
# la persistencia.  (Para los momentos de la celda 5, en cambio, el bootstrap
# por bloques sí es el adecuado, porque ahí el estimador sí depende de la
# estructura temporal de la muestra.)
#
# La banda es **simétrica** ($\pm h_k$) alrededor de cero y el lag 0 no lleva
# banda (su autocorrelación vale 1 por definición).

# %%
if __name__ == "__main__":
    max_lags = CONFIG["max_lags"]
    alpha = CONFIG["alpha"]

    # Retornos: ACF con bandas por bootstrap i.i.d.
    acf_r, ci_r = bootstrap_acf(
        log_ret.values,
        max_lags=max_lags,
        reps=CONFIG["n_boot"],
        alpha=alpha,
        seed=CONFIG["seed"],
    )
    # Retornos al cuadrado: proxy de volatilidad (misma banda bootstrap).
    acf_sq, ci_sq = bootstrap_acf(
        log_ret.values**2,
        max_lags=max_lags,
        reps=CONFIG["n_boot"],
        alpha=alpha,
        seed=CONFIG["seed"],
    )

    fig_r, ax_r = plt.subplots(figsize=CONFIG["figsize"])

    markerline, *_ = ax_r.stem(np.arange(len(acf_r)), acf_r, basefmt=" ")
    plt.setp(markerline, markersize=3)
    ax_r.fill_between(
        np.arange(len(ci_r)),
        -ci_r,
        ci_r,
        color=CONFIG["color_band"],
        alpha=0.7,
        label=f"IC {1 - alpha:.0%} (bootstrap i.i.d.)",
    )
    ax_r.set_title(r"Retornos $x_t$")
    ax_r.set_xlabel("Lag $k$ (días)")
    ax_r.set_ylabel(r"$\rho_k$")
    ax_r.legend(loc="upper right")

    fig_r.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_r.savefig(
            OUTPUT_DIR / "spy_acf_returns.png", dpi=300, bbox_inches="tight"
        )
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close(fig_r)

    fig_sq, ax_r2 = plt.subplots(figsize=CONFIG["figsize"])

    markerline, *_ = ax_r2.stem(np.arange(len(acf_sq)), acf_sq, basefmt=" ")
    plt.setp(markerline, markersize=3)
    ax_r2.fill_between(
        np.arange(len(acf_sq)),
        -ci_sq,
        ci_sq,
        color=CONFIG["color_band"],
        alpha=0.7,
        label=f"IC {1 - alpha:.0%} (bootstrap i.i.d.)",
    )
    # ax_r2.set_title(r"Retornos al cuadrado $x_t^2$")
    ax_r2.set_xlabel("Lag $k$ (días)")
    ax_r2.legend(loc="upper right")
    ax_r2.set_ylabel(r"$\rho_k$")

    fig_sq.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_sq.savefig(
            OUTPUT_DIR / "spy_acf_squared_returns.png", dpi=300, bbox_inches="tight"
        )
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close(fig_sq)

    print(
        f"Lags 1-{max_lags} fuera de la banda bootstrap — retornos  : "
        f"{np.sum(np.abs(acf_r[1:]) > ci_r[1:])}/{max_lags}"
    )
    print(
        f"Lags 1-{max_lags} fuera de la banda bootstrap — retornos² : "
        f"{np.sum(np.abs(acf_sq[1:]) > ci_sq[1:])}/{max_lags}"
    )

# %% [markdown]
# ## Celda 7 — histograma de los retornos al cuadrado
#
# El histograma de $x_t^2$ muestra la asimetría positiva de la distribución de
# la magnitud del retorno (panel superior). En el panel inferior se observa
# cómo el logaritmo de los retornos al cuadrado tiene una forma más próxima a
# una normal. Se superponen los ajustes log-normal para $x_t^2$ y normal para
# $\log(x_t^2)$; los retornos exactamente nulos se excluyen del segundo ajuste.
#
# El número de bines de cada panel es el que sugiere la **regla de
# Freedman–Diaconis** (robusta frente a los outliers que el propio panel está
# mostrando).

# %%
if __name__ == "__main__":
    r_2 = log_ret.values**2
    r_2_log = np.log(r_2[r_2 > 0])
    log_r_2_moments = {
        "media": float(np.mean(r_2_log)),
        "varianza": float(np.var(r_2_log)),
        "asimetría": float(stats.skew(r_2_log)),
        "curtosis": float(stats.kurtosis(r_2_log, fisher=True)),
    }
    squared_params = fit_squared_return_distributions(log_ret.values)
    r_2_grid = np.linspace(0.0, np.percentile(r_2, 99.5), 500)
    r_2_log_grid = np.linspace(r_2_log.min(), r_2_log.max(), 500)

    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(8,5))

    ax_lin.hist(
        r_2,
        bins=freedman_diaconis_bins(r_2),
        density=True,
        color=CONFIG["color_hist"],
        label="Histograma",
    )
    ax_lin.plot(
        r_2_grid,
        stats.lognorm.pdf(
            r_2_grid,
            squared_params["lognorm_shape"],
            loc=squared_params["lognorm_loc"],
            scale=squared_params["lognorm_scale"],
        ),
        color=CONFIG["color_normal"],
        linewidth=1.8,
        linestyle="--",
        alpha = 0.8,
        label="Log-normal",
    )
    ax_lin.set_xlabel(r"$x_t^2$")
    ax_lin.set_ylabel("Densidad")
    ax_lin.set_xlim(left=-1e-5, right=float(np.percentile(r_2, 90)))
    # ax_lin.set_title("Retornos al cuadrado")
    ax_lin.text(
        0.1,
        0.95,
        "a",
        transform=ax_lin.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )

    ax_log.hist(
        r_2_log,
        bins=freedman_diaconis_bins(r_2_log),
        density=True,
        color=CONFIG["color_hist"],
        label="Histograma",
    )
    ax_log.plot(
        r_2_log_grid,
        stats.norm.pdf(
            r_2_log_grid,
            squared_params["log_mu"],
            squared_params["log_sigma"],
        ),
        color=CONFIG["color_normal"],
        linewidth=1.8,
        linestyle="--",
        alpha = 0.8,
        label="Normal",
    )
    ax_log.set_xlabel(r"$\log(x_t^2)$")
    ax_log.set_ylabel("Densidad")
    # ax_log.set_title("Logaritmo de retornos al cuadrado (escala lineal)")
    ax_log.text(
        0.1,
        0.95,
        "b",
        transform=ax_log.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_lin.legend(loc="best")
    ax_log.legend(loc="best")

    fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUTPUT_DIR / "spy_sq_returns_hist.png", dpi=300, bbox_inches="tight"
        )
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close("all")

    print(
        f"Log-normal r²: shape={squared_params['lognorm_shape']:.5f}, "
        f"scale={squared_params['lognorm_scale']:.5f}"
    )
    print(
        f"Normal log(r²): mu={squared_params['log_mu']:.5f}, "
        f"sigma={squared_params['log_sigma']:.5f}"
    )
    print("Primeros cuatro momentos de log(r²):")
    for name, value in log_r_2_moments.items():
        print(f"  {name}: {value:.8f}")

# %% [markdown]
# ## Celda 8 — clustering de volatilidad y regímenes
#
# La autocorrelación de $x_t^2$ tiene una contrapartida visual directa: los
# períodos de alta y baja volatilidad se agrupan.  Se estima la volatilidad
# condicional con un promedio móvil exponencial (EWMA, convención RiskMetrics
# $\lambda = 0.94$),
#
# $$\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,x_{t-1}^2,$$
#
# que es el caso límite de un GARCH(1,1) integrado sin término constante.  Se
# elige EWMA por sobre un GARCH ajustado por máxima verosimilitud porque no
# requiere optimización numérica y no introduce parámetros estimados en un
# análisis que es puramente descriptivo.
#
# La estimación es **causal**: en la fecha $t$ se usa únicamente $x_{t-1}^2$.
#
# Se sombrean los períodos en que $\sigma_t$ supera su percentil 90: 1998,
# 2000–2002, 2008–2009, 2020 y 2022 aparecen como los regímenes de tensión, y
# cada uno coincide con una caída significativa del precio — la manifestación
# del *efecto apalancamiento*.
#
# .. warning::
#    El percentil se calcula sobre **toda** la muestra: es una comparación
#    **descriptiva**, no una señal libre de look-ahead.

# %%
if __name__ == "__main__":
    volatility = compute_ewma_volatility(
        log_ret,
        lambda_ewma=CONFIG["lambda_ewma"],
        trading_days=CONFIG["trading_days"],
    )
    vol_threshold = float(volatility.dropna().quantile(CONFIG["vol_quantile"]))
    high_vol_mask = high_volatility_mask(
        volatility, quantile=CONFIG["vol_quantile"], index=price.index
    )

    fig, (ax_vol, ax_price) = plt.subplots(
        2,
        1,
        figsize=CONFIG["figsize_vol"],
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    ax_vol.plot(
        volatility.index,
        volatility.values,
        linewidth=0.8,
        color=CONFIG["color_vol"],
        label=rf"Volatilidad EMA ($\lambda$={CONFIG['lambda_ewma']})",
    )
    ax_vol.axhline(
        vol_threshold,
        linestyle="--",
        color=CONFIG["color_high_vol"],
        alpha=0.6,
        linewidth=1.0,
        label=f"Percentil {CONFIG['vol_quantile']:.0%} = {vol_threshold:.1%}",
    )
    ax_vol.set_ylabel(r"$\sigma$")
    # ax_vol.set_title(f"{CONFIG['ticker']}: precio y regímenes de alta volatilidad")
    ax_vol.legend(loc="upper left")

    ax_price.semilogy(
        price.index,
        price.values,
        linewidth=1.0,
        color=CONFIG["color_price"],
        label=CONFIG["ticker"],
    )
    ax_price.fill_between(
        price.index,
        price.min(),
        price.max(),
        where=high_vol_mask.values,
        color=CONFIG["color_high_vol"],
        alpha=0.18,
        interpolate=True,
        label=rf"$\sigma_t$ > percentil {CONFIG['vol_quantile']:.0%}",
    )
    ax_price.set_ylabel("Precio (USD, escala log)")
    ax_price.set_xlabel("Fecha")
    ax_price.legend(loc="upper left")
    ax_price.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    price_ticks = [25, 50, 100, 200, 500]
    ax_price.set_yticks(price_ticks)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_price.yaxis.set_minor_locator(mticker.NullLocator())

    fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / "spy_vol_regimes.png", dpi=300, bbox_inches="tight")
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close("all")

    # Media EWMA causal: en la fecha t se usa únicamente x_(t-1).
    ewma_mean = (
        log_ret.shift(1)
        .ewm(alpha=1.0 - CONFIG["lambda_ewma"], adjust=False, min_periods=30)
        .mean()
        * CONFIG["trading_days"]
    )

    fig_mu_vol, (ax_mean, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(9,4),
        sharex=True,
    )

    ax_mean.plot(
        ewma_mean.index,
        ewma_mean.values,
        linewidth=0.8,
        color=CONFIG["color_mu"],
    )
    ax_mean.text(
        0.02,
        0.95,
        "a",
        transform=ax_mean.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_mean.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_mean.set_ylabel(r"$\mu$")
    # ax_mean.set_title(
    #     rf"{CONFIG['ticker']}: media y volatilidad EWMA "
    #     rf"($\lambda$={CONFIG['lambda_ewma']})"
    # )

    ax_vol.plot(
        volatility.index,
        volatility.values,
        linewidth=0.8,
        color=CONFIG["color_vol"],
    )
    ax_vol.text(
        0.02,
        0.95,
        "b",
        transform=ax_vol.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_vol.set_ylabel(r"$\sigma$")
    ax_vol.set_xlabel("Fecha")
    ax_vol.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig_mu_vol.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_mu_vol.savefig(
            OUTPUT_DIR / "spy_ewma_mean_volatility.png",
            dpi=300,
            bbox_inches="tight",
        )
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close(fig_mu_vol)



# %% [markdown]
# ## Celda 9 — bandas incondicionales: por qué $\sigma$ constante no alcanza
#
# Graficar la serie de retornos contra bandas $\pm k\sigma$ calculadas sobre
# **toda** la muestra muestra el problema desde otro ángulo: las excursiones
# fuera de la banda no están repartidas de manera homogénea en el tiempo, sino
# concentradas en los mismos episodios de la etapa anterior.  Además, la
# fracción observada fuera de $\pm 2\sigma$ excede a la que predice una
# gaussiana, que es la traducción cuantitativa de las colas pesadas de la
# etapa 4.

# %%
if __name__ == "__main__":
    mu = float(log_ret.mean())
    sigma = float(log_ret.std())

    fig, ax = plt.subplots(figsize=CONFIG["figsize_bands"])
    ax.plot(
        log_ret.index,
        log_ret.values,
        color="black",
        lw=0.5,
        alpha=0.9,
        label="Retorno logarítmico",
    )
    ax.fill_between(
        log_ret.index,
        mu - sigma, 
        mu + sigma,
        color=CONFIG["color_sigma1"],
        alpha=0.35,
        label=r"$\pm 1\sigma$",
    )
    ax.fill_between(
        log_ret.index,
        mu - 2 * sigma,
        mu + 2 * sigma,
        color=CONFIG["color_sigma2"],
        alpha=0.18,
        label=r"$\pm 2\sigma$",
    )
    ax.axhline(
        mu, color=CONFIG["color_mu"], ls="--", lw=1.0, label=rf"$\mu$ = {mu:.5f}"
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno logarítmico")
    ax.set_title(
        f"{CONFIG['ticker']}: retornos diarios y bandas de volatilidad incondicional"
    )
    ax.legend(loc="lower right", ncol=4)
    fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / "spy_uncond_bands.png", dpi=300, bbox_inches="tight")
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close("all")

    # Fracción observada fuera de las bandas vs. la predicción gaussiana.
    excursions = unconditional_band_excursions(log_ret, ks=CONFIG["band_ks"])
    print(excursions.to_string(index=False))

# %% [markdown]
# ## Celda 10 — promedio móvil de retornos
#
# El promedio móvil de cinco años suaviza las fluctuaciones diarias de los
# retornos logarítmicos y permite observar la evolución de su media local. Tanto
# la media móvil como la línea horizontal se expresan como retornos
# logarítmicos anualizados, usando el número de ruedas configurado.

# %%
if __name__ == "__main__":
    return_ma_window = CONFIG["return_ma_years"] * CONFIG["trading_days"]
    annualization_factor = CONFIG["trading_days"]
    log_ret_ma = (
        log_ret.rolling(return_ma_window).mean() * annualization_factor
    )
    historical_mean = float(log_ret.mean() * annualization_factor)

    fig, ax = plt.subplots(figsize=(6,3), dpi = 300)
    ax.axhline(
        0.0,
        color="black",
        linewidth=0.2,
        # alpha=0.5,
    )
    ax.plot(
        log_ret_ma.index,
        log_ret_ma.values,
        color=CONFIG["color_return_ma"],
        linewidth=1.2,
        label=(
            f"Media móvil de {CONFIG['return_ma_years']} años "
        ),
    )
    ax.axhline(
        historical_mean,
        color=CONFIG["color_mu"],
        linestyle="--",
        linewidth=1.0,
        label=f"Promedio histórico = {historical_mean:.2%}",
    )

    ax.set_xlabel("Fecha")
    ax.set_ylabel(r"$\mu_{\text{anual}}$")
    # ax.set_title(
    #     f"{CONFIG['ticker']}: promedio móvil de "
    #     f"{CONFIG['return_ma_years']} años de retornos logarítmicos anualizados"
    # )
    ax.legend(loc="best")
    fig.tight_layout()

    if CONFIG["save_figures"]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUTPUT_DIR / "spy_log_return_5y_annualized_moving_average.png",
            dpi=300,
            bbox_inches="tight",
        )
    if CONFIG["show_figure"] and _INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)

# %% [markdown]
# ## Celda 11 — tabla resumen
#
# La tabla se calcula a partir de los datos (no hay números escritos a mano en
# el texto): cambiar `CONFIG["ticker"]` o la ventana temporal regenera el
# resumen completo.

# %%
if __name__ == "__main__":
    summary = build_summary_table(
        log_ret,
        ticker=CONFIG["ticker"],
        alpha=CONFIG["alpha"],
        max_lags=CONFIG["max_lags"],
        acf_returns=(acf_r, ci_r),
        acf_squared=acf_sq,
        squared_margin=ci_sq,
    )
    print(summary.to_string(index=False))

# %% [markdown]
# ### Conclusiones
#
# 1. El precio crece de forma aproximadamente exponencial, pero con drawdowns
#    que la escala lineal oculta.
# 2. Los retornos diarios **no son gaussianos**: el exceso de curtosis es de
#    un orden de magnitud incompatible con la normalidad incluso considerando
#    la incertidumbre del bootstrap por bloques, y la $t$ de Student ajustada
#    indica colas con pocos momentos finitos.
# 3. La asimetría negativa es sistemática: las caídas son más abruptas que las
#    subas.
# 4. No hay autocorrelación lineal explotable en los retornos, pero sí una
#    dependencia fuerte y persistente en su magnitud.  Cualquier modelo
#    razonable debe ser **no correlacionado pero dependiente**: ARCH/GARCH,
#    volatilidad estocástica o modelos multifractales.
# 5. La volatilidad se agrupa en regímenes identificables, y los de alta
#    volatilidad coinciden con caídas de precio (efecto apalancamiento).
#
# En conjunto, esto descarta el movimiento browniano geométrico con $\sigma$
# constante como modelo de trabajo y motiva el marco de volatilidad condicional
# que se desarrolla en el resto del proyecto.
#
# ---
#
# ### Referencias
#
# - Mandelbrot, B. (1963). *The Variation of Certain Speculative Prices*.
#   Journal of Business, 36(4).
# - Fama, E. (1965). *The Behavior of Stock-Market Prices*. Journal of
#   Business, 38(1).
# - Diebold, F. X. (1986). *Testing for Serial Correlation in the Presence of
#   ARCH*. Proceedings of the ASA.
# - Bollerslev, T. (1986). *Generalized Autoregressive Conditional
#   Heteroskedasticity*. Journal of Econometrics, 31(3).
# - Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General
#   Stationary Observations*. Annals of Statistics, 17(3).
# - Cont, R. (2001). *Empirical properties of asset returns: stylized facts and
#   statistical issues*. Quantitative Finance, 1(2).
# - J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.

# %%
# ## Métricas del experimento
#
# ``metrics_df`` recoge las magnitudes ya calculadas en las celdas
# anteriores (no se recalcula nada): la tabla resumen, los momentos de los
# retornos y los parámetros de la densidad ajustada (normal y t de Student).
#
# Las fuentes tienen métricas heterogéneas (tabla resumen en texto, momentos
# y parámetros numéricos), así que el frame va en formato largo: una fila por
# magnitud, con ``model`` como etiqueta de la magnitud y ``value`` como su
# valor.

if __name__ == "__main__":
    _summary_obs_col = summary.columns[1]
    _summary_ref_col = summary.columns[2]

    _summary_rows = [
        {"model": f"{row['Métrica']} (observado)", "value": row[_summary_obs_col]}
        for _, row in summary.iterrows()
    ] + [
        {
            "model": f"{row['Métrica']} (normalidad i.i.d.)",
            "value": row[_summary_ref_col],
        }
        for _, row in summary.iterrows()
    ]

    _moment_rows = [
        {
            "model": f"{stat} ({label})",
            "value": float(moments.loc[stat, label]),
        }
        for label in moments.columns
        for stat in ("media", "desvío", "asimetría", "curtosis (exceso)")
    ]

    _density_rows = [
        {"model": "mu (Normal)", "value": float(params["mu"])},
        {"model": "sigma (Normal)", "value": float(params["sigma"])},
        {"model": "nu (t-Student)", "value": float(params["df"])},
        {"model": "loc (t-Student)", "value": float(params["loc"])},
        {"model": "scale (t-Student)", "value": float(params["scale"])},
    ]

    metrics_df = pd.DataFrame(
        [*_summary_rows, *_moment_rows, *_density_rows]
    )

# %%

# %%
# ## Publicación de métricas en el ledger
#
# Celda final que envía las métricas disponibles al ledger estandarizado de la
# tesis.  No recalcula resultados; captura las variables del notebook.

if __name__ == "__main__":
    from thesis.results import publish

    publish(
        experiment_id="spy/characterization",
        payload={"config": CONFIG, "metrics_df": metrics_df},
        chapter="SPY",
    )

# %%
