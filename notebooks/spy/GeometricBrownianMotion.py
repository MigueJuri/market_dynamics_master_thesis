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

# %% [markdown]
# # Movimiento Geométrico Browniano: ¿describe a SPY?
#
# **¿Es consistente el S&P 500 con un GBM de parámetros constantes?**
#
# El movimiento geométrico browniano (GBM) es el modelo de referencia de Black–Scholes–Merton: el
# precio $S_t$ evoluciona según
#
# $$d(\ln S_t) = \left(\mu - \tfrac{1}{2}\sigma^2\right) dt + \sigma\, dW_t
# \qquad\Longrightarrow\qquad
# S_t = S_0 \exp\!\Big(\big(\mu-\tfrac{\sigma^2}{2}\big)t+\sigma W_t\Big),$$
#
# con $W_t$ un proceso de Wiener y $\mu$, $\sigma$ constantes. Trabajando con los log-retornos
# $x_t = \ln(S_t/S_{t-1})$, el Lema de Itô da la distribución exacta en cada paso:
#
# $$x_t \sim \mathcal{N}\!\left(\big(\mu - \tfrac{1}{2}\sigma^2\big)\Delta t,\; \sigma^2 \Delta t\right).$$
#
# Este cuaderno calibra $\mu$ y $\sigma$ con datos in-sample y pone a prueba, out-of-sample, las dos
# hipótesis que sostienen al modelo:
#
# | # | Se pone a prueba | Etapa |
# |---|------------------|-------|
# | 1 | La distribución exacta de $S_t$ contra la trayectoria realizada | 4 |
# | 2 | Cobertura empírica de las bandas de confianza | 5 |
# | 3 | Constancia de $\mu$ y $\sigma$ en el tiempo | 6 |
# | 4 | Independencia entre drift y volatilidad (efecto apalancamiento) | 7 |
#
# > Los *stylized facts* de los retornos (colas pesadas, asimetría, clustering de volatilidad) se
# > caracterizan en detalle en [`SPY_caracterizacion.py`](SPY_caracterizacion.py); acá se usan sólo
# > en la medida en que son necesarios para falsear el GBM.
#
# ### Estructura del cuaderno
#
# - **celda 0**: magics (`%autoreload 2`) para recargar `src/` en caliente;
# - **celda 1**: `CONFIG`, con **todos** los parámetros del análisis;
# - **celda 2**: datos (cierre ajustado cacheado + log-retornos + split);
# - **celda 3**: calibración MLE con *moving block bootstrap*;
# - **celda 4**: *fan chart* (densidad exacta del GBM);
# - **celda 5**: cobertura empírica de las bandas;
# - **celda 6**: $\mu$ y $\sigma$ en ventana móvil;
# - **celda 7**: drift vs. volatilidad (efecto apalancamiento);
# - **celda 8**: tabla resumen.
#
# El **cómputo** vive en `src/thesis/gbm.py` (importable, sin `matplotlib`); el **dibujo** vive
# aquí, para poder retocar colores, bines, escalas o `xlim` re-ejecutando **sólo** la celda
# afectada, sin reiniciar el kernel.
#
# ### Cómo correrlo
#
# ```bash
# uv run jupyter notebook notebooks/spy/GeometricBrownianMotion.py
# uv run python notebooks/spy/GeometricBrownianMotion.py   # modo script
# ```

# %% [markdown]
# ## Celda 0 — recarga en caliente de `src/`
#
# `%autoreload 2` hace que los cambios en `src/thesis/gbm.py` se reflejen en la **siguiente
# celda** sin reiniciar el kernel.  Los magics sólo existen en IPython: la guarda `get_ipython()`
# mantiene el archivo ejecutable como script (`uv run python notebooks/spy/GeometricBrownianMotion.py`).

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
# **Todos** los parámetros del análisis (datos, split, bootstrap, ventana móvil, colores, tamaños
# y rutas de salida) viven en este diccionario: cambiar un valor y re-ejecutar la celda de la
# figura afectada basta para rehacerla.

# %%
import sys
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from numpy import ma
from scipy import stats

# --- Ruta del repositorio (permite ejecutar el cuaderno desde cualquier CWD) --
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _HERE.parents[1] if (_HERE.parents[1] / "src" / "thesis").is_dir() else _HERE
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.characterization import block_bootstrap_ci
from thesis.config import PLOT_RC
from thesis.gbm import (
    BAND_LEVELS,
    gbm_coverage,
    gbm_density_grid,
    gbm_mle_params,
    gbm_quantile_bands,
    rolling_gbm_params,
)
from thesis.market_data import load_single_price
from thesis.timeseries import log_returns

#: Directorio de figuras (anclado a la raíz del repositorio, no al CWD).
OUTPUT_DIR = _ROOT / "outputs"

plt.rcParams.update(PLOT_RC)

CONFIG: dict[str, Any] = {
    # --- Datos ---
    "ticker": "SPY",
    "start": "1993-01-01",
    "end": "2026-01-01",
    # Fin del tramo in-sample / inicio de la validación out-of-sample.
    "split_date": "2016-01-01",
    "trading_days": 252,
    # --- Bootstrap por bloques (etapa 3) ---
    "block_size": 10,
    "n_boot": 3_000,
    "alpha": 0.05,
    "seed": 42,
    # --- Ventana móvil (etapa 6) ---
    "rolling_window": 60,
    # --- Bandas y grilla del fan chart (etapa 4) ---
    "band_levels": BAND_LEVELS,
    "n_price_bins": 200,
    # --- Figuras ---
    "figsize_fan": (10.0, 4.5),
    "figsize_rolling": (11.0, 6.0),
    "figsize_drift": (11.0, 4.2),
    "save_figures": True,
    "show_figure": True,
}

# %% [markdown]
# ## Celda 2 — datos y split in-sample / out-of-sample
#
# Precio de cierre **ajustado** del ETF `SPY` (misma serie y misma caché `data/` que
# `SPY_caracterizacion.py`: no se descarga dos veces).  El split `asof` ubica la sesión hábil más
# cercana a la fecha de corte: la calibración usa **sólo** el tramo in-sample y las bandas se
# proyectan sobre el tramo de validación.

# %%
price = load_single_price(
    ticker=CONFIG["ticker"],
    start=CONFIG["start"],
    end=CONFIG["end"],
)
log_ret = log_returns(price)

split_idx = price.index.asof(pd.Timestamp(CONFIG["split_date"]))
train_ret = log_ret.loc[:split_idx]  # in-sample: calibración
test_price = price.loc[split_idx:]  # out-of-sample: "realidad" contra la que se valida
S0 = float(price.loc[split_idx])  # precio de arranque de la proyección

print(
    f"Serie completa   : {price.index[0]:%Y-%m-%d} a {price.index[-1]:%Y-%m-%d}  "
    f"({len(price):,} precios)"
)
print(
    f"Entrenamiento    : {train_ret.index[0]:%Y-%m-%d} a {train_ret.index[-1]:%Y-%m-%d}  "
    f"({len(train_ret):,} retornos)"
)
print(
    f"Validación (OOS) : {test_price.index[0]:%Y-%m-%d} a {test_price.index[-1]:%Y-%m-%d}  "
    f"({len(test_price):,} precios)"
)
print(f"S0 (precio de arranque de la proyección): {S0:.2f} USD")

# %% [markdown]
# ## Celda 3 — calibración (MLE + bootstrap por bloques)
#
# Los estimadores de máxima verosimilitud para un GBM de parámetros constantes son la media y el
# desvío muestral de los log-retornos, anualizados. El drift *geométrico* $\tilde\mu = \mathbb{E}[x_t]/\Delta t$
# es lo que se estima directamente; el drift *aritmético* $\mu$ (el que aparece en la SDE) requiere la
# corrección de Itô $\mu = \tilde\mu + \sigma^2/2$.
#
# Como en `SPY_caracterizacion.py`, el intervalo de confianza se calcula con un *moving block
# bootstrap* en lugar de la fórmula asintótica $\mathrm{SE}(\hat\sigma)\approx\hat\sigma/\sqrt{2N}$:
# esa fórmula asume retornos i.i.d., que es exactamente la hipótesis que este cuaderno está poniendo
# a prueba, así que usarla para el propio intervalo de confianza sería circular.  La semilla es
# **explícita** (`seed=CONFIG["seed"]`): la versión original dependía del `np.random.seed` global.

# %%
point, ci = block_bootstrap_ci(
    train_ret.to_numpy(),
    gbm_mle_params,
    block_size=CONFIG["block_size"],
    reps=CONFIG["n_boot"],
    alpha=CONFIG["alpha"],
    seed=CONFIG["seed"],
)
drift_geo, sigma_est, mu_est = point

names = ["Drift geométrico (μ̃)", "Sigma (σ)", "Drift aritmético (μ)"]
for name, val, lo, hi in zip(names, point, ci[0], ci[1]):
    print(
        f"{name:<24}: {val:7.4f}   (IC {1 - CONFIG['alpha']:.0%}: [{lo:.4f}, {hi:.4f}])"
    )

# %% [markdown]
# ## Celda 4 — distribución exacta y validación (*fan chart*)
#
# Con $\mu$ y $\sigma$ calibrados sobre el período de entrenamiento, la distribución de $S_t$ para
# $t>$ fecha de corte es exacta —no hace falta simular Monte Carlo—, porque el Lema de Itô ya la da
# en forma cerrada:
#
# $$\ln S_t \sim \mathcal{N}\big(\ln S_0 + \tilde\mu\, t,\; \sigma^2 t\big), \qquad t = 0, \Delta t, 2\Delta t,\ldots$$
#
# Se grafica la densidad de esa distribución en cada instante como un mapa de calor (*fan chart* o
# río de probabilidad), enmascarando fuera del intervalo central del 95 %, contra la trayectoria
# realmente observada.  La densidad es **relativa**: cada nivel de precio se normaliza por su
# máximo a lo largo del tiempo, de modo que el color indica *cuándo* fue plausible cada precio.

# %%
dt = 1.0 / CONFIG["trading_days"]
# Grilla temporal con conteo entero (sin paso flotante acumulado).
t_grid = np.arange(len(test_price), dtype=float) * dt

bands = gbm_quantile_bands(
    S0,
    drift_geo,
    sigma_est,
    t_grid,
    levels=CONFIG["band_levels"],
)
median_path = bands[0.50]

log_grid, density = gbm_density_grid(
    S0,
    drift_geo,
    sigma_est,
    t_grid,
    n_bins=CONFIG["n_price_bins"],
)

# Enmascara fuera de la banda central del 95 % (densidad: n_bins precios x n_t instantes).
outside_95 = (log_grid[:, None] < np.log(bands[0.025])[None, :]) | (
    log_grid[:, None] > np.log(bands[0.975])[None, :]
)
density_masked = ma.masked_where(outside_95, density)

cmap = mcolors.ListedColormap(plt.cm.Blues(np.linspace(0, 0.8, 256)))
cmap.set_bad(alpha=0)

fig_fan, ax = plt.subplots(figsize=CONFIG["figsize_fan"], dpi=300)
x_dates = mdates.date2num(test_price.index.to_numpy())
mesh = ax.pcolormesh(
    x_dates,
    np.exp(log_grid),
    density_masked,
    cmap=cmap,
    shading="gouraud",
    zorder=1,
)
fig_fan.colorbar(mesh, ax=ax, label="Densidad de probabilidad relativa")

ax.plot(
    test_price.index,
    median_path,
    linestyle="--",
    linewidth=1,
    color="#7FDBFF",
    label="Mediana GBM (post-calibración)",
    zorder=4,
)
ax.plot(
    price.index,
    price.values,
    linewidth=1,
    color="#02001E",
    label=CONFIG["ticker"],
    zorder=5,
)
ax.axvline(
    price.loc[:split_idx].index[-1],
    color="gray",
    lw=0.8,
    ls=":",
    label="Fin de entrenamiento",
)

ax.set_yscale("log")
ax.set_yticks([25, 50, 100, 200, 400, 800])
formatter = mticker.ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(price.index[0], price.index[-1])
ax.set_ylabel("Precio (USD, log)")
ax.set_xlabel("Fecha")
ax.legend(loc="upper left")
fig_fan.tight_layout()

if CONFIG["save_figures"]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_fan.savefig(
        OUTPUT_DIR / "spy_gbm_fan_chart.png",
        dpi=300,
        bbox_inches="tight",
    )
if CONFIG["show_figure"] and _INTERACTIVE:
    plt.show()
else:
    plt.close("all")

# %% [markdown]
# ## Celda 5 — cobertura empírica de las bandas
#
# Una forma cuantitativa —no sólo visual— de poner a prueba el modelo: si el GBM calibrado fuera
# correcto, el precio observado debería caer dentro de la banda central del $p\%$ exactamente el
# $p\%$ de los días hábiles del período de validación. Se lo compara contra el 50 %, 80 % y 95 %
# nominal.

# %%
coverage = gbm_coverage(test_price.to_numpy(), bands, levels=(0.5, 0.8, 0.95))

coverage_table = coverage.assign(
    **{
        "Banda nominal": coverage["nominal"].map("{:.0%}".format),
        "Cuantil inf.": coverage["lower_quantile"].map("{:.3f}".format),
        "Cuantil sup.": coverage["upper_quantile"].map("{:.3f}".format),
        "Cobertura empírica": coverage["coverage"].map("{:.1%}".format),
    }
)[["Banda nominal", "Cuantil inf.", "Cuantil sup.", "Cobertura empírica"]]
print(coverage_table.to_string(index=False))

# %% [markdown]
# ## Celda 6 — estacionariedad: $\mu$ y $\sigma$ en ventana móvil
#
# El GBM supone $\mu$ y $\sigma$ constantes en el tiempo. Se los reestima en una ventana móvil
# *trailing* de 60 sesiones (~3 meses) sobre toda la serie y se los compara contra el valor
# estático (ajustado una sola vez, con toda la muestra): si el modelo fuera correcto, la curva
# local debería oscilar sin tendencia alrededor de la línea estática. No es lo que se observa.
#
# Las estadísticas de esta celda (y de la siguiente) son **descriptivas**: se calculan sobre toda
# la muestra y **no** son señales.

# %%
rolling = rolling_gbm_params(
    log_ret,
    window=CONFIG["rolling_window"],
    trading_days=CONFIG["trading_days"],
)
rolling_sigma = rolling["sigma"]
rolling_drift = rolling["mu"]

static_sigma = float(log_ret.std() * np.sqrt(CONFIG["trading_days"]))
static_drift = float(log_ret.mean() * CONFIG["trading_days"] + 0.5 * static_sigma**2)

fig_roll, (ax_sigma, ax_drift) = plt.subplots(
    2,
    1,
    figsize=CONFIG["figsize_rolling"],
    sharex=True,
)

ax_sigma.plot(
    rolling_sigma.index,
    rolling_sigma.values,
    linewidth=0.8,
    color="#1f77b4",
    label=f"σ local ({CONFIG['rolling_window']}d)",
)
ax_sigma.axhline(
    static_sigma,
    color="red",
    ls="--",
    lw=1.0,
    label=f"σ estática ({static_sigma:.2f})",
)
ax_sigma.set_ylabel(r"Volatilidad anualizada $\sigma$")
ax_sigma.set_title("Volatilidad local vs. estática")
ax_sigma.legend(loc="upper left")

ax_drift.plot(
    rolling_drift.index,
    rolling_drift.values,
    linewidth=0.8,
    color="#2ca02c",
    label=f"μ local ({CONFIG['rolling_window']}d)",
)
ax_drift.axhline(
    static_drift,
    color="red",
    ls="--",
    lw=1.0,
    label=f"μ estático ({static_drift:.2f})",
)
ax_drift.axhline(0, color="black", lw=0.5)
ax_drift.set_ylabel(r"Drift anualizado $\mu$")
ax_drift.set_xlabel("Fecha")
ax_drift.set_title("Drift local vs. estático")
ax_drift.legend(loc="upper left")

fig_roll.tight_layout()

if CONFIG["save_figures"]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_roll.savefig(
        OUTPUT_DIR / "spy_gbm_rolling_params.png",
        dpi=300,
        bbox_inches="tight",
    )
if CONFIG["show_figure"] and _INTERACTIVE:
    plt.show()
else:
    plt.close("all")

print(
    f"σ local — mín {rolling_sigma.min():.2f}, máx {rolling_sigma.max():.2f}  "
    f"(estática: {static_sigma:.2f})"
)
print(
    f"μ local — mín {rolling_drift.min():.2f}, máx {rolling_drift.max():.2f}  "
    f"(estático: {static_drift:.2f}, cambia incluso de signo)"
)

# %% [markdown]
# ## Celda 7 — drift vs. volatilidad
#
# El GBM asume $\mu$ y $\sigma$ independientes. Si estuvieran correlacionados —en particular, si la
# volatilidad sube cuando el drift cae (el *efecto apalancamiento* de Black, 1976: una caída de precio
# aumenta el ratio deuda/equity y con él el riesgo percibido)— el modelo de parámetros constantes e
# independientes pierde justificación incluso localmente.
#
# Se usa el mismo par de series de la etapa 6 ($\mu_t$, $\sigma_t$ en ventana móvil) y se calcula
# su correlación de Pearson. El color del panel izquierdo codifica el tiempo, para ver si la relación
# es estable o dominada por un puñado de episodios (2008, 2020).

# %%
valid = rolling_drift.notna() & rolling_sigma.notna()
drift_v = rolling_drift[valid].to_numpy(dtype=float)
sigma_v = rolling_sigma[valid].to_numpy(dtype=float)
dates_v = mdates.date2num(rolling_drift[valid].index.to_numpy())

r, p_value = stats.pearsonr(drift_v, sigma_v)
slope, intercept = np.polyfit(drift_v, sigma_v, 1)

fig_drift, (ax_time, ax_density) = plt.subplots(
    1,
    2,
    figsize=CONFIG["figsize_drift"],
)

sc = ax_time.scatter(drift_v, sigma_v, c=dates_v, cmap="viridis", s=5, alpha=0.5)
cbar = fig_drift.colorbar(sc, ax=ax_time)
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
cbar.set_label("Año")
ax_time.set_xlabel(r"$\mu$ local (anualizado)")
ax_time.set_ylabel(r"$\sigma$ local (anualizado)")
ax_time.set_title("Coloreado por fecha")

hb = ax_density.hexbin(drift_v, sigma_v, gridsize=40, cmap="Blues", mincnt=1)
fig_drift.colorbar(hb, ax=ax_density, label="Observaciones por hexágono")
x_fit = np.linspace(drift_v.min(), drift_v.max(), 100)
ax_density.plot(
    x_fit,
    slope * x_fit + intercept,
    color="red",
    lw=1.5,
    label=f"Ajuste lineal (pendiente {slope:.2f})",
)
ax_density.set_xlabel(r"$\mu$ local (anualizado)")
ax_density.set_title("Densidad de observaciones")
ax_density.legend(loc="upper right")

fig_drift.suptitle(
    rf"$\mu$ vs. $\sigma$ locales — Pearson $r$={r:.2f}, p={p_value:.1e}",
    y=1.03,
)
fig_drift.tight_layout()

if CONFIG["save_figures"]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_drift.savefig(
        OUTPUT_DIR / "spy_gbm_drift_vs_sigma.png",
        dpi=300,
        bbox_inches="tight",
    )
if CONFIG["show_figure"] and _INTERACTIVE:
    plt.show()
else:
    plt.close("all")

# %% [markdown]
# ## Celda 8 — tabla resumen

# %%
coverage_text = ", ".join(f"{value:.1%}" for value in coverage["coverage"])
summary = pd.DataFrame(
    [
        (
            "σ constante",
            (
                f"rango local [{rolling_sigma.min():.2f}, {rolling_sigma.max():.2f}] vs. "
                f"estática {static_sigma:.2f}"
            ),
            "Rechazada",
        ),
        (
            "μ constante",
            (
                f"rango local [{rolling_drift.min():.2f}, {rolling_drift.max():.2f}], "
                "cambia incluso de signo"
            ),
            "Rechazada",
        ),
        (
            "μ, σ independientes",
            f"Pearson r={r:.2f} (p={p_value:.1e})",
            "Rechazada" if p_value < CONFIG["alpha"] else "No rechazada",
        ),
        (
            "Cobertura empírica ≈ nominal",
            coverage_text,
            "Ver tabla de la etapa 5",
        ),
    ],
    columns=["Hipótesis del GBM", "Evidencia", "Veredicto"],
)
print(summary.to_string(index=False))

# %% [markdown]
# ### Conclusiones
#
# 1. La distribución exacta de $S_t$ bajo GBM calibrado captura razonablemente el orden de magnitud
#    del rango de precios, pero la cobertura empírica de las bandas se aparta de la nominal — el
#    modelo subestima el riesgo en los extremos (consistente con las colas pesadas documentadas en
#    `SPY_caracterizacion.py`).
# 2. Tanto $\mu$ como $\sigma$ varían sustancialmente en el tiempo: la hipótesis de parámetros
#    constantes no se sostiene ni siquiera en ventanas de pocos meses.
# 3. La relación entre drift y volatilidad locales es consistente con un efecto apalancamiento
#    (drift bajo ⟷ volatilidad alta), lo que viola también la independencia asumida entre ambos.
#
# En conjunto, el GBM de parámetros constantes funciona como *benchmark* — es la distribución nula
# frente a la que se contrastan modelos con volatilidad estocástica o condicional (ARCH/GARCH,
# Heston) — pero no como descripción adecuada de SPY.
#
# ---
#
# ### Referencias
#
# - Osborne, M. F. M. (1959). *Brownian Motion in the Stock Market*. Operations Research, 7(2).
# - Black, F., Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3).
# - Merton, R. C. (1973). *Theory of Rational Option Pricing*. Bell Journal of Economics, 4(1).
# - Black, F. (1976). *Studies of Stock Price Volatility Changes*. Proceedings of the ASA.
# - Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General Stationary Observations*. Annals of Statistics, 17(3).
