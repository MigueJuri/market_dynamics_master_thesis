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

# %%
# # Análisis de predictibilidad del PER sobre retornos futuros
#
# Este cuaderno cuantifica si el ratio PER (price/earnings) del S&P 500
# anticipa los retornos futuros del ETF ``SPY``.
#
# A diferencia de los otros análisis de este directorio (momentum,
# volatilidad, distancia a la media móvil), el PER **no** es una
# característica *windowed*: no hay eje $\tau_p$ de lookback.  Es un
# **nivel puntual** en $t$, disponible mensualmente y propagado al
# calendario de sesiones con *forward fill*.
#
# El objetivo es el retorno futuro acumulado
#
# .. math:: F_t(\tau_f) = \log P_{t+\tau_f} - \log P_{t+1}
#
# con horizontes $\tau_f > 21$ sesiones (> 1 mes).  El PER es un dato
# lento (cambia ~12 veces al año) mientras que el objetivo se mueve a
# 21 sesiones, así que la comparación requiere horizontes largos para
# que la señal tenga tiempo de materializarse.
#
# **Riesgos tratados** (ver ``docs/plans/per-forward-return-analysis.md``):
#
# 1. *Punto temporal del PER*: el CSV está fechado el día 1 de mes pero el
#    valor puede ser de cierre de mes (fechado "as-of").  Se comparan la
#    variante sin desplazar con una variante conservadora de un mes de
#    retraso.
# 2. *Empates de rango*: con *forward fill* el PER es constante ~21 días
#    seguidos; por eso sólo se usan $\tau_f > 21$ sesiones y se reportan
#    Pearson y Spearman (además de dos submuestreos de robustez).
# 3. *Outliers* (PER ≈ 124 en 2009, ≈ 46 en 2001): se reportan también
#    $\log(\mathrm{PER})$ y el *earnings yield* $1/\mathrm{PER}$.
# 4. *Ventanas solapadas*: los retornos futuros se solapan, por lo que no
#    se hace inferencia formal; se añade un submuestreo con paso $\tau_f$.

# %%
import sys
from pathlib import Path

# Permite ejecutar el archivo desde ``notebooks/`` o desde ``notebooks/feature_analysis/``.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # raíz del repositorio (``thesis/``)
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.config import DATA_DIR, END, START, TICKER, TRADING_DAYS
from thesis.market_data import load_single_price
from thesis.plotting import plot_feature_boxplots
from thesis.timeseries import log_returns

# %%
# ## Carga de datos: precios de SPY y PER mensual del S&P 500
#
# El PER se lee de ``data/sp500_pe_mensual.csv`` (columnas ``fecha, pe``,
# mensual desde 1871) y se propaga al calendario de sesiones con
# ``reindex(..., method="ffill")``: cada día de trading adopta el último
# valor mensual conocido en ese momento.

price = load_single_price(TICKER, START, END, DATA_DIR)
log_ret = log_returns(price)
log_price = np.log(price)

PE_CSV = DATA_DIR / "sp500_pe_mensual.csv"
pe = (
    pd.read_csv(PE_CSV, parse_dates=["fecha"])
    .set_index("fecha")["pe"]
    .sort_index()
)

# Propagación al calendario diario de sesiones (sin look-ahead: el valor
# de un mes sólo se usa a partir de su fecha en el CSV).
pe_daily = pe.reindex(price.index, method="ffill")

# Comprobaciones de integridad del índice y de la cobertura.
assert pe_daily.index.equals(price.index), "pe_daily debe compartir índice con price"
first_valid = pe_daily.first_valid_index()
assert pe_daily.loc[first_valid:].notna().all(), "No debe haber NaN tras el primer PER válido"

# Nº exacto de observaciones mensuales que aportan valor al calendario diario.
_pos = pe.index.searchsorted(price.index, side="right") - 1
n_monthly_used = int(np.unique(_pos[_pos >= 0]).size)
n_valid_days = int(pe_daily.notna().sum())
# Días cuyo valor procede de una observación mensual anterior (propagados).
n_filled_days = n_valid_days - n_monthly_used

print(f"Ticker                : {TICKER}")
print(f"Rango de precios      : {price.index[0]:%Y-%m-%d} a {price.index[-1]:%Y-%m-%d}")
print(f"Observaciones         : {len(price):,} precios | {len(log_ret):,} retornos")
print(f"Rango del PER (mensual): {pe.index[0]:%Y-%m-%d} a {pe.index[-1]:%Y-%m-%d} ({len(pe):,} obs.)")
print(f"PER min / max (muestra): {pe.loc[pe.index >= price.index[0]].min():.2f} / "
      f"{pe.loc[pe.index >= price.index[0]].max():.2f}")
print(f"Primer PER en el calendario: {first_valid:%Y-%m-%d}")
print(f"Días con PER          : {n_valid_days:,} ({100 * n_valid_days / len(price):.1f}% de las sesiones)")
print(f"Días rellenados (ffill): {n_filled_days:,} "
      f"({100 * n_filled_days / n_valid_days:.1f}% de los días con PER)")
print(f"Obs. mensuales usadas : {n_monthly_used:,}")

# %%
# ## Variantes de la característica
#
# * ``pe_daily``: nivel del PER propagado al calendario diario (supuesto
#   "as-of": el valor fechado el día 1 de mes es válido desde ese día).
# * ``pe_daily_shifted``: ``pe_daily.shift(1)``, un día de sesión de
#   retraso.  Como el PER sólo cambia el primer día de cada mes, el
#   efecto real es **retardar un día** la incorporación de cada valor
#   mensual; no es un retraso de un mes.
# * ``pe_daily_lag1m``: retraso real de **un mes** (se desplaza la serie
#   mensual una observación antes de propagarla).  Es la variante
#   conservadora frente al riesgo de fechado "as-of".
# * ``log_pe_daily`` y ``log_pe_daily_shifted``: logaritmo del nivel, para
#   comprimir los outliers (PER ≈ 124 en 2009).
# * ``earnings_yield = 1 / PER``: unidades 1/año, directamente comparables
#   con el retorno.

pe_daily_shifted = pe_daily.shift(1)
pe_daily_lag1m = pe.shift(1).reindex(price.index, method="ffill")

log_pe_daily = np.log(pe_daily)
log_pe_daily_shifted = np.log(pe_daily_shifted)
earnings_yield = 1.0 / pe_daily

assert pe_daily.notna().sum() == n_valid_days
# El retraso de un mes no pierde días: el primer día de la muestra ya usa
# el PER del mes anterior al del CSV.
assert pe_daily_lag1m.notna().sum() == n_valid_days
print(f"pe_daily_shifted  : {int(pe_daily_shifted.notna().sum()):,} valores "
      f"(1er día sin valor por el shift)")
print(f"pe_daily_lag1m    : {int(pe_daily_lag1m.notna().sum()):,} valores "
      f"(PER del mes anterior)")

# %%
# ## Retorno futuro acumulado
#
# ``F_t(tau_f) = log P_{t+tau_f} - log P_{t+1}``: el mismo criterio que
# usa ``forward_return`` en ``momentum_analysis.py`` (el retorno empieza
# en ``t+1``, nunca en ``t``).
#
# Todos los horizontes son **mayores de 21 sesiones** (> 1 mes): con
# horizontes más cortos el objetivo apenas cubre un cambio del PER y
# queda dominado por el ruido (riesgo 2 del plan).

MIN_HORIZON_SESSIONS = 21
TAU_F = [42, 63, 126, 252, 504, 756]  # sesiones: 2, 3, 6, 12, 24 y 36 meses
assert all(tau > MIN_HORIZON_SESSIONS for tau in TAU_F), "τ_f debe superar 21 sesiones"


def forward_return(log_price_series: pd.Series, tau_f: int) -> pd.Series:
    """Retorno log acumulado entre ``t+1`` y ``t+tau_f`` (anclado en ``t``)."""
    return log_price_series.shift(-tau_f) - log_price_series.shift(-1)


fwd_by_tau = {tau: forward_return(log_price, tau) for tau in TAU_F}

print("Horizontes (sesiones → años):")
for tau in TAU_F:
    print(f"  τ_f = {tau:>3d} sesiones = {tau / TRADING_DAYS:>4.1f} años")

# %%
# ## Correlación ρ(τ_f) entre el PER y el retorno futuro
#
# Para cada horizonte se correlaciona cada variante de la característica
# con ``F_t(τ_f)`` usando **sólo las posiciones válidas** (pares finitos).
#
# Se reportan tres esquemas por robustez:
#
# * **muestra diaria**: todos los días de trading con dato válido.
# * **paso τ_f**: submuestreo cada ``τ_f`` sesiones, de modo que los
#   bloques de retorno futuro no se solapen (riesgo 5).
# * **mensual**: una sola observación por mes (primer día de trading de
#   cada mes), que elimina los empates de rango massivos del *forward
#   fill* (riesgo 2).

VARIANTS: dict[str, pd.Series] = {
    "PER (nivel)": pe_daily,
    "PER (nivel, shift 1d)": pe_daily_shifted,
    "log(PER)": log_pe_daily,
    "log(PER, shift 1d)": log_pe_daily_shifted,
    "1/PER (earnings yield)": earnings_yield,
    "PER (lag 1 mes)": pe_daily_lag1m,
}

# Primer día de trading de cada mes (para el submuestreo mensual).
month_first_days = pe_daily.groupby(pe_daily.index.to_period("M")).head(1).index


def _corr_row(feature: pd.Series, target: pd.Series, tau_f: int, name: str) -> dict:
    """Correlaciones de Pearson/Spearman de ``feature`` contra ``target``."""
    pair = pd.concat(
        [feature.rename("f"), target.rename("y")], axis=1
    ).dropna()
    sub = pair.iloc[::tau_f]                      # submuestreo con paso τ_f
    mon = pair.loc[pair.index.intersection(month_first_days)]  # submuestreo mensual
    return {
        "variante": name,
        "tau_f": tau_f,
        "tau_f_años": tau_f / TRADING_DAYS,
        "n": len(pair),
        "pearson": pair["f"].corr(pair["y"], method="pearson"),
        "spearman": pair["f"].corr(pair["y"], method="spearman"),
        "n_sub": len(sub),
        "pearson_sub": sub["f"].corr(sub["y"], method="pearson"),
        "spearman_sub": sub["f"].corr(sub["y"], method="spearman"),
        "n_m": len(mon),
        "pearson_m": mon["f"].corr(mon["y"], method="pearson"),
        "spearman_m": mon["f"].corr(mon["y"], method="spearman"),
    }


corr_rows = [
    _corr_row(feat, fwd_by_tau[tau], tau, name)
    for name, feat in VARIANTS.items()
    for tau in TAU_F
]
corr_table = pd.DataFrame(corr_rows)

print("Pearson (muestra diaria):")
print(corr_table.pivot(index="tau_f", columns="variante", values="pearson").round(3).to_string())
print()
print("Spearman (muestra diaria):")
print(corr_table.pivot(index="tau_f", columns="variante", values="spearman").round(3).to_string())

# %%
# ### Curva ρ(τ_f) para las cuatro variantes requeridas
#
# Las cuatro variantes obligatorias son la rejilla *nivel/shift* ×
# *raw/log*; las dos restantes son las comprobaciones de sensibilidad
# (outliers y fechado).  El shift mide la sensibilidad al fechado
# "as-of" y el log la sensibilidad a los outliers.

STYLES = {
    "PER (nivel)": {"color": "C0", "ls": "-"},
    "PER (nivel, shift 1d)": {"color": "C1", "ls": "-"},
    "log(PER)": {"color": "C2", "ls": "-"},
    "log(PER, shift 1d)": {"color": "C3", "ls": "-"},
    "1/PER (earnings yield)": {"color": "C4", "ls": "--"},
    "PER (lag 1 mes)": {"color": "C5", "ls": ":"},
}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True, constrained_layout=True)
for name, style in STYLES.items():
    sub = corr_table[corr_table["variante"] == name].sort_values("tau_f")
    for ax, method in zip(axes, ["pearson", "spearman"]):
        ax.plot(
            sub["tau_f_años"], sub[method],
            marker="o", markersize=3.5, linewidth=1.4, label=name, **style,
        )

for ax, ylabel in zip(axes, ["Correlación de Pearson", "Correlación de Spearman"]):
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(r"$\tau_f$ (años)")
    ax.set_ylabel(ylabel)
    # Se etiquetan los horizontes desde 3 meses: 2 y 3 meses (0.17 y 0.25
    # años) se solapan en el eje.
    ticks = [t for t in TAU_F if t > 42]
    ax.set_xticks([t / TRADING_DAYS for t in ticks])
    ax.set_xticklabels([f"{t / TRADING_DAYS:.2f}".rstrip("0").rstrip(".") for t in ticks])
    ax.legend(fontsize=7, ncol=2)
axes[0].set_title("PER → retorno futuro: Pearson")
axes[1].set_title("PER → retorno futuro: Spearman")
plt.show()

# %%
# ### Robustez: submuestreo con paso τ_f y muestreo mensual
#
# Sin inferencia formal (las ventanas se solapan); se comparan los tres
# esquemas de muestreo sobre la variante de referencia.

main_variant = "log(PER, shift 1d)"
robustez = corr_table[corr_table["variante"] == main_variant][
    ["tau_f", "n", "pearson", "n_sub", "pearson_sub", "n_m", "pearson_m",
     "spearman", "spearman_sub", "spearman_m"]
].set_index("tau_f")
print(f"Robustez — {main_variant}")
print(robustez.round(3).to_string())
print(
    "\nNota: el submuestreo con paso τ_f deja pocas observaciones en "
    "horizontes largos (n_sub), por lo que sus correlaciones son ruidosas "
    "y no deben leerse como estimaciones precisas."
)

# %%
# ## Grilla de scatter: PER × retorno futuro
#
# Eje X: ``log(PER)`` con un día de sesión de retraso (sin look-ahead),
# centrado en un PER de referencia **constante** (20) para que la línea
# vertical en cero sea interpretable (PER = 20).  Desplazar por una
# constante no altera las correlaciones.
#
# Eje Y: retorno futuro acumulado ``F_t(τ_f)``.
#
# El scatter se dibuja aquí de forma directa (y no con
# ``thesis.plotting.plot_scatter_grid``) porque el PER no tiene eje de
# lookback: la única característica daría una grilla de una sola columna,
# que el helper colapsa a un vector 1-D.  Se replican su convención
# (color por fecha con escala compartida y colorbar común).

PE_REFERENCE = 20.0
log_pe_rel_shifted = np.log(pe_daily_shifted / PE_REFERENCE)


def plot_pe_scatter_grid(
    feature: pd.Series,
    fwd_by_tau: dict[int, pd.Series],
    taus: list[int],
    ncols: int = 3,
    figsize: tuple[float, float] = (11.0, 7.0),
) -> None:
    """Scatter PER (eje X) × retorno futuro (eje Y) coloreado por fecha."""
    nrows = int(np.ceil(len(taus) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False,
    )
    flat = axes.ravel()

    frames = [
        pd.concat(
            [feature.rename("f"), fwd_by_tau[tau].rename("y")], axis=1
        ).dropna()
        for tau in taus
    ]
    if any(frame.empty for frame in frames):
        print("No hay datos comunes entre el PER y el retorno futuro.")
        return

    vmin = mdates.date2num(min(frame.index.min() for frame in frames))
    vmax = mdates.date2num(max(frame.index.max() for frame in frames))

    scatter = None
    for ax, tau, data in zip(flat, taus, frames):
        scatter = ax.scatter(
            data["f"], data["y"],
            c=mdates.date2num(data.index), cmap="viridis",
            vmin=vmin, vmax=vmax, s=2, alpha=0.6,
        )
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.8)  # PER = 20
        ax.set_title(rf"$\tau_f$={tau} sesiones ({tau / TRADING_DAYS:.1f} años)")
        ax.set_xlabel(rf"$\log(\mathrm{{PER}}/{PE_REFERENCE:.0f})$")
        ax.set_ylabel("Retorno futuro")

    for ax in flat[len(taus):]:
        ax.set_visible(False)

    cbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    cbar.set_label("Fecha")
    plt.show()


plot_pe_scatter_grid(log_pe_rel_shifted, fwd_by_tau, TAU_F)

# %%
# ## Boxplots del retorno futuro por bins del PER
#
# Se agrupa el retorno futuro en 10 bins de igual anchura sobre el rango
# de ``log(PER)`` (con un día de retraso) y se muestran las distribuciones
# junto con la ECDF del feature, para dos horizontes representativos.

for tau_f in (126, 504):
    pearson_corr = plot_feature_boxplots(
        log_pe_daily_shifted, fwd_by_tau[tau_f],
        n_quantiles=10,
        x_axis="bins",
        show_cumulative=True,
        feature_name=r"$\log(\mathrm{PER})$",
        target_name="Retorno futuro",
        target_window=tau_f,
    )
    print(f"Pearson (τ_f={tau_f} sesiones = {tau_f / TRADING_DAYS:.1f} años): "
          f"{pearson_corr:.3f}")

# %%
# ## Nota sobre la interpretación
#
# * El PER es un **nivel** sin memoria: su contenido predictivo, si lo
#   hay, es de largo plazo (regímenes de valoración), no de corto plazo.
# * La correlación instantánea entre un PER constante durante ~21 días y
#   un retorno futuro de 126 sesiones mezcla dos fuentes de ruido: los
#   empates del *forward fill* y el solapamiento de ventanas.  Por eso se
#   reportan los tres esquemas de muestreo.
# * No se construye ninguna estrategia: una correlación distinta de cero
#   no implica rentabilidad tras costes.
# * El desplazamiento de un mes (``PER (lag 1 mes)``) es la variante
#   relevante si el CSV está fechado "as-of" (valor de fin de mes
#   publicado el día 1 del mes siguiente): en ese caso el PER del mes en
#   curso no era información pública durante ese mes.
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
        experiment_id="feature_analysis/per",
        payload=_payload,
        chapter="PER",
    )

# %%
