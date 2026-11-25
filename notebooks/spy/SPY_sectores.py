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
# # Composición sectorial del S&P 500 y del SPY
#
# Este cuaderno estudia la **composición sectorial** del mercado estadounidense
# en cuatro niveles:
#
# 1. **snapshot actual**: los pesos sectoriales del ETF `SPY` (los publica
#    yfinance) en un gráfico de anillo (*donut*);
# 2. **reconstrucción histórica**: la composición sectorial del S&P 500 por
#    capitalización de mercado, reconstruida a partir de la lista actual de
#    constituyentes (Wikipedia), los precios (yfinance) y las acciones en
#    circulación (yfinance);
# 3. **entropía de Shannon**: cómo se ha diversificado (o concentrado) la
#    composición sectorial a lo largo del tiempo;
# 4. **HHI por empresa**: cómo se ha concentrado (o diversificado) la
#    capitalización del S&P 500 en unas pocas compañías.
#
# La lógica reutilizable vive en `src/thesis/sector_composition.py` (datos,
# ponderación y entropía); aquí el cuaderno la coordina, **por etapas**, con
# una celda por sección del pipeline.  El **código de dibujo de las figuras
# está en el cuaderno** (no en el paquete), para poder retocar etiquetas,
# escalas y colores sin tocar `src/thesis/`.
#
# .. warning::
#    La reconstrucción histórica es una **aproximación**: usa la lista de
#    constituyentes y las etiquetas GICS **actuales** y mantiene las acciones
#    en circulación constantes en el tiempo.  Sufre **sesgo de supervivencia**
#    (las empresas que salen del índice desaparecen de toda la historia) y
#    **clasificación sectorial retroactiva** (p. ej. "Servicios de
#    Comunicación" no existía como sector GICS antes de 2018).  Es una
#    herramienta descriptiva de la rotación sectorial, no una reconstrucción
#    point-in-time.

# %%
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
from thesis.sector_composition import (
    GICS_SECTOR_ES,
    MINOR_SECTOR_THRESHOLD,
    PRICE_CACHE_FILENAME_TMPL,
    SECTOR_COMPOSITION_CACHE_FILENAME,
    aggregate_small_sectors,
    build_sector_weights,
    company_hhi,
    download_price_history,
    fetch_spy_sector_weights,
    get_shares_outstanding,
    get_sp500_constituents,
    price_cache_path,
    sector_entropy,
)

#: Figuras de este cuaderno: ``figures/SPY_sectores/`` en la raíz del repositorio.
FIG_DIR = _ROOT / "figures" / "SPY_sectores"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas

# %% [markdown]
# ## Configuración
#
# Todos los parámetros del experimento viven aquí: se cambian **en sitio** y
# se re-ejecuta la celda de la etapa correspondiente.

# %%
CONFIG: dict[str, Any] = {
    # Activo del snapshot actual
    "ticker": "SPY",
    # Umbral (fracción) por debajo del cual un sector se agrupa en "Otros"
    "minor_threshold": 0.05,#MINOR_SECTOR_THRESHOLD,
    # Traducción de los sectores GICS al español
    "translation": GICS_SECTOR_ES,
    # Ventana de la reconstrucción histórica
    "start_date": "2000-01-01",
    # Fecha final: ``None`` -> hoy (sin caché de precios: la ventana no se
    # puede congelar en el nombre del archivo).  Pon una fecha explícita
    # (p. ej. "2026-01-01") para reutilizar la caché de precios.
    "end_date": None,
    "resample_freq": "ME",  # snapshots de fin de mes ("QE" para trimestral)
    # Descarga de precios
    "batch_size": 100,  # tickers por petición a yfinance
    "sleep_between_batches": 1.0,  # segundos entre lotes
    # Caché del frame de precios (CSV en ``data/``): sólo se usa si
    # ``end_date`` es explícito.
    "use_price_cache": True,
    "refresh_prices": False,
    "prices_cache_filename": PRICE_CACHE_FILENAME_TMPL,
    # Caché de las acciones en circulación (CSV en ``data/``)
    "use_shares_cache": True,
    "refresh_shares": False,
    # Caché de los pesos sectoriales históricos (CSV en ``data/``)
    "use_cache": False,
    "save_cache": True,
    "cache_filename": SECTOR_COMPOSITION_CACHE_FILENAME,
    # Figuras
    "show_figure": True,
    "save_figures": True,
}

# %% [markdown]
# ## Etapa 1: snapshot actual (donut)
#
# Descarga los pesos sectoriales que publica yfinance para el ETF `SPY`
# (`funds_data.sector_weightings`), agrupa en "Otros" los sectores por debajo
# del umbral y dibuja el **anillo** con el código de `matplotlib` escrito
# aquí mismo (etiquetas en español).

# %%
if __name__ == "__main__":
    # --- Datos: pesos sectoriales actuales (fracción) ---
    spy_weights = fetch_spy_sector_weights(CONFIG["ticker"])
    snapshot = aggregate_small_sectors(
        spy_weights,
        threshold=CONFIG["minor_threshold"],
        translation=CONFIG["translation"],
    )

    print(f"Pesos sectoriales de {CONFIG['ticker']} (yfinance):")
    print(spy_weights.to_string())
    print(f"\nSuma de los pesos: {spy_weights.sum():.4f}")

    # --- Figura: anillo (donut) ---
    labels = snapshot.index.tolist()
    sizes = snapshot.to_numpy(dtype=float)

    # Paleta: colormap perceptual; "Otros" en gris neutro al final.
    n = len(labels)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.05, 0.92, n)]
    for i, lab in enumerate(labels):
        if lab.strip().lower() in ("otros", "others"):
            colors[i] = "#B0B5BA"

    donut_fig, donut_ax = plt.subplots(figsize=(W, 0.45 * W))
    wedges, _, autotexts = donut_ax.pie(
        sizes,
        colors=colors,
        autopct=lambda p: rf"{p:.1f}\%" if p >= 4 else "",
        startangle=90,
        counterclock=False,
        pctdistance=0.80,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
    )
    for text, wedge in zip(autotexts, wedges):
        r, g, b, _ = wedge.get_facecolor()
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        text.set_color("black" if lum > 0.55 else "white")

    # Texto central: ticker y subtítulo.
    donut_ax.text(
        0, 0.06, CONFIG["ticker"],
        ha="center", va="center", color="#222222",
    )
    donut_ax.text(
        0, -0.12, "Pesos sectoriales",
        ha="center", va="center", color="#666666",
    )

    # Leyenda con el porcentaje de cada sector.
    legend_labels = [rf"{lab}  ·  {100 * s:.1f}\%" for lab, s in zip(labels, sizes)]
    donut_ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        handlelength=1.2,
        handleheight=1.2,
        labelspacing=0.6,
    )

    # Nota sobre "Otros", a la derecha bajo la leyenda.
    donut_ax.text(
        1.02, 0.02,
        f"``Otros'': sectores $<$ {100 * CONFIG['minor_threshold']:.0f}\\%\nagrupados",
        transform=donut_ax.transAxes,
        ha="left", va="bottom", color="#666666", style="italic",
    )

    donut_ax.axis("equal")

    if CONFIG["save_figures"]:
        donut_fig.savefig(
            FIG_DIR / "spy_sector_weights.pdf", facecolor="white",
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ## Etapa 2: datos históricos
#
# La etapa más pesada (red): lista de constituyentes del S&P 500 desde
# Wikipedia, precios de cierre ajustado de todos los tickers (por lotes) y
# acciones en circulación (best-effort).  Es la **única** celda que toca la
# red: al re-ejecutar las etapas siguientes no se vuelve a descargar nada.
#
# Los precios y las acciones se cachean en `data/` (un único CSV por
# ventana), de modo que re-ejecutar esta celda **no** vuelve a bajar ~500
# tickers.  La caché de precios exige una `end_date` **explícita** (con
# `None` la ventana sería "hasta hoy" y no se puede congelar en el nombre
# del archivo); `refresh_prices=True` fuerza la re-descarga.

# %%
if __name__ == "__main__":
    print("Descargando la lista de constituyentes del S&P 500...")
    constituents = get_sp500_constituents()
    tickers = constituents["Symbol"].tolist()
    sector_map = constituents.set_index("Symbol")["Sector"]
    print(f"Constituyentes: {len(tickers)}")

    if CONFIG["use_price_cache"] and CONFIG["end_date"] is not None:
        print(
            "Caché de precios: "
            f"{price_cache_path(Path(DATA_DIR), CONFIG['start_date'], CONFIG['end_date'])}"
        )

    print(
        f"Descargando precios de {len(tickers)} tickers desde "
        f"{CONFIG['start_date']} (puede tardar varios minutos)..."
    )
    prices = download_price_history(
        tickers,
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        batch_size=CONFIG["batch_size"],
        sleep=CONFIG["sleep_between_batches"],
        cache_dir=Path(DATA_DIR) if CONFIG["use_price_cache"] else None,
        refresh=CONFIG["refresh_prices"],
    )
    # Snapshots de fin de mes (el último precio disponible de cada mes).
    # El remuestreo se aplica **después** de cargar: cambiar
    # ``resample_freq`` no obliga a volver a descargar.
    prices = prices.resample(CONFIG["resample_freq"]).last()
    print(f"Precios: {prices.shape[0]} fechas × {prices.shape[1]} tickers")

    print("Descargando acciones en circulación (actuales, constantes)...")
    shares = get_shares_outstanding(
        tickers,
        cache_dir=Path(DATA_DIR) if CONFIG["use_shares_cache"] else None,
        refresh=CONFIG["refresh_shares"],
    )
    print(f"Tickers con acciones conocidas: {int(shares.notna().sum())}")

# %% [markdown]
# ## Etapa 3: pesos sectoriales
#
# Pondera la capitalización de mercado (`precio · acciones`) por sector y
# devuelve porcentajes que suman 100 en cada fecha.  El resultado se puede
# guardar en `data/<cache_filename>` para no repetir el cálculo.

# %%
if __name__ == "__main__":
    cache_path = Path(DATA_DIR) / CONFIG["cache_filename"]

    if CONFIG["use_cache"] and cache_path.exists():
        sector_weights = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Pesos sectoriales leídos de la caché: {cache_path}")
    else:
        sector_weights = build_sector_weights(prices, shares, sector_map)
        if CONFIG["save_cache"]:
            Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
            sector_weights.to_csv(cache_path)
            print(f"Pesos sectoriales guardados en: {cache_path}")

    print(f"Sectores: {list(sector_weights.columns)}")
    print(f"Rango: {sector_weights.index[0]:%Y-%m} — {sector_weights.index[-1]:%Y-%m}")
    print("\nÚltima composición disponible (%):")
    print(sector_weights.iloc[-1].round(2).to_string())

# %% [markdown]
# ## Etapa 4: figura de área apilada al 100%
#
# Evolución de la composición sectorial: los sectores se ordenan por peso
# medio (de mayor a menor) para que el gráfico sea legible, y se anotan a la
# derecha los sectores con peso superior al 2% en la última fecha.

# %%
if __name__ == "__main__":
    # Orden de los sectores: de mayor a menor peso medio en el histórico.
    order = sector_weights.mean().sort_values(ascending=False).index
    weights_ordered = sector_weights[order]

    area_fig, area_ax = plt.subplots(figsize=(W, 0.5 * W))
    area_ax.stackplot(
        weights_ordered.index,
        weights_ordered.T.to_numpy(),
        labels=weights_ordered.columns,
        alpha=0.9,
    )

    # Etiquetas de cada sector al final del gráfico (sólo si caben).
    last_date = weights_ordered.index[-1]
    last_weights = weights_ordered.iloc[-1]
    midpoints = last_weights.cumsum() - (last_weights / 2)
    for sector, y_pos, weight in zip(last_weights.index, midpoints, last_weights):
        if weight > 2.0:
            area_ax.text(
                last_date, y_pos, f" {sector}", va="center", ha="left"
            )

    area_ax.set_xlim(weights_ordered.index[0], weights_ordered.index[-1])
    area_ax.set_xlabel("Fecha")
    area_ax.set_ylabel(r"Proporción de la capitalización total del S\&P 500 (\%)")
    area_ax.set_ylim(0, 100)
    area_ax.margins(x=0)

    if CONFIG["save_figures"]:
        area_fig.savefig(
            FIG_DIR / "sp500_sector_composition.pdf",
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ## Etapa 5: entropía de Shannon
#
# La entropía `H = -Σ p·ln(p)` (con `p` los pesos sectoriales en fracción)
# mide la **diversificación** de la composición: vale `ln(n)` cuando los `n`
# sectores pesan lo mismo y cae cuando la composición se concentra.

# %%
if __name__ == "__main__":
    entropy = sector_entropy(sector_weights)
    max_entropy = np.log(len(sector_weights.columns))

    entropy_fig, entropy_ax = plt.subplots(figsize=(W, 0.5 * W))
    entropy_ax.plot(entropy.index, entropy.to_numpy(), color="darkred", linewidth=1.5)
    entropy_ax.axhline(
        max_entropy,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Entropía máxima (ln {len(sector_weights.columns)} = {max_entropy:.3f})",
    )
    entropy_ax.set_xlabel("Fecha")
    entropy_ax.set_ylabel("Entropía (nats)")
    entropy_ax.legend(loc="lower left")
    entropy_ax.grid(True, alpha=0.3)
    entropy_ax.margins(x=0)

    print(f"Entropía media: {entropy.mean():.4f} nats")
    print(f"Entropía mínima: {entropy.min():.4f} ({entropy.idxmin():%Y-%m})")
    print(f"Entropía máxima: {entropy.max():.4f} ({entropy.idxmax():%Y-%m})")

    if CONFIG["save_figures"]:
        entropy_fig.savefig(
            FIG_DIR / "sp500_sector_entropy.pdf"
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %% [markdown]
# ## Etapa 6: HHI por empresa
#
# El **índice de Herfindahl-Hirschman** `HHI = Σ_i w_i²` (con `w_i` el peso
# de capitalización de la empresa `i` dentro del S&P 500) mide la
# **concentración** de la capitalización: vale `1/N` cuando las `N` empresas
# pesan lo mismo y tiende a `1` cuando la capitalización se concentra en
# pocas compañías.  Es adimensional (fracción); la convención habitual de
# los informes multiplica el índice por 10 000.
#
# .. warning::
#    Hereda los sesgos de la reconstrucción: supervivencia de las empresas
#    que hoy están en el índice, clasificación sectorial retroactiva (aquí
#    irrelevante) y acciones en circulación constantes en el tiempo.  Es una
#    medida descriptiva de la concentración, no una reconstrucción
#    point-in-time.

# %%
if __name__ == "__main__":
    hhi = company_hhi(prices, shares)

    latest_market_caps = prices.iloc[-1].mul(shares).dropna()
    top_10_weights = (
        latest_market_caps / latest_market_caps.sum() * 100
    ).nlargest(10)
    print(
        f"\nPorcentaje de las 10 empresas con mayor capitalización "
        f"({prices.index[-1]:%Y-%m-%d}):"
    )
    print(top_10_weights.map(lambda weight: f"{weight:.2f}%").to_string())

    hhi_fig, hhi_ax = plt.subplots(figsize=(W, 0.5 * W))
    hhi_ax.plot(hhi.index, hhi.to_numpy(), color="darkgreen", linewidth=1.5)
    hhi_ax.set_xlabel("Fecha")
    hhi_ax.set_ylabel(r"HHI $= \sum w_i^2$ (adimensional)")
    hhi_ax.grid(True, alpha=0.3)
    hhi_ax.margins(x=0)

    print(f"HHI medio: {hhi.mean():.6f}")
    print(f"HHI mínimo: {hhi.min():.6f} ({hhi.idxmin():%Y-%m})")
    print(f"HHI máximo: {hhi.max():.6f} ({hhi.idxmax():%Y-%m})")

    if CONFIG["save_figures"]:
        hhi_fig.savefig(
            FIG_DIR / "sp500_company_hhi.pdf"
        )
    if CONFIG["show_figure"]:
        plt.show()
    else:
        plt.close("all")

# %%
# ## Métricas del experimento
#
# ``metrics_df`` recoge las magnitudes ya calculadas en las etapas
# anteriores (no se recalcula nada): la entropía de Shannon y el HHI de
# concentración, en su media, mínimo y máximo, más la suma de los pesos del
# snapshot actual.

if __name__ == "__main__":
    metrics_df = pd.DataFrame(
        [
            {
                "model": CONFIG["ticker"],
                "entropy_mean": float(entropy.mean()),
                "entropy_min": float(entropy.min()),
                "entropy_max": float(entropy.max()),
                "hhi_mean": float(hhi.mean()),
                "hhi_min": float(hhi.min()),
                "hhi_max": float(hhi.max()),
                "snapshot_weight_sum": float(snapshot.sum()),
            }
        ]
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
        experiment_id="spy/sectors",
        payload={"config": CONFIG, "metrics_df": metrics_df},
        chapter="SPY",
    )

# %%
