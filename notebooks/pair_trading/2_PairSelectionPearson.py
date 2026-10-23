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
# # Selección de Pares vía Correlación de Pearson
#
# ## Definición
#
# La **correlación de Pearson** entre dos retornos $r_i$ y $r_j$ es:
#
# $$\rho_{ij} = \frac{\mathrm{Cov}(r_i, r_j)}
#                     {\sigma_i\, \sigma_j}$$
#
# Toma valores en $[-1, 1]$:
# - $\rho \approx 1$: movimientos altamente alineados;
# - $\rho \approx 0$: ausencia de dependencia lineal;
# - $\rho \approx -1$: movimientos opuestos.
#
# ## Limitaciones
#
# Pearson **sólo captura relaciones lineales**.  Dos activos con un
# acoplamiento monótono no lineal (por ejemplo $r_j = r_i^2$) pueden
# tener $\rho$ cercano a cero y aun así estar altamente coordinados.
# Por eso este notebook se complementa con
# [`1_PairSelectionMutualInformation.py`](./1_PairSelectionMutualInformation.py),
# que usa Información Mutua KSG para detectar dependencias no lineales.
#
# ## Flujo del notebook
#
# 1. Universo S&P 500 completo con ventana explícita y filtro de historia,
#    más el diagnóstico de cobertura del panel.
# 2. Matriz de correlación de Pearson.
# 3. Top 20 pares por correlación, filtrando emisores del mismo grupo.
# 4. Test ADF del spread de cada par sobre la historia completa, con el
#    tamaño muestral de cada par.
# 5. Clustering jerárquico de Ward sobre la distancia
#    $\sqrt{1 - \rho}$.
# 6. Evaluación de la calidad sectorial del clustering.
# 7. Visualización de matrices, composición sectorial y animación
#    rolling sobre el sub-manifold reducido.

# %%
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from thesis.config import (
    DATA_DIR,
    END,
    PAIR_SELECTION_MIN_HISTORY_OBS,
    START,
)
from thesis.pair_trading import (
    adf_test,
    compute_hedge_ratio,
    correlation_distance,
    evaluate_cluster_sector_alignment,
    export_pair_selection_table,
    hierarchical_clustering,
    load_log_prices,
    prepare_pair_selection_universe,
    sector_average_matrix,
    top_pairs,
    translate_gics_sectors,
    unmapped_gics_sectors,
)
from thesis.plotting import (
    plot_cluster_sector_composition,
    plot_cluster_sector_heatmap,
    plot_cluster_sector_summary,
    plot_sector_average_heatmap,
    plot_sector_reordered_matrix,
)

# %%
# ## Universo, ventana de análisis y retornos
#
# ``prepare_pair_selection_universe`` ejecuta la fase compartida con
# ``1_PairSelectionMutualInformation.py``: elimina una clase de acción por
# emisor (ej. ``GOOGL``/``GOOG`` o ``FOXA``/``FOX``), construye el panel de
# log-retornos con ``load_returns_panel`` y alinea sectores y emisores.
#
# ## Por qué el panel filtra por historia antes de alinear
#
# ``load_returns_panel`` descarta primero los activos sin historia
# suficiente y **sólo después** alinea las fechas.  Sin ese orden, ``dropna``
# conserva únicamente las fechas en las que cotizan los ~500 activos a la
# vez, y basta un único emisor recién salgado a bolsa para que un rango de
# treinta años se convierta en dos meses.  Medido sobre la caché actual, la
# intersección sin filtro da 43 sesiones (2025-10-28 → 2025-12-31) en lugar
# de las ~8.300 del rango pedido.
#
# El umbral ``PAIR_SELECTION_MIN_HISTORY_OBS`` vive en ``thesis.config`` y
# fija el equilibrio entre historia y número de activos.  Con la ventana
# completa 1993-2026 sobre el universo actual:
#
# ==========  ========  ==========  =============
# mínimo      activos   sesiones    desde
# ==========  ========  ==========  =============
# 0           497       43         2025-10-28
# 1260 (~5a)  484       1 267      2020-12-11
# 5040 (~20a) 388       5 047      2005-12-06
# 7560 (~30a) 301       7 557      1995-12-18
# ==========  ========  ==========  =============
#
# El valor por defecto son veinte años: mantiene 388 activos (suficientes
# para el clustering sectorial) y veinte años de observaciones para la
# correlación.  Subir el umbral acorta la intersección; bajarlo obliga a
# usar un panel reciente o directamente inservible.
#
# El diagnóstico que devuelve el panel (sesiones, cobertura frente al
# calendario disponible, ritmo anual, historia por activo y descartes) se
# imprime para que cualquier recorte sea visible antes de interpretar los
# resultados.
universe = prepare_pair_selection_universe(
    start=START,
    end=END,
    cache_dir=DATA_DIR,
    min_history_obs=PAIR_SELECTION_MIN_HISTORY_OBS,
)

returns_full = universe.returns
issuer_map = universe.issuers
n_clusters = universe.n_sectors

# Los sectores GICS llegan en inglés desde Wikipedia y se traducen aquí para
# que toda la salida de la tesis (métricas, figuras y tablas) esté en
# castellano.  Traducir no altera los resultados: las métricas de alineación
# sólo dependen de la partición.
sectors_en = universe.sectors
available_sectors = translate_gics_sectors(sectors_en)

sin_traducir = unmapped_gics_sectors(sectors_en)
if sin_traducir:
    print(f"⚠ Sectores GICS sin traducción CASTELLANO: {sin_traducir}")
print("Sectores GICS (traducidos):")
for sector in sorted(available_sectors.unique()):
    n_activos = int((available_sectors == sector).sum())
    print(f"  {sector}: {n_activos} activos")

# %%
# ## Correlación de Pearson sobre el universo completo
#
# ``pandas.DataFrame.corr`` calcula todas las correlaciones de forma
# vectorizada.  La diagonal queda implícita en 1; los ``NaN`` se
# descartan automáticamente par a par.
corr_matrix = returns_full.corr(method="pearson")
top_20_pairs = top_pairs(corr_matrix, n=50, issuer_map=issuer_map)

print("\nTop 20 pares por correlación de Pearson:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f}")

# %%
# ## Tabla de pares principales con sector, cluster y test de ADF
#
# Cada par lleva el sector GICS de ambos miembros.  Permite distinguir
# combinaciones intra-sectoriales (Utilities, Energy, ...) de
# combinaciones inter-sectoriales.
#
# Para cada par se toman los log-precios de las observaciones comunes a
# ambos activos mediante ``load_log_prices`` y se estima el hedge ratio por
# OLS de
# $\log A_t = \alpha + \beta \log B_t + \varepsilon_t$.  El spread
# resultante $\varepsilon_t$ es la serie sobre la que se aplica el test
# Augmented Dickey-Fuller:
#
# - ``estadistico_adf``: valores muy negativos rechazan la raíz unitaria;
# - ``p_valor_adf``: por debajo de 0.05 el spread se considera
#   estacionario;
# - ``estacionario``: bandera booleana con ese mismo criterio.
#
# El ADF se estima sobre $\log P$ y no sobre retornos: acumular los
# log-retornos con ``cumsum`` perdería el primer nivel y todas las fechas en
# las que un activo no cotizó, dando series distintas de las que produce
# ``PairTrading.py`` para un mismo par.
#
# ## Dos ventanas distintas y por qué
#
# El panel de correlación y el ADF no usan exactamente la misma muestra.  La
# correlación se estima sobre el panel alineado (intersección de fechas de
# los activos con historia suficiente), mientras que el ADF usa
# ``load_log_prices`` sobre las fechas comunes de **ese par**, que recupera
# historia adicional de los activos que no cotizaban en el periodo en que
# empezó el último superviviente del universo.
#
# La diferencia no es un descuido: la selección de candidatos necesita un
# panel común para poder clusterizar, mientras que la cointegración se
# estima mejor sobre toda la historia del par.  Ambas consultas parten de
# ``START`` y terminan en ``END``, y la del ADF **coincide con el rango de
# ``PairTrading.py`**, así que el estadístico de un mismo par es
# directamente comparable entre ambos cuadernos.  La columna
# ``n_sesiones_par`` deja explícito cuánto soporte tiene cada resultado.
#
# El objetivo del estadístico es el spread, no la correlación: dos
# activos muy correlacionados pueden tener un spread no estacionario.
SIGNIFICANCIO_ADF = 0.05

#: Un ADF sobre menos de un año de sesiones no es interpretable.
MIN_SESIONES_ADF = 252

top_pairs_df = pd.DataFrame(
    top_20_pairs,
    columns=["ticker_a", "ticker_b", "correlacion_pearson"],
)
top_pairs_df["sector_a"] = top_pairs_df["ticker_a"].map(available_sectors)
top_pairs_df["sector_b"] = top_pairs_df["ticker_b"].map(available_sectors)


def adf_para_par(ticker_a, ticker_b):
    """Calcula el estadístico, el p-value y el tamaño muestral del par.

    Devuelve ``(np.nan, np.nan, n)`` si faltan datos, no se puede calcular
    el spread o la muestra es demasiado corta, para no interrumpir la tabla
    completa.  ``n`` es el número de sesiones comunes del par, que acota
    cuánta evidencia respalda cada estadístico.
    """
    try:
        log_precios = load_log_prices([ticker_a, ticker_b], START, END, DATA_DIR)
    except ValueError:
        return np.nan, np.nan, 0

    n = len(log_precios)
    if n < MIN_SESIONES_ADF:
        return np.nan, np.nan, n

    try:
        hedge = compute_hedge_ratio(log_precios[ticker_a], log_precios[ticker_b])
        spread = log_precios[ticker_a] - (
            hedge.alpha + hedge.beta * log_precios[ticker_b]
        )
        adf = adf_test(spread, significance=SIGNIFICANCIO_ADF)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan, n
    return adf.statistic, adf.pvalue, n


adf_stats = [
    adf_para_par(row.ticker_a, row.ticker_b)
    for row in top_pairs_df.itertuples(index=False)
]
top_pairs_df["n_sesiones_par"] = [n for _, _, n in adf_stats]
top_pairs_df["estadistico_adf"] = [s for s, _, _ in adf_stats]
top_pairs_df["p_valor_adf"] = [p for _, p, _ in adf_stats]
top_pairs_df["estacionario"] = top_pairs_df["p_valor_adf"] < SIGNIFICANCIO_ADF

print("\nTop 20 pares con sector y diagnóstico ADF del spread:")
print(top_pairs_df.to_string(index=False))
print(
    f"\nSpread estacionario al {SIGNIFICANCIO_ADF:.0%} en "
    f"{int(top_pairs_df['estacionario'].sum())} de {len(top_pairs_df)} pares."
)
print(
    f"Muestra del ADF: mediana {top_pairs_df['n_sesiones_par'].median():.0f} "
    f"sesiones, rango {top_pairs_df['n_sesiones_par'].min()}-"
    f"{top_pairs_df['n_sesiones_par'].max()}."
)

# %%
# ## Exportación de la tabla
#
# ``export_pair_selection_table`` escribe tres versiones de la tabla en
# ``outputs/tables``: el CSV con los valores sin formatear, el LaTeX listo
# para insertar en la tesis y un JSON con las condiciones de cálculo.
#
# El ``.tex`` incluye como nota las condiciones que producen la tabla:
# método, ventanas solicitada y efectiva, sesiones, ritmo anual, tamaño
# del universo, filtro de historia y cobertura.  Sin ellas la tabla sería
# irreproducible: los mismos tickers sobre otra ventana dan ADF distintos.
#
# Los valores ausentes (``NaN``) se imprimen como ``--`` y los booleanos
# como ``Sí``/``No``, con el ``Sí`` en negrita para localizar de un vistazo
# los spreads estacionarios.
#
# ## Columnas de la tabla
#
# ``COLUMNAS_TABLA`` decide qué se exporta al CSV y al LaTeX, y en qué
# orden.  Edita esta lista para recortar la tabla de la tesis sin tocar
# nada más; el DataFrame completo sigue disponible arriba.  Para una tabla
# más estrecha, por ejemplo sin las columnas del ADF:
#
# ```python
# COLUMNAS_TABLA = [
#     "rank",
#     "ticker_a", "ticker_b", "sector_a", "sector_b",
#     "correlacion_pearson", "p_valor_adf",
# ]
# ```
COLUMNAS_TABLA = [
    "rank",
    "ticker_a",
    "ticker_b",
    "sector_a",
    "sector_b",
    "correlacion_pearson",
    "n_sesiones_par",
    "estadistico_adf",
    "p_valor_adf",
    "estacionario",
]

export_paths = export_pair_selection_table(
    top_pairs_df,
    universe.report,
    stem="pearson_top20",
    columns=COLUMNAS_TABLA,
    method="correlación de Pearson sobre log-retornos diarios",
    caption=(
        "Principales pares por correlación de Pearson y diagnóstico ADF "
        "de su spread"
    ),
    label="tab:pearson_top20",
    extra_metadata={
        "significancia_adf": SIGNIFICANCIO_ADF,
        "sesiones_minimas_adf": MIN_SESIONES_ADF,
        "n_sesiones_adf_mediana": float(
            top_pairs_df["n_sesiones_par"].median()
        ),
        "pares_estacionarios": int(top_pairs_df["estacionario"].sum()),
    },
    notes=[
        (
            f"Test ADF sobre el spread con significancia "
            f"{SIGNIFICANCIO_ADF:.0%} y al menos {MIN_SESIONES_ADF} sesiones "
            f"por par; mediana de {top_pairs_df['n_sesiones_par'].median():.0f} "
            f"sesiones (rango {top_pairs_df['n_sesiones_par'].min()}-"
            f"{top_pairs_df['n_sesiones_par'].max()})."
        ),
        (
            f"Spread estacionario en "
            f"{int(top_pairs_df['estacionario'].sum())} de "
            f"{len(top_pairs_df)} pares."
        ),
    ],
)
print(f"LaTeX: {export_paths['tex']}")

# %%
# ## Clustering jerárquico de Ward con k = número de sectores
#
# Para clustering se usa la distancia ``sqrt(1 - rho)`` porque cumple
# la desigualdad triangular y estabiliza el método de Ward (que asume
# distancias euclidianas).
#
# El número de clusters se fija igual al número de sectores GICS para
# facilitar la evaluación sectorial 1-a-1.
distance_pearson = correlation_distance(corr_matrix, use_sqrt=True)
cluster_labels, leaf_order = hierarchical_clustering(
    distance_pearson, n_clusters=n_clusters,
)
labels_ordered = corr_matrix.columns[leaf_order]
corr_ordered = corr_matrix.loc[labels_ordered, labels_ordered]

clusters = pd.Series(cluster_labels, index=corr_matrix.columns, name="cluster")
print(f"\nClustering jerárquico de Ward (k = {n_clusters}):")
for cid in sorted(clusters.unique()):
    members = clusters.index[clusters == cid].tolist()
    print(f"  Cluster {cid}: {members[:10]}{'...' if len(members) > 10 else ''}")

# %%
# ## Evaluación de la calidad sectorial del clustering
#
# Métricas globales y por cluster (ver ``evaluate_cluster_sector_alignment``
# para las fórmulas de purity, entropía normalizada, HHI, ARI, AMI,
# homogeneity, completeness y V-measure).  No miden calidad
# financiera: miden cuánto se parece el clustering a la partición
# oficial del S&P 500.
report = evaluate_cluster_sector_alignment(clusters, available_sectors)

print("\nMétricas globales de alineación sectorial:")
print(f"  Purity ponderada        : {report.weighted_purity:.4f}")
print(f"  Entropía normalizada    : {report.normalized_entropy:.4f}")
print(f"  HHI medio               : {report.mean_hhi:.4f}")
print(f"  Adjusted Rand Index     : {report.ari:.4f}")
print(f"  Adjusted Mutual Info    : {report.ami:.4f}")
print(f"  Homogeneity             : {report.homogeneity:.4f}")
print(f"  Completeness            : {report.completeness:.4f}")
print(f"  V-measure               : {report.v_measure:.4f}")

print("\nMétricas por cluster:")
print(report.per_cluster.to_string())

# %%
# ## Visualización sectorial
#
# Tres vistas complementarias:
# - **Heatmap**: cuántos activos de cada sector caen en cada cluster.
# - **Composición apilada**: porcentaje del cluster ocupado por cada
#   sector.
# - **Resumen**: purity, entropía normalizada y HHI por cluster.
plot_cluster_sector_heatmap(
    report.contingency,
    title=f"Sectores vs clusters (Pearson, k = {n_clusters})",
)

plot_cluster_sector_composition(
    report.contingency,
    title=f"Composición sectorial por cluster (Pearson, k = {n_clusters})",
)

plot_cluster_sector_summary(report.per_cluster)

# %%
# ## Matriz de correlación ordenada por cluster (sin etiquetas de ticker)
#
# La matriz se reordena **globalmente por cluster**, no por sector: los
# bloques del heatmap son así exactamente los clusters que evalúa el
# análisis sectorial.  Los separadores dibujados son sólo los cambios de
# cluster (línea discontinua), y no se marcan los límites sectoriales, que
# en una matriz tan densa sólo añaden ruido.
#
# ## Escala de color comparable con la de información mutua
#
# La escala se fija con los valores **medidos** en la matriz (``vmin`` y
# ``vmax`` reales, ignorando la diagonal ``NaN``) y la paleta es la misma
# que usa el cuaderno de información mutua.  No se transforma ningún valor:
# la correlación se dibuja tal cual, en su rango real.
#
# Eso permite comparar **visualmente** ambas matrices con el mismo criterio
# de intensidad, pero un mismo color no significa el mismo valor numérico:
# Pearson e información mutua tienen unidades distintas, y cada barra de
# color sigue labelling las suyas.  Lo comparable es el patrón de bloques,
# no la magnitud.
CMAP_COMPARABLE = "viridis"

plot_sector_reordered_matrix(
    corr_matrix,
    sector_labels=available_sectors,
    cluster_labels=clusters,
    cmap=CMAP_COMPARABLE,
    title="Correlación Pearson ordenada por cluster",
    cbar_label="Correlación de Pearson",
)

# %%
# ## Matriz promedio sector × sector
#
# ``sector_average_matrix`` agrega la matriz de correlación de
# activos a una matriz sector × sector usando la media.  La diagonal
# mide el acoplamiento intra-sectorial promedio.
#
# Usa la misma paleta y la misma escala medida que el heatmap de la
# matriz de activos, de modo que las dos vistas son directamente
# comparables entre sí y con las del cuaderno de información mutua.
sector_corr = sector_average_matrix(corr_matrix, available_sectors, operation="mean")

plot_sector_average_heatmap(
    sector_corr,
    cmap=CMAP_COMPARABLE,
    title="Correlación Pearson promedio entre sectores GICS",
    cbar_label="Correlación media",
)

# %%
# ## Correlación rolling (animación) sobre el sub-manifold reducido
#
# Para visualizar la estabilidad temporal de la correlación se calcula
# una ventana móvil de ``WINDOW_SIZE`` sesiones y se avanza con
# ``STEP_SIZE`` entre fotogramas:
#
# - WINDOW_SIZE = 126 sesiones ≈ 6 meses bursátiles.
# - STEP_SIZE   = 5 sesiones   ≈ 1 semana.
#
# El orden de los activos se fija con el clustering de la ventana
# completa para que los bloques visuales no salten entre frames.
surviving_nodes = sorted({t1 for t1, _, _ in top_20_pairs}
                         | {t2 for _, t2, _ in top_20_pairs})
print(f"\nSub-manifold para animación: {len(surviving_nodes)} activos")
print(f"Tickers: {surviving_nodes}")

# El sub-manifold se recorta del panel ya calculado: recargar los mismos
# tickers volvería a alinear (y potencialmente a descargar) por duplicado.
returns_rolling = returns_full[surviving_nodes]
print(
    f"Panel rolling: {len(returns_rolling)} sesiones "
    f"({returns_rolling.index[0]:%Y-%m-%d} → "
    f"{returns_rolling.index[-1]:%Y-%m-%d}), "
    f"{returns_rolling.shape[1]} activos"
)

corr_static = returns_rolling.corr(method="pearson")
distance_static = correlation_distance(corr_static, use_sqrt=True)
_, static_order = hierarchical_clustering(distance_static, n_clusters=n_clusters)
static_labels = corr_static.columns[static_order]

WINDOW_SIZE = 126
STEP_SIZE = 5

n_frames = max(0, (len(returns_rolling) - WINDOW_SIZE) // STEP_SIZE)
if n_frames == 0:
    raise ValueError(
        f"El panel tiene {len(returns_rolling)} sesiones, "
        f"insuficientes para una ventana de {WINDOW_SIZE}. "
        "Amplía la ventana de análisis o reduce WINDOW_SIZE."
    )
print(f"Fotogramas de la animación: {n_frames}")

matrices, dates = [], []
for i in range(0, len(returns_rolling) - WINDOW_SIZE, STEP_SIZE):
    window = returns_rolling.iloc[i : i + WINDOW_SIZE]
    rolling = window.corr(method="pearson").reindex(
        index=static_labels, columns=static_labels,
    )
    matrices.append(rolling.values)
    dates.append(window.index[-1].strftime("%Y-%m-%d"))

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(matrices[0], cmap="coolwarm", interpolation="nearest",
                vmin=-1, vmax=1)
ax.set_xticks(range(len(static_labels)), static_labels, rotation=45)
ax.set_yticks(range(len(static_labels)), static_labels)
fig.colorbar(cax, label="Correlación de Pearson")
title = ax.set_title(
    f"Correlación rolling ({WINDOW_SIZE} días) — {dates[0]}"
)


def update(frame_idx):
    cax.set_data(matrices[frame_idx])
    title.set_text(
        f"Correlación rolling ({WINDOW_SIZE} días) — {dates[frame_idx]}"
    )
    return [cax, title]


ani = animation.FuncAnimation(
    fig, update, frames=len(matrices), interval=30, blit=True,
)

OUTPUT_FILENAME = "rolling_correlation.gif"
ani.save(OUTPUT_FILENAME, writer="pillow")
print(f"GIF guardado en {OUTPUT_FILENAME}")

 # %%
# ## Símbolos finales del top 20 Pearson
#
# Lista única de tickers que aparecen en el top-20.  Estos son los
# candidatos a estudiar en `PairTrading.py` para el análisis del
# spread y el modelo OU.
final_symbols = sorted({t1 for t1, _, _ in top_20_pairs}
                       | {t2 for _, t2, _ in top_20_pairs})
print(final_symbols)
# %%
