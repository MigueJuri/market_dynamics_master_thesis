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
# # Selección de Pares vía Información Mutua (KSG)
#
# ## ¿Por qué MI en lugar de correlación?
#
# La **correlación de Pearson** sólo captura dependencia *lineal*.  Dos
# series pueden tener correlación más fuerte (un valor bajo) y aun así
# estar acopladas de forma no lineal (por ejemplo, $Y = f(X)$ monótona).
# La **Información Mutua** (MI) detecta dependencias estadísticas
# generales:
#
# $$I(X;Y) = \sum_{y \in Y} \sum_{x \in X}
#   p(x,y)\, \log \frac{p(x,y)}{p(x)\,p(y)}$$
#
# $I(X;Y) = 0$ si y sólo si $X$ e $Y$ son independientes.
#
# ## El estimador KSG (k-NN)
#
# Para datos continuos no podemos estimar $p(x,y)$ mediante
# histogramas sin sesgo de discretización.  Usamos el estimador de
# **Kraskov-Stögbauer-Grassberger (KSG)**, basado en k-vecinos
# cercanos.  Las ideas clave son:
#
# - Para cada punto, se busca la distancia al k-ésimo vecino en el
#   espacio conjunto y en cada marginal.
# - La diferencia entre conteos de vecinos en el espacio conjunto y
#   en los marginales aproxima el logaritmo de la densidad conjunta
#   dividido por el producto de marginales.
# - Se promedia sobre todos los puntos y se simetriza estimando
#   $MI(X,Y)$ y $MI(Y,X)$ por separado.
#
# El parámetro ``k_neighbors=3`` controla el balance sesgo-varianza:
# valores más bajos reducen el sesgo pero aumentan la varianza.
#
# ## Flujo del notebook
#
# 1. Universo S&P 500 con ventana explícita y filtro de historia.
# 2. Matriz MI KSG sobre todo el universo (calculada y cacheada).
# 3. Top 20 pares por MI, filtrando emisores del mismo grupo.
# 4. Clustering jerárquico de Ward con tantos clusters como sectores.
# 5. Evaluación de la calidad sectorial del clustering.
# 6. Visualización de matrices y composición sectorial.

# %%
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from thesis.config import (
    DATA_DIR,
    END,
    PAIR_SELECTION_MIN_HISTORY_OBS,
    SEED,
    START,
)
from thesis.pair_trading import (
    cached_ksg_mi_matrix,
    evaluate_cluster_sector_alignment,
    export_pair_selection_table,
    hierarchical_clustering,
    mi_distance,
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
# ``2_PairSelectionPearson.py``:
#
# 1. ``filter_duplicate_issuers`` normaliza el nombre del valor (elimina
#    sufijos como ``Class A``, ``Inc.``, ``Corp.``, etc.) y conserva **una
#    sola clase de acción por emisor** (ej. ``GOOGL``/``GOOG`` o
#    ``FOXA``/``FOX``), evitando evaluar pares inválidos y reduciendo el
#    coste del KSG.
# 2. ``load_returns_panel`` **filtra por historia antes de alinear las
#    fechas**.  Sin ese filtro, ``dropna`` conservaría sólo las fechas en las
#    que cotizan los ~500 activos a la vez y un único emisor recién salgado a
#    bolsa reduciría el rango completo a dos meses.
# 3. Se alinean sectores y emisores con los tickers que sobreviven y se
#    imprime el diagnóstico de cobertura.
#
# El umbral de historia vive en ``thesis.config`` como
# ``PAIR_SELECTION_MIN_HISTORY_OBS``, de modo que ambos cuadernos estiman su
# métrica sobre exactamente la misma muestra y las dos matrices son
# comparables.  Medido sobre la caché actual, con la ventana 1993-2026:
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
# Se elige el mínimo de veinte años: mantiene 388 activos (suficientes para
# el clustering sectorial) y veinte años de observaciones.  Subir el umbral
# acorta la intersección; bajarlo obliga a usar un panel reciente o
# directamente inservible.
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
    print(f"⚠ Sectores GICS sin traducción al castellano: {sin_traducir}")
print("Sectores GICS (traducidos):")
for sector in sorted(available_sectors.unique()):
    n_activos = int((available_sectors == sector).sum())
    print(f"  {sector}: {n_activos} activos")

# %%
# ## Información mutua KSG sobre el universo completo
#
# ``cached_ksg_mi_matrix`` persiste la matriz en
# ``data/pair_trading/mi/ksg/`` con un hash determinista de fechas,
# activos, número de observaciones y contenido de los retornos.
# Reejecuciones idénticas recuperan el resultado sin volver a
# calcular el estimador KSG (O(N²) en el número de activos).
#
# Cada par ``(i, j)`` se estima dos veces (``mi_xy`` y ``mi_yx``) y
# se simetriza como la media, mitigando asimetrías numéricas de la
# búsqueda k-NN.  Series con varianza cero se omiten y la diagonal
# queda vacía.
# El parámetro ``k_neighbors`` controla el balance sesgo-varianza del
# estimador.  Se fija una única constante y se reutiliza tanto en el cálculo
# de la matriz como en las condiciones de la tabla exportada, para que la
# tabla documente exactamente el estimador que la produjo.
K_NEIGHBORS = 3

mi_matrix = cached_ksg_mi_matrix(returns_full, k_neighbors=K_NEIGHBORS)
top_20_pairs = top_pairs(mi_matrix, n=20, issuer_map=issuer_map)

print("\nTop 20 pares por Información Mutua KSG:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f} nats")

# %%
# ## Tabla de pares principales con sector y cluster
#
# Cada par lleva el sector GICS de ambos miembros.  Esto permite
# identificar combinaciones inter-sectoriales o intra-sectoriales.
top_pairs_df = pd.DataFrame(
    top_20_pairs,
    columns=["ticker_a", "ticker_b", "mi"],
)
top_pairs_df["sector_a"] = top_pairs_df["ticker_a"].map(available_sectors)
top_pairs_df["sector_b"] = top_pairs_df["ticker_b"].map(available_sectors)
print("\nTop 20 pares con sector:")
print(top_pairs_df.to_string(index=False))

# %%
# ## Exportación de la tabla
#
# ``export_pair_selection_table`` escribe tres versiones de la tabla en
# ``outputs/tables``: el CSV con los valores sin formatear, el LaTeX listo
# para insertar en la tesis y un JSON con las condiciones de cálculo.
#
# El ``.tex`` incluye como nota las condiciones que producen la tabla:
# método, ventanas solicitada y efectiva, sesiones, ritmo anual, tamaño del
# universo, filtro de historia y cobertura.  Aquí es especialmente
# importante registrar el número de vecinos, porque la MI KSG es
# sesgo-varianza dependiente de ``k`` y dos ejecuciones con distinto ``k``
# no son comparables.
#
# ## Columnas de la tabla
#
# ``COLUMNAS_TABLA`` decide qué se exporta al CSV y al LaTeX, y en qué
# orden.  Edita esta lista para recortar la tabla de la tesis; el DataFrame
# completo sigue disponible arriba.
COLUMNAS_TABLA = [
    "rank",
    "ticker_a",
    "ticker_b",
    "sector_a",
    "sector_b",
    "mi",
]

export_paths = export_pair_selection_table(
    top_pairs_df,
    universe.report,
    stem="mutual_information_top20",
    columns=COLUMNAS_TABLA,
    method=(
        f"información mutua estimada con KSG (k = {K_NEIGHBORS} vecinos) "
        f"sobre log-retornos diarios"
    ),
    caption="Principales pares por información mutua KSG",
    label="tab:mi_top20",
    extra_metadata={"vecinos_ksg": K_NEIGHBORS, "semilla": SEED},
    notes=[
        (
            f"Estimador KSG con k = {K_NEIGHBORS} vecinos sobre "
            f"{universe.report.n_sessions} observaciones; cada par se estima "
            f"en ambos sentidos y se simetriza."
        ),
        "Métrica expresada en nats.",
    ],
)
print(f"LaTeX: {export_paths['tex']}")

# %%
# ## Clustering jerárquico de Ward con k = número de sectores
#
# ``mi_distance`` convierte MI en distancia restando el máximo:
# $d_{ij} = \max(MI) - MI_{ij}$.  Activos con MI alta quedan
# cercanos; activos independientes quedan lejos.
#
# El método de **Ward** minimiza el incremento de varianza al fusionar
# clusters.  El número de clusters se fija igual al número de sectores
# GICS presentes para hacer la evaluación sectorial 1-a-1.
distance_matrix = mi_distance(mi_matrix)
cluster_labels, leaf_order = hierarchical_clustering(
    distance_matrix, n_clusters=n_clusters,
)
labels_ordered = mi_matrix.columns[leaf_order]
mi_ordered = mi_matrix.loc[labels_ordered, labels_ordered]

clusters = pd.Series(cluster_labels, index=mi_matrix.columns, name="cluster")
print(f"\nClustering jerárquico de Ward (k = {n_clusters}):")
for cid in sorted(clusters.unique()):
    members = clusters.index[clusters == cid].tolist()
    print(f"  Cluster {cid}: {members[:10]}{'...' if len(members) > 10 else ''}")

# %%
# ## Evaluación de la calidad sectorial del clustering
#
# Mide cuánto se parece el clustering a la partición oficial por
# sectores GICS.  Las métricas se imprimen primero en formato global
# y después desglosadas por cluster.  No son una medida absoluta de
# calidad financiera: evalúan **alineación con la taxonomía GICS**.
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
#   sector (útil para comparar clusters de distinto tamaño).
# - **Resumen**: purity, entropía normalizada y HHI por cluster.
plot_cluster_sector_heatmap(
    report.contingency,
    title=f"Sectores vs clusters (MI KSG, k = {n_clusters})",
)

plot_cluster_sector_composition(
    report.contingency,
    title=f"Composición sectorial por cluster (MI KSG, k = {n_clusters})",
)

plot_cluster_sector_summary(report.per_cluster)

# %%
# ## Matriz MI ordenada por cluster (sin etiquetas de ticker)
#
# La matriz se reordena **globalmente por cluster**, igual que en
# ``2_PairSelectionPearson.py``: los bloques del heatmap son los clusters
# que evalúa el análisis sectorial, y los únicos separadores dibujados son
# los cambios de cluster (línea discontinua).
#
# ## Escala de color comparable con la de Pearson
#
# Se usa la misma paleta (``viridis``) y la misma política de escala: los
# límites salen de los valores **medidos** en la matriz, ignorando la
# diagonal ``NaN``.  Ningún valor se transforma; la MI se dibuja en nats.
#
# Un mismo color equivale a la misma intensidad **dentro de cada métrica**,
# no al mismo valor numérico: Pearson e información mutua tienen unidades
# distintas y cada barra de color etiqueta las suyas.  Lo comparable entre
# ambos cuadernos es el patrón de los bloques, no la magnitud.
CMAP_COMPARABLE = "viridis"

plot_sector_reordered_matrix(
    mi_matrix,
    sector_labels=available_sectors,
    cluster_labels=clusters,
    cmap=CMAP_COMPARABLE,
    title="MI KSG ordenada por cluster",
    cbar_label="MI (nats)",
)

# %%
# ## Matriz promedio sector × sector
#
# ``sector_average_matrix`` agrega la matriz de activos a una matriz
# sector × sector usando la media (omitiendo la diagonal dentro del
# bloque).  Valores altos en la diagonal confirman acoplamiento
# intra-sectorial; valores altos fuera de la diagonal indican
# dependencias inter-sectoriales.
#
# Comparte paleta y escala medida con el heatmap de la matriz de activos
# y con las figuras del cuaderno de Pearson.
sector_mi = sector_average_matrix(mi_matrix, available_sectors, operation="mean")

plot_sector_average_heatmap(
    sector_mi,
    cmap=CMAP_COMPARABLE,
    title="MI KSG promedio entre sectores GICS",
    cbar_label="MI media (nats)",
)

# %%
# ## Símbolos finales del top 20
#
# Lista única de tickers que aparecen en el top-20.  Estos son los
# candidatos a estudiar en `PairTrading.py` para el análisis del
# spread y el modelo OU.
final_symbols = sorted({t1 for t1, _, _ in top_20_pairs}
                       | {t2 for _, t2, _ in top_20_pairs})
print(final_symbols)
# %%
