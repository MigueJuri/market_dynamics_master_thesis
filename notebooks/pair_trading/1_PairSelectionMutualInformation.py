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
# 1. Universo S&P 500 completo con ``START:END``.
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

import matplotlib.pyplot as plt
import pandas as pd

from thesis.config import DATA_DIR, END, START
from thesis.pair_trading import (
    cached_ksg_mi_matrix,
    evaluate_cluster_sector_alignment,
    filter_duplicate_issuers,
    get_sp500_constituents,
    hierarchical_clustering,
    load_log_returns,
    mi_distance,
    sector_average_matrix,
    top_pairs,
)
from thesis.plotting import (
    plot_cluster_sector_composition,
    plot_cluster_sector_heatmap,
    plot_cluster_sector_summary,
    plot_sector_reordered_matrix,
)

# %%
# ## Universo, retornos y sector GICS de cada activo
#
# ``filter_duplicate_issuers`` normaliza el nombre del valor (elimina
# sufijos como ``Class A``, ``Inc.``, ``Corp.``, etc.) y elimina una
# clase de acción por emisor (ej. ``GOOGL``/``GOOG`` o
# ``FOXA``/``FOX``) **antes** de descargar datos y calcular la matriz
# MI.  Esto evita evaluar pares inválidos y reduce el coste del KSG.
constituents, removed = filter_duplicate_issuers(get_sp500_constituents())
tickers_sp500 = constituents["Symbol"].tolist()
print(f"Tamaño del universo S&P 500: {len(tickers_sp500)} activos únicos")
if not removed.empty:
    print("Símbolos descartados (mismo emisor que otro ya presente):")
    print(removed.to_string(index=False))

sector_map = constituents.set_index("Symbol")["GICS Sector"]
issuer_map = constituents.set_index("Symbol")["Issuer"]

returns_full = load_log_returns(tickers_sp500, START, END, DATA_DIR)
print(f"Retornos cargados: {returns_full.shape[0]} sesiones × {returns_full.shape[1]} activos")

# Alineamos sector con los tickers que finalmente tienen datos
available_sectors = sector_map.reindex(returns_full.columns).dropna()
n_clusters = available_sectors.nunique()
print(f"Sectores GICS presentes: {n_clusters}")

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
mi_matrix = cached_ksg_mi_matrix(returns_full, k_neighbors=3)
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
# ## Matriz MI ordenada por sector (sin etiquetas de ticker)
#
# La matriz se reordena agrupando activos por sector y, dentro de cada
# sector, por cluster.  Se eliminan las etiquetas individuales de
# ticker (ilegibles con ≈500 activos) y se dibujan líneas gruesas
# negras en las fronteras entre sectores.  Permite ver si la MI es
# alta dentro de cada sector y baja entre sectores.
plot_sector_reordered_matrix(
    mi_matrix,
    sector_labels=available_sectors,
    cluster_labels=clusters,
    cmap="inferno",
    title="MI KSG ordenada por sector y cluster",
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
sector_mi = sector_average_matrix(mi_matrix, available_sectors, operation="mean")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sector_mi, cmap="inferno", interpolation="nearest")
ax.set_xticks(range(len(sector_mi.columns)), sector_mi.columns, rotation=45, ha="right")
ax.set_yticks(range(len(sector_mi.index)), sector_mi.index)
ax.set_title("MI KSG promedio entre sectores GICS")
fig.colorbar(im, ax=ax, label="MI media (nats)")
plt.tight_layout()
plt.show()

# %%
# ## Símbolos finales del top 20
#
# Lista única de tickers que aparecen en el top-20.  Estos son los
# candidatos a estudiar en `PairTrading.py` para el análisis del
# spread y el modelo OU.
final_symbols = sorted({t1 for t1, _, _ in top_20_pairs}
                       | {t2 for _, t2, _ in top_20_pairs})
print(final_symbols)