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
# Para encontrar pares cointegrados usamos **Información Mutua** estimada
# con el método de Kraskov-Stögbauer-Grassberger (KSG), que detecta
# acoplamientos tanto lineales como no lineales:
#
# $$I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)}$$
#
# Flujo del notebook:
#
# 1. Universo S&P 500 completo con ``START:END``.
# 2. Matriz MI KSG sobre todo el universo.
# 3. Top 20 pares por MI.
# 4. Clustering jerárquico de Ward con tantos clusters como sectores GICS.
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
# ``filter_duplicate_issuers`` elimina una clase de acción por emisor
# (ej. ``GOOGL``/``GOOG`` o ``FOXA``/``FOX``) antes de descargar datos y
# calcular la matriz MI, evitando evaluar pares inválidos.
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
# ``cached_ksg_mi_matrix`` persiste la matriz en ``data/pair_trading/mi/ksg/``
# con un hash determinista de fechas, activos, número de observaciones y
# contenido de los retornos.  Reejecuciones idénticas recuperan el resultado
# sin volver a calcular el estimador KSG (O(N²) en el número de activos).
mi_matrix = cached_ksg_mi_matrix(returns_full, k_neighbors=3)
top_20_pairs = top_pairs(mi_matrix, n=20, issuer_map=issuer_map)

print("\nTop 20 pares por Información Mutua KSG:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f} nats")

# %%
# ## Tabla de pares principales con sector y cluster
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
final_symbols = sorted({t1 for t1, _, _ in top_20_pairs}
                       | {t2 for _, t2, _ in top_20_pairs})
print(final_symbols)