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
# # Selección de Pares vía Información Mutua
#
# La correlación de Pearson sólo captura dependencias lineales.  Para
# encontrar pares cointegrados usamos **Información Mutua** (MI), que
# detecta también acoplamientos no lineales:
#
# $$I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)}$$
#
# El flujo es paralelo al notebook de Pearson: mismo universo, mismo
# rango temporal, mismo número de clusters y misma visualización.
#
# 1. Universo S&P 500 completo con ``START:END``.
# 2. Matriz MI gaussiana sobre todo el universo (criba rápida O(N²)).
# 3. Top 20 pares por MI.
# 4. Clustering jerárquico de Ward (k = 3) sobre la distancia MI.
# 5. Matriz reordenada por hojas y gráfico con fronteras de clúster.
# 6. Validación con estimador KSG sobre el sub-manifold reducido.

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
    gaussian_mi_matrix,
    get_sp500_tickers_by_sector,
    hierarchical_clustering,
    ksg_mi_matrix,
    load_log_returns,
    mi_distance,
    top_pairs,
)

# %%
# ## Universo y retornos
tickers_sp500 = get_sp500_tickers_by_sector("all")
print(f"Tamaño del universo S&P 500: {len(tickers_sp500)} activos")

returns_full = load_log_returns(tickers_sp500, START, END, DATA_DIR)
print(f"Retornos cargados: {returns_full.shape[0]} sesiones × {returns_full.shape[1]} activos")

# %%
# ## Información mutua gaussiana sobre el universo completo
# Para series gaussianas ``MI = -1/2 · log(1 - ρ²)``.  Es O(N²) sin
# búsqueda de vecinos, así que escala al S&P 500 completo.
mi_gaussian = gaussian_mi_matrix(returns_full)
top_20_pairs = top_pairs(mi_gaussian, n=20)

print("Top 20 pares por Información Mutua gaussiana:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f} nats")

# %%
# ## Clustering jerárquico de Ward sobre la distancia MI
n_clusters = 3
distance_gaussian = mi_distance(mi_gaussian)
cluster_labels, leaf_order = hierarchical_clustering(
    distance_gaussian, n_clusters=n_clusters,
)
labels_ordered = mi_gaussian.columns[leaf_order]
mi_ordered = mi_gaussian.loc[labels_ordered, labels_ordered]

clusters = pd.Series(cluster_labels, index=mi_gaussian.columns, name="cluster")
print(f"\nClustering jerárquico de Ward (k = {n_clusters}):")
for cid in sorted(clusters.unique()):
    members = clusters.index[clusters == cid].tolist()
    print(f"  Cluster {cid}: {members}")

# %%
# ## Visualización de la matriz MI reordenada
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mi_ordered, cmap="inferno", interpolation="nearest")
ax.set_title("Información Mutua gaussiana — hojas ordenadas por clustering Ward")
ax.set_xticks(range(len(labels_ordered)), labels_ordered, rotation=45)
ax.set_yticks(range(len(labels_ordered)), labels_ordered)
fig.colorbar(im, ax=ax, label="MI (nats)")

ordered_clusters = [int(clusters[t]) for t in labels_ordered]
boundaries = [i for i in range(1, len(ordered_clusters))
              if ordered_clusters[i] != ordered_clusters[i - 1]]
for b in boundaries:
    ax.axhline(b - 0.5, color="white", linewidth=2)
    ax.axvline(b - 0.5, color="white", linewidth=2)

plt.tight_layout()
plt.show()

# %%
# ## Validación KSG sobre el sub-manifold reducido
# Aplicamos el estimador KSG (k-NN) sobre los tickers supervivientes del
# top-20 para validar los acoplamientos sin la hipótesis gaussiana.
surviving_nodes = sorted({t1 for t1, _, _ in top_20_pairs}
                         | {t2 for _, t2, _ in top_20_pairs})

# Excluir tickers duplicados (ej. GOOGL/GOOG) que yfinance no resuelve igual
KSG_EXCLUDE = ("GOOGL", "FOXA", "NWSA")
ksg_tickers = [t for t in surviving_nodes if t not in KSG_EXCLUDE]
print(f"\nSub-manifold KSG: {len(ksg_tickers)} activos")
print(f"Tickers: {ksg_tickers}")

returns_ksg = load_log_returns(ksg_tickers, "2020-01-01", END, DATA_DIR)

mi_ksg = ksg_mi_matrix(returns_ksg, k_neighbors=3)

distance_ksg = mi_distance(mi_ksg)
cluster_labels_ksg, leaf_order_ksg = hierarchical_clustering(
    distance_ksg, n_clusters=n_clusters,
)
labels_ksg_ordered = mi_ksg.columns[leaf_order_ksg]
mi_ksg_ordered = mi_ksg.loc[labels_ksg_ordered, labels_ksg_ordered]

clusters_ksg = pd.Series(cluster_labels_ksg, index=mi_ksg.columns, name="cluster")
print(f"\nClustering KSG (k = {n_clusters}):")
for cid in sorted(clusters_ksg.unique()):
    members = clusters_ksg.index[clusters_ksg == cid].tolist()
    print(f"  Cluster {cid}: {members}")

top_ksg_pairs = top_pairs(mi_ksg, n=20)
print("\nTop 20 pares por MI KSG continua:")
for rank, (t1, t2, val) in enumerate(top_ksg_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2}: {val:.4f}")

# %%
# ## Visualización de la matriz KSG reordenada
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

im0 = axes[0].imshow(mi_ksg_ordered, cmap="inferno", interpolation="nearest")
axes[0].set_title("KSG — hojas ordenadas por clustering")
axes[0].set_xticks(range(len(labels_ksg_ordered)), labels_ksg_ordered, rotation=45)
axes[0].set_yticks(range(len(labels_ksg_ordered)), labels_ksg_ordered)
fig.colorbar(im0, ax=axes[0], label="MI (nats)")

im1 = axes[1].imshow(mi_ksg_ordered, cmap="inferno", interpolation="nearest")
ordered_clusters_ksg = [int(clusters_ksg[t]) for t in labels_ksg_ordered]
boundaries_ksg = [i for i in range(1, len(ordered_clusters_ksg))
                  if ordered_clusters_ksg[i] != ordered_clusters_ksg[i - 1]]
for b in boundaries_ksg:
    axes[1].axhline(b - 0.5, color="white", linewidth=2)
    axes[1].axvline(b - 0.5, color="white", linewidth=2)
axes[1].set_title(f"KSG — fronteras de clustering Ward (k = {n_clusters})")
axes[1].set_xticks(range(len(labels_ksg_ordered)), labels_ksg_ordered, rotation=45)
axes[1].set_yticks(range(len(labels_ksg_ordered)), labels_ksg_ordered)
fig.colorbar(im1, ax=axes[1], label="MI (nats)")

plt.tight_layout()
plt.show()

# %%
# ## Símbolos finales del top 20 KSG
final_symbols = sorted({t1 for t1, _, _ in top_ksg_pairs}
                       | {t2 for _, t2, _ in top_ksg_pairs})
print(final_symbols)