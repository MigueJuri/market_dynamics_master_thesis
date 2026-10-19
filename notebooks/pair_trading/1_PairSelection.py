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
# Para encontrar pares cointegrados no basta con mirar la correlación
# lineal.  Usamos **Información Mutua** (MI) para capturar dependencias
# tanto lineales como no lineales entre los retornos fraccionarios:
#
# $$I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)}$$
#
# El notebook sigue este flujo:
#
# 1. Universo inicial (S&P 500 completo) → filtro gaussiano rápido.
# 2. Sub-manifold reducido → estimador KSG (k-NN) más preciso.
# 3. Clustering jerárquico de Ward sobre la matriz de distancias MI.
# 4. Comparación con la correlación de Pearson.
# 5. Animación de la correlación rolling.

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

from thesis.config import DATA_DIR, END, START
from thesis.pair_trading import (
    correlation_distance,
    gaussian_mi_matrix,
    get_sp500_tickers_by_sector,
    hierarchical_clustering,
    ksg_mi_matrix,
    load_log_returns,
    mi_distance,
    top_pairs,
)

# %%
# ## Universo y retornos (S&P 500 completo)
tickers_sp500 = get_sp500_tickers_by_sector("all")
print(f"Tamaño del universo S&P 500: {len(tickers_sp500)} activos")

returns_full = load_log_returns(tickers_sp500, START, END, DATA_DIR)
print(f"Retornos cargados: {returns_full.shape[0]} sesiones × {returns_full.shape[1]} activos")

# %%
# ## Información mutua gaussiana (criba rápida)
# Para series gaussianas ``MI = -0.5 · log(1 - ρ²)``.  Es O(N²) sin búsqueda
# de vecinos, así que es ideal para reducir un universo grande.
mi_gaussian = gaussian_mi_matrix(returns_full)
top_50_pairs = top_pairs(mi_gaussian, n=50)

print("Top 50 pares por MI gaussiana (listos para validación KSG):")
for rank, (t1, t2, val) in enumerate(top_50_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f} nats")

# %%
# ## Sub-manifold reducido para validación KSG
top_20_pairs = top_pairs(mi_gaussian, n=20)
surviving_nodes = sorted({t1 for t1, _, _ in top_20_pairs}
                         | {t2 for _, t2, _ in top_20_pairs})

# Excluir tickers duplicados (ej. GOOGL/GOOG) que yfinance no resuelve igual
KSG_EXCLUDE = ("GOOGL", "FOXA", "NWSA")
ksg_tickers = [t for t in surviving_nodes if t not in KSG_EXCLUDE]

print(f"Tamaño del sub-manifold: {len(ksg_tickers)} activos")
print(f"Tickers para validación: {ksg_tickers}")

returns_ksg = load_log_returns(ksg_tickers, "2020-01-01", END, DATA_DIR)

# %%
# ## Información mutua KSG (k-NN) y clustering jerárquico
mi_ksg = ksg_mi_matrix(returns_ksg, k_neighbors=3)

n_clusters = 3
distance_ksg = mi_distance(mi_ksg)
cluster_labels, leaf_order = hierarchical_clustering(distance_ksg, n_clusters=n_clusters)
labels_ordered = mi_ksg.columns[leaf_order]
mi_ksg_ordered = mi_ksg.loc[labels_ordered, labels_ordered]

clusters = pd.Series(cluster_labels, index=mi_ksg.columns, name="cluster")
print(f"Clustering jerárquico (k = {n_clusters}):")
for cid in sorted(clusters.unique()):
    members = clusters.index[clusters == cid].tolist()
    print(f"  Cluster {cid}: {members}")

top_ksg_pairs = top_pairs(mi_ksg, n=20)
print("\nTop 20 pares por MI KSG continua:")
for rank, (t1, t2, val) in enumerate(top_ksg_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2}: {val:.4f}")

# %%
# ## Visualización de las matrices MI
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

im0 = axes[0].imshow(mi_ksg_ordered, cmap="inferno", interpolation="nearest")
axes[0].set_title("Mutual Information (KSG) — hojas ordenadas por clustering")
axes[0].set_xticks(range(len(labels_ordered)), labels_ordered, rotation=45)
axes[0].set_yticks(range(len(labels_ordered)), labels_ordered)
fig.colorbar(im0, ax=axes[0], label="MI (nats)")

ordered_clusters = [int(clusters[t]) for t in labels_ordered]
boundaries = [i for i in range(1, len(ordered_clusters))
              if ordered_clusters[i] != ordered_clusters[i - 1]]

im1 = axes[1].imshow(mi_ksg_ordered, cmap="inferno", interpolation="nearest")
for b in boundaries:
    axes[1].axhline(b - 0.5, color="white", linewidth=2)
    axes[1].axvline(b - 0.5, color="white", linewidth=2)
axes[1].set_title("Clustering Ward (k = 3)")
axes[1].set_xticks(range(len(labels_ordered)), labels_ordered, rotation=45)
axes[1].set_yticks(range(len(labels_ordered)), labels_ordered)
fig.colorbar(im1, ax=axes[1], label="MI (nats)")

plt.tight_layout()
plt.show()

# %%
# ## Comparación con correlación de Pearson
corr_matrix = returns_ksg.corr(method="pearson")
distance_corr = correlation_distance(corr_matrix, use_sqrt=True)
_, corr_leaf_order = hierarchical_clustering(distance_corr, n_clusters=n_clusters)
labels_corr = corr_matrix.columns[corr_leaf_order]
corr_ordered = corr_matrix.loc[labels_corr, labels_corr].clip(lower=0, upper=1)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_ordered, cmap="inferno", interpolation="nearest",
               vmin=0, vmax=float(np.nanmax(corr_ordered.values)))
ax.set_title("Clustering lineal (Pearson) — hojas ordenadas por Ward")
ax.set_xticks(range(len(labels_corr)), labels_corr, rotation=45)
ax.set_yticks(range(len(labels_corr)), labels_corr)
fig.colorbar(im, ax=ax, label="Correlación de Pearson")
plt.tight_layout()
plt.show()

top_corr_pairs = top_pairs(corr_matrix, n=20)
print("Top 20 pares por correlación de Pearson:")
for rank, (t1, t2, val) in enumerate(top_corr_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2}: {val:.4f}")

# %%
# ## Correlación rolling (animación)
returns_rolling = load_log_returns(
    ksg_tickers, "2019-01-01", END, DATA_DIR,
)

corr_static = returns_rolling.corr(method="pearson")
distance_static = correlation_distance(corr_static, use_sqrt=True)
_, static_order = hierarchical_clustering(distance_static, n_clusters=n_clusters)
static_labels = corr_static.columns[static_order]

WINDOW_SIZE = 126
STEP_SIZE = 5

matrices, dates = [], []
for i in range(0, len(returns_rolling) - WINDOW_SIZE, STEP_SIZE):
    window = returns_rolling.iloc[i : i + WINDOW_SIZE]
    rolling = window.corr(method="pearson").reindex(index=static_labels,
                                                     columns=static_labels)
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
# ## Símbolos finales del top 20 KSG
final_symbols = sorted({t1 for t1, _, _ in top_ksg_pairs}
                       | {t2 for _, t2, _ in top_ksg_pairs})
print(final_symbols)
