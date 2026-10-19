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
# La **correlación de Pearson** mide la dependencia lineal entre dos
# retornos.  Es la métrica más usada para encontrar pares candidatos,
# aunque sólo captura relaciones estrictamente lineales:
#
# $$\rho_{X,Y} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\, \sigma_Y}$$
#
# Este notebook replica exactamente el flujo del cuaderno de Información
# Mutua: mismo universo, mismo rango temporal, mismo clustering y misma
# visualización, para que ambas métricas sean comparables.
#
# 1. Universo S&P 500 completo con ``START:END``.
# 2. Matriz de correlación de Pearson.
# 3. Top 20 pares por correlación.
# 4. Clustering jerárquico de Ward (k = 3) sobre la distancia
#    ``sqrt(1 - ρ)``.
# 5. Matriz reordenada por hojas y gráfico con fronteras de clúster.
# 6. Animación de la correlación rolling sobre el sub-manifold reducido.

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
from matplotlib import animation

from thesis.config import DATA_DIR, END, START
from thesis.pair_trading import (
    correlation_distance,
    get_sp500_tickers_by_sector,
    hierarchical_clustering,
    load_log_returns,
    top_pairs,
)

# %%
# ## Universo y retornos
tickers_sp500 = get_sp500_tickers_by_sector("all")
print(f"Tamaño del universo S&P 500: {len(tickers_sp500)} activos")

returns_full = load_log_returns(tickers_sp500, START, END, DATA_DIR)
print(f"Retornos cargados: {returns_full.shape[0]} sesiones × {returns_full.shape[1]} activos")

# %%
# ## Correlación de Pearson sobre el universo completo
corr_matrix = returns_full.corr(method="pearson")
top_20_pairs = top_pairs(corr_matrix, n=20)

print("Top 20 pares por correlación de Pearson:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f}")

# %%
# ## Clustering jerárquico de Ward sobre la distancia sqrt(1 - ρ)
n_clusters = 3
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
    print(f"  Cluster {cid}: {members}")

# %%
# ## Visualización de la matriz de correlación reordenada
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_ordered, cmap="coolwarm", interpolation="nearest",
               vmin=-1, vmax=1)
ax.set_title("Correlación de Pearson — hojas ordenadas por clustering Ward")
ax.set_xticks(range(len(labels_ordered)), labels_ordered, rotation=45)
ax.set_yticks(range(len(labels_ordered)), labels_ordered)
fig.colorbar(im, ax=ax, label="Correlación de Pearson")

ordered_clusters = [int(clusters[t]) for t in labels_ordered]
boundaries = [i for i in range(1, len(ordered_clusters))
              if ordered_clusters[i] != ordered_clusters[i - 1]]
for b in boundaries:
    ax.axhline(b - 0.5, color="white", linewidth=2)
    ax.axvline(b - 0.5, color="white", linewidth=2)

plt.tight_layout()
plt.show()

# %%
# ## Sub-manifold reducido para la animación rolling
surviving_nodes = sorted({t1 for t1, _, _ in top_20_pairs}
                         | {t2 for _, t2, _ in top_20_pairs})
print(f"\nSub-manifold: {len(surviving_nodes)} activos")
print(f"Tickers: {surviving_nodes}")

returns_rolling = load_log_returns(
    surviving_nodes, "2019-01-01", END, DATA_DIR,
)

corr_static = returns_rolling.corr(method="pearson")
distance_static = correlation_distance(corr_static, use_sqrt=True)
_, static_order = hierarchical_clustering(distance_static, n_clusters=n_clusters)
static_labels = corr_static.columns[static_order]

# %%
# ## Correlación rolling (animación)
WINDOW_SIZE = 126
STEP_SIZE = 5

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
final_symbols = sorted({t1 for t1, _, _ in top_20_pairs}
                       | {t2 for _, t2, _ in top_20_pairs})
print(final_symbols)