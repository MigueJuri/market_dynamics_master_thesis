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
# Este notebook replica el flujo del cuaderno de Información Mutua: mismo
# universo, mismo rango temporal, mismo número de clusters (uno por sector
# GICS), y la misma evaluación sectorial, para que ambas métricas sean
# comparables.
#
# 1. Universo S&P 500 completo con ``START:END``.
# 2. Matriz de correlación de Pearson.
# 3. Top 20 pares por correlación.
# 4. Clustering jerárquico de Ward con tantos clusters como sectores.
# 5. Evaluación de la calidad sectorial del clustering.
# 6. Visualización de matrices, composición sectorial y animación rolling.

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
    evaluate_cluster_sector_alignment,
    filter_duplicate_issuers,
    get_sp500_constituents,
    hierarchical_clustering,
    load_log_returns,
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
# calcular la matriz de correlación, evitando evaluar pares inválidos.
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
# ## Correlación de Pearson sobre el universo completo
corr_matrix = returns_full.corr(method="pearson")
top_20_pairs = top_pairs(corr_matrix, n=20, issuer_map=issuer_map)

print("\nTop 20 pares por correlación de Pearson:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f}")

# %%
# ## Tabla de pares principales con sector y cluster
top_pairs_df = pd.DataFrame(
    top_20_pairs,
    columns=["ticker_a", "ticker_b", "pearson"],
)
top_pairs_df["sector_a"] = top_pairs_df["ticker_a"].map(available_sectors)
top_pairs_df["sector_b"] = top_pairs_df["ticker_b"].map(available_sectors)
print("\nTop 20 pares con sector:")
print(top_pairs_df.to_string(index=False))

# %%
# ## Clustering jerárquico de Ward con k = número de sectores
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
    title=f"Sectores vs clusters (Pearson, k = {n_clusters})",
)

plot_cluster_sector_composition(
    report.contingency,
    title=f"Composición sectorial por cluster (Pearson, k = {n_clusters})",
)

plot_cluster_sector_summary(report.per_cluster)

# %%
# ## Matriz de correlación ordenada por sector (sin etiquetas de ticker)
plot_sector_reordered_matrix(
    corr_matrix,
    sector_labels=available_sectors,
    cluster_labels=clusters,
    cmap="coolwarm",
    title="Correlación Pearson ordenada por sector y cluster",
    cbar_label="Correlación de Pearson",
)

# %%
# ## Matriz promedio sector × sector
sector_corr = sector_average_matrix(corr_matrix, available_sectors, operation="mean")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sector_corr, cmap="coolwarm", interpolation="nearest",
               vmin=-1, vmax=1)
ax.set_xticks(range(len(sector_corr.columns)), sector_corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(sector_corr.index)), sector_corr.index)
ax.set_title("Correlación Pearson promedio entre sectores GICS")
fig.colorbar(im, ax=ax, label="Correlación media")
plt.tight_layout()
plt.show()

# %%
# ## Correlación rolling (animación) sobre el sub-manifold reducido
surviving_nodes = sorted({t1 for t1, _, _ in top_20_pairs}
                         | {t2 for _, t2, _ in top_20_pairs})
print(f"\nSub-manifold para animación: {len(surviving_nodes)} activos")
print(f"Tickers: {surviving_nodes}")

returns_rolling = load_log_returns(
    surviving_nodes, "2019-01-01", END, DATA_DIR,
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