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
# 1. Universo S&P 500 completo con ``START:END``.
# 2. Matriz de correlación de Pearson.
# 3. Top 20 pares por correlación, filtrando emisores del mismo grupo.
# 4. Clustering jerárquico de Ward sobre la distancia
#    $\sqrt{1 - \rho}$.
# 5. Evaluación de la calidad sectorial del clustering.
# 6. Visualización de matrices, composición sectorial y animación
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
#
# ``filter_duplicate_issuers`` elimina una clase de acción por emisor
# (ej. ``GOOGL``/``GOOG`` o ``FOXA``/``FOX``) antes de descargar
# datos y calcular la matriz de correlación, evitando evaluar pares
# inválidos.
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
#
# ``pandas.DataFrame.corr`` calcula todas las correlaciones de forma
# vectorizada.  La diagonal queda implícita en 1; los ``NaN`` se
# descartan automáticamente par a par.
corr_matrix = returns_full.corr(method="pearson")
top_20_pairs = top_pairs(corr_matrix, n=20, issuer_map=issuer_map)

print("\nTop 20 pares por correlación de Pearson:")
for rank, (t1, t2, val) in enumerate(top_20_pairs, 1):
    print(f"{rank:>3}. {t1} - {t2} : {val:.4f}")

# %%
# ## Tabla de pares principales con sector y cluster
#
# Cada par lleva el sector GICS de ambos miembros.  Permite distinguir
# combinaciones intra-sectoriales (Utilities, Energy, ...) de
# combinaciones inter-sectoriales.
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
# ## Matriz de correlación ordenada por sector (sin etiquetas de ticker)
#
# La matriz se reordena agrupando activos por sector y, dentro de
# cada sector, por cluster.  Se eliminan las etiquetas individuales
# de ticker (ilegibles con ≈500 activos) y se dibujan líneas gruesas
# negras en las fronteras entre sectores.  Permite ver si la
# correlación es alta dentro de cada sector y baja entre sectores.
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
#
# ``sector_average_matrix`` agrega la matriz de correlación de
# activos a una matriz sector × sector usando la media.  La diagonal
# mide el acoplamiento intra-sectorial promedio.
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
#
# Lista única de tickers que aparecen en el top-20.  Estos son los
# candidatos a estudiar en `PairTrading.py` para el análisis del
# spread y el modelo OU.
final_symbols = sorted({t1 for t1, _, _ in top_20_pairs}
                       | {t2 for _, t2, _ in top_20_pairs})
print(final_symbols)