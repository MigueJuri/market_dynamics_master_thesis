"""Funciones de visualización reutilizables para análisis de características."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from thesis.config import PLOT_RC, TRADING_DAYS


plt.rcParams.update(PLOT_RC)


# =============================================================================
# Utilidades de ejes / ticks
# =============================================================================

def _year_ticks(
    windows: Sequence[int],
    trading_days: int = TRADING_DAYS,
    ticks_per_year: int = 4,
) -> Tuple[List[float], List[str]]:
    """Posiciones (centro de celda) y etiquetas en fracciones de año."""
    windows = np.asarray(windows)
    n_fractions = int(np.floor(windows[-1] / trading_days * ticks_per_year))
    fractions = np.arange(n_fractions + 1) / ticks_per_year

    positions, labels, used = [], [], set()
    for frac in fractions:
        j = 0 if frac == 0 else int(np.argmin(np.abs(windows - frac * trading_days)))
        if j in used:
            continue
        used.add(j)
        positions.append(j + 0.5)
        labels.append(f"{frac:.2f}")

    return positions, labels


# =============================================================================
# Visualización
# =============================================================================

def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    lookbacks: Sequence[int],
    lookaheads: Sequence[int],
    title: str = "Rank Correlation by Lag",
    xlabel: str = r"$\tau_p$ (años)",
    ylabel: str = r"$\tau_f$ (años)",
    trading_days: int = TRADING_DAYS,
    ticks_per_year: int = 4,
    cbar_label: str = "Correlación",
    cmap: str = "coolwarm",
) -> None:
    """Heatmap de la matriz de correlación con ticks en fracciones de año."""
    plt.figure(figsize=(10 / 2, 8 / 2), dpi=300)
    ax = sns.heatmap(corr_matrix, cmap=cmap, center=0,
                     cbar_kws={"label": cbar_label})

    xticks, xlabels = _year_ticks(lookbacks, trading_days, ticks_per_year)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45)

    yticks, ylabels = _year_ticks(lookaheads, trading_days, ticks_per_year)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, rotation=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_scatter_grid(
    feature_by_window: Dict[int, pd.Series],
    target_by_window: Dict[int, pd.Series],
    feature_label: str = "Feature",
    target_label: str = "Target",
    feature_format: Callable[[int], str] = str,
    target_format: Callable[[int], str] = str,
    figsize: Tuple[float, float] = (6.0, 5.0),
    scatter_kwargs: Optional[dict] = None,
    color_by_date: bool = True,
    cmap: str = "viridis",
    date_format: str = "%Y",
) -> None:
    """Grilla de scatter feature × target para todas las combinaciones de ventanas.

    ``feature_by_window`` y ``target_by_window`` deben compartir índices (o,
    al menos, solapar en sus fechas).  Las claves determinan los rótulos de
    cada eje.

    Si ``color_by_date`` es ``True`` (por defecto), cada punto se colorea
    según su fecha con el ``cmap`` indicado y se añade una colorbar
    compartida por toda la rejilla con marcas formateadas con
    ``date_format`` (por defecto sólo el año).  El rango de colores se
    computa una vez sobre la unión de fechas válidas de todos los
    paneles, de modo que paneles distintos comparten escala.
    ``scatter_kwargs`` puede ajustar el resto de propiedades del scatter
    (``s``, ``alpha``, ...); ``c``, ``cmap``, ``vmin`` y ``vmax`` los
    gestiona siempre la función.
    """
    base_kwargs = {"s": 2, "alpha": 0.6}
    extra_kwargs = {
        k: v for k, v in (scatter_kwargs or {}).items()
        if k not in {"c", "cmap", "vmin", "vmax"}
    }
    kwargs = {**base_kwargs, **extra_kwargs}
    feature_windows = list(feature_by_window.keys())
    target_windows = list(target_by_window.keys())

    fig, axes = plt.subplots(len(target_windows), len(feature_windows),
                             figsize=figsize, sharex=False, sharey=False,
                             constrained_layout=True)

    global_min, global_max = None, None
    for fw in feature_windows:
        for tw in target_windows:
            common_idx = feature_by_window[fw].index.intersection(
                target_by_window[tw].index,
            )
            if len(common_idx) == 0:
                continue
            if global_min is None or common_idx.min() < global_min:
                global_min = common_idx.min()
            if global_max is None or common_idx.max() > global_max:
                global_max = common_idx.max()
    if global_min is None:
        return
    vmin = mdates.date2num(global_min)
    vmax = mdates.date2num(global_max)

    first_scatter = None
    last_row = len(target_windows) - 1
    for i, tw in enumerate(target_windows):
        for j, fw in enumerate(feature_windows):
            ax = axes[i, j]
            data = pd.concat(
                [feature_by_window[fw].rename(feature_label),
                 target_by_window[tw].rename(target_label)],
                axis=1,
            ).dropna()
            if data.empty:
                continue
            if color_by_date:
                scatter = ax.scatter(
                    data.iloc[:, 0], data.iloc[:, 1],
                    c=mdates.date2num(data.index),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    **kwargs,
                )
                if first_scatter is None:
                    first_scatter = scatter
            else:
                ax.scatter(data.iloc[:, 0], data.iloc[:, 1], **kwargs)
            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.axvline(0, color="black", lw=0.8, alpha=0.6)
            if i == last_row:
                ax.set_xlabel(f"{feature_label} {feature_format(fw)}")
            if j == 0:
                ax.set_ylabel(f"{target_label} {target_format(tw)}")

    if color_by_date and first_scatter is not None:
        cbar = fig.colorbar(first_scatter, ax=axes,
                            fraction=0.025, pad=0.02)
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter(date_format))
        cbar.set_label("Fecha")

    plt.show()


def plot_feature_boxplots(
    feature: pd.Series,
    target: pd.Series,
    n_quantiles: int = 5,
    x_axis: str = "quantiles",
    feature_name: str = "feature",
    target_name: str = "target",
    feature_window: Optional[int] = None,
    target_window: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    palette: str = "viridis",
    return_correlation: bool = True,
    show_cumulative: bool = False,
) -> Optional[float]:
    """Boxplot del objetivo agrupado por la característica.

    El eje X puede construirse de dos formas equivalentes:

    * ``x_axis='quantiles'`` (por defecto): usa :func:`pandas.qcut` para
      dividir la característica en ``n_quantiles`` grupos con el mismo
      número de observaciones cada uno (cuantiles clásicos).
    * ``x_axis='bins'``: usa :func:`pandas.cut` para dividir el rango
      ``[min(feature), max(feature)]`` en ``n_quantiles`` intervalos de
      igual anchura en valor crudo.

    ``n_quantiles`` controla el número de grupos en ambos casos.

    Si ``show_cumulative`` es ``True``, añade debajo del boxplot la
    función de distribución acumulada empírica (ECDF) de la
    característica, permitiendo ver la frecuencia con la que aparecen
    los distintos valores del feature.  La ECDF usa los valores
    crudos del feature, no las etiquetas de cuantiles o bins, ya que
    la frecuencia por cuantil es aproximadamente uniforme y poco
    informativa.

    Si ``return_correlation`` es ``True`` devuelve la correlación de
    Pearson entre ``feature`` y ``target`` sobre la muestra conjunta; si
    es ``False`` devuelve ``None``.
    """
    if x_axis not in ("quantiles", "bins"):
        raise ValueError(
            f"x_axis debe ser 'quantiles' o 'bins'; se recibió '{x_axis}'."
        )

    data = pd.concat(
        [feature.rename(feature_name), target.rename(target_name)],
        axis=1,
    ).dropna()

    if x_axis == "quantiles":
        labels = [f"Q{i + 1}" for i in range(n_quantiles)]
        if n_quantiles >= 2:
            labels[0] = "Q1\n(Más bajo)"
            labels[-1] = f"Q{n_quantiles}\n(Más Alto)"
        data["quantile"] = pd.qcut(data[feature_name],
                                    q=n_quantiles, labels=labels)
        x_label_suffix = f"{feature_name.capitalize()} cuantiles"
    else:
        f_min = float(data[feature_name].min())
        f_max = float(data[feature_name].max())
        edges = np.linspace(f_min, f_max, n_quantiles + 1)
        edges = np.unique(edges)
        if len(edges) < 2:
            raise ValueError(
                "No se pueden construir bins: la característica es constante."
            )
        bin_labels = [f"{(lo + hi) / 2:.2g}"
                      for lo, hi in zip(edges[:-1], edges[1:])]
        data["quantile"] = pd.cut(data[feature_name],
                                   bins=edges, include_lowest=True,
                                   labels=bin_labels)
        x_label_suffix = f"{feature_name.capitalize()}"

    fw_str = f"({feature_window}d) " if feature_window is not None else ""
    tw_str = f"({target_window}d) " if target_window is not None else ""
    y_label = f"{target_name} {tw_str}".strip()
    x_label_full = f"{x_label_suffix} {fw_str}".strip()

    if show_cumulative:
        created_fig = ax is None
        if created_fig:
            fig, ax_box = plt.subplots(figsize=(6, 4.5))
        else:
            ax_box = ax
        divider = make_axes_locatable(ax_box)
        ax_ecdf = divider.append_axes("bottom", size="30%", pad=0.2)

        sns.boxplot(data=data, x="quantile", y=target_name, ax=ax_box,
                    palette=palette, width=0.6, fliersize=3)
        ax_box.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
        # Ocultamos tick labels del boxplot pero conservamos la etiqueta
        # del eje X, que aparecerá entre el boxplot y la ECDF.
        ax_box.tick_params(labelbottom=False)
        ax_box.set_xlabel(x_label_full)
        ax_box.set_ylabel(y_label)

        sorted_feat = data[feature_name].dropna().sort_values().to_numpy()
        ecdf_y = np.arange(1, len(sorted_feat) + 1) / len(sorted_feat)
        ax_ecdf.plot(sorted_feat, ecdf_y, color="C0", lw=1.5)
        ax_ecdf.set_ylim(0.0, 1.05)
        ax_ecdf.set_xlim(sorted_feat[0], sorted_feat[-1])
        ax_ecdf.set_ylabel("Frecuencia acum.")

        if created_fig:
            plt.tight_layout()
            plt.show()
    else:
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(6, 3))

        sns.boxplot(data=data, x="quantile", y=target_name, ax=ax,
                    palette=palette, width=0.6, fliersize=3)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
        ax.set_xlabel(x_label_full)
        ax.set_ylabel(y_label)

        if created_fig:
            plt.tight_layout()
            plt.show()

    if return_correlation:
        return data[feature_name].corr(data[target_name])
    return None


def plot_cross_sectional_curves(
    corr_matrix: pd.DataFrame,
    avg_corr: pd.Series,
    asset_groups: Optional[Dict[str, Sequence[str]]] = None,
    group_colors: Optional[Dict[str, str]] = None,
    avg_label: str = "Promedio",
    xlabel: str = r"$\tau$ (años)",
    ylabel: str = "Correlación de Pearson",
    figsize: Tuple[float, float] = (6.0, 4.0),
    tau_to_yf: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
    trading_days: int = TRADING_DAYS,
) -> None:
    """Curvas de correlación por activo con promedio cross-sectional."""
    fig, ax = plt.subplots(figsize=figsize)

    if tau_to_yf is None:
        tau = 2 * corr_matrix.index.to_numpy() / trading_days
    else:
        tau = tau_to_yf(corr_matrix.index.to_numpy())

    if asset_groups is None:
        for asset in corr_matrix.columns:
            ax.plot(tau, corr_matrix[asset], linewidth=1, alpha=0.9,
                    marker="o", markersize=5, label=asset)
    else:
        for group_name, assets in asset_groups.items():
            color = (group_colors or {}).get(group_name)
            for asset in assets:
                if asset not in corr_matrix.columns:
                    continue
                kwargs = {"linewidth": 1, "alpha": 0.9,
                          "marker": "o", "markersize": 5,
                          "label": group_name}
                if color is not None:
                    kwargs["color"] = color
                ax.plot(tau, corr_matrix[asset], **kwargs)

    ax.plot(tau, avg_corr, color="black", linewidth=2.5,
            linestyle="--", marker="o", markersize=6, label=avg_label)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Evaluación sectorial de clusters
# =============================================================================


def plot_sector_average_heatmap(
    matrix: pd.DataFrame,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 8),
    title: str = "Matriz promedio sector x sector",
    cbar_label: str = "Valor medio",
    xlabel: str = "Sector GICS",
    ylabel: str = "Sector GICS",
) -> Optional[plt.Figure]:
    """Heatmap de una matriz sector x sector con escala medida.

    Comparte la política de color de :func:`plot_sector_reordered_matrix`:
    la paleta se recibe como parámetro y los límites salen de los valores
    realmente presentes en la matriz.  Así una correlación de Pearson y una
    información mutua se dibujan con la misma paleta y el mismo criterio de
    intensidad, sin transformar los datos ni sus unidades.
    """
    valores = matrix.to_numpy(dtype=float)
    finitos = valores[np.isfinite(valores)]
    if finitos.size == 0:
        print("La matriz no contiene valores finitos.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix, cmap=cmap, interpolation="nearest",
        vmin=float(finitos.min()), vmax=float(finitos.max()),
    )
    ax.set_xticks(range(len(matrix.columns)), matrix.columns,
                  rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.show()
    return fig

def plot_cluster_sector_heatmap(
    contingency: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 7),
    cmap: str = "Blues",
    title: str = "Sectores vs clusters",
) -> None:
    """Heatmap de conteos sector × cluster.

    Se espera una tabla de contingencia que aún incluya las filas
    ``cluster_size`` y ``cluster_purity``; la función las descarta
    automáticamente.
    """
    data = contingency.drop(
        index=["cluster_size", "cluster_purity"], errors="ignore"
    ).astype(int)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=True, fmt="d", cmap=cmap, cbar=True, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sector GICS")
    plt.tight_layout()
    plt.show()


def plot_cluster_sector_composition(
    contingency: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 6),
    palette: str = "tab20",
    title: str = "Composición sectorial por cluster",
) -> None:
    """Barras apiladas al 100 % con la composición sectorial de cada cluster."""
    data = contingency.drop(
        index=["cluster_size", "cluster_purity"], errors="ignore"
    )
    shares = data.div(data.sum(axis=0), axis=1).T

    fig, ax = plt.subplots(figsize=figsize)
    shares.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        colormap=palette,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proporción del cluster")
    ax.set_ylabel("Cluster")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        title="Sector",
    )
    plt.tight_layout()
    plt.show()


def plot_cluster_sector_summary(
    per_cluster: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 4.5),
) -> None:
    """Gráficas de pureza, entropía normalizada y HHI por cluster."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    per_cluster.plot(
        y="purity",
        kind="bar",
        ax=axes[0],
        color="C0",
        legend=False,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Pureza sectorial")
    axes[0].set_ylabel("Pureza")

    per_cluster.plot(
        y="norm_entropy",
        kind="bar",
        ax=axes[1],
        color="C1",
        legend=False,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Entropía normalizada")
    axes[1].set_ylabel("Entropía normalizada")

    per_cluster.plot(
        y="hhi",
        kind="bar",
        ax=axes[2],
        color="C2",
        legend=False,
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("HHI sectorial")
    axes[2].set_ylabel("HHI")

    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_sector_reordered_matrix(
    matrix: pd.DataFrame,
    sector_labels: pd.Series | None = None,
    cluster_labels: pd.Series | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 9),
    title: str = "Matriz ordenada por cluster",
    cbar_label: str = "Valor",
separator_color: str = "white",
    sector_labels_visible: bool = False,
    cluster_labels_visible: bool = True,
    xlabel: str = "Clusters",
) -> plt.Figure | None:
    """Muestra la matriz reordenada por clúster y sin etiquetas de ticker.

    El orden es global por clúster, no «sector y dentro del sector»: así
    los bloques del heatmap son exactamente los clusters que evalúa el
    análisis sectorial.  Los separadores dibujados son únicamente los
    cambios de clúster; no se marcan los límites sectoriales, que en un
    mapa de calor denso sólo añaden ruido.

    Bajo cada bloque se imprime su número, centrado, de modo que se sabe a
    qué cluster corresponde cada zona.  La numeración es la posición del
    bloque de izquierda a derecha y no el identificador interno del
    clustering, para que la lectura sea inmediata.

    La escala de color se fija con los valores **medidos** en la matriz
    (``vmin``/``vmax`` reales, ignorando la diagonal ``NaN``), no con un
    rango teórico.  Esto permite comparar visualmente dos métricas
    distintas —correlación de Pearson e información mutua— con la misma
    paleta y el mismo criterio de intensidad, sin transformar los datos:
    cada barra sigue labeling sus unidades reales.  Un mismo color
    significa "intensidad equivalente dentro de cada métrica", no el mismo
    valor numérico, porque Pearson e MI tienen unidades distintas.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matriz cuadrada con tickers en índice y columnas.  Los valores se
        dibujan sin transformar.
    sector_labels : pd.Series, opcional
        Sectores por ticker.  Se usan como desempate al ordenar dentro de
        cada cluster y, si ``sector_labels_visible`` es ``True``, para
        rotular el eje Y.
    cluster_labels : pd.Series, opcional
        Etiqueta de cluster por ticker; si falta, la función no dibuja
        ningún separador ni numeración.
    cmap : str
        Paleta; por defecto ``viridis`` para compartir entre métricas.
    separator_color : str
        Color de las líneas discontinuas entre clusters.
    sector_labels_visible : bool
        Si ``True``, añade los sectores como etiquetas del eje Y.  El eje X
        queda reservado para la numeración de clusters.
    cluster_labels_visible : bool
        Si ``True``, numera los bloques de cluster en el eje X.
    xlabel : str
        Título del eje X.

    Returns
    -------
    Optional[plt.Figure]
        La figura, para poder guardarla; ``None`` si no hay activos comunes.
    """
    common = matrix.index.intersection(matrix.columns)
    if sector_labels is not None:
        common = common.intersection(sector_labels.index)
    if cluster_labels is not None:
        common = common.intersection(cluster_labels.index)

    if len(common) == 0:
        print("No hay activos comunes para ordenar.")
        return None

    if cluster_labels is not None:
        sub_clusters = cluster_labels.reindex(common)
        # Orden global por cluster; a igualdad, por sector para que los
        # bloques no salgan mezclados.
        if sector_labels is not None:
            sub_sectors = sector_labels.reindex(common)
            frame = pd.DataFrame({"cluster": sub_clusters, "sector": sub_sectors})
            order = frame.sort_values(["cluster", "sector"]).index.tolist()
        else:
            order = sub_clusters.sort_values().index.tolist()
    elif sector_labels is not None:
        sub_sectors = sector_labels.reindex(common)
        order = sorted(sub_sectors.unique(), key=str)
        order = [
            t
            for sector in order
            for t in sub_sectors[sub_sectors == sector].index.tolist()
        ]
    else:
        order = common.tolist()

    ordered_matrix = matrix.reindex(index=order, columns=order)

    # Escala medida: se ignoran los NaN de la diagonal.
    valores = ordered_matrix.to_numpy(dtype=float)
    finitos = valores[np.isfinite(valores)]
    if finitos.size == 0:
        print("La matriz no contiene valores finitos.")
        return None
    vmin, vmax = float(finitos.min()), float(finitos.max())

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        ordered_matrix, cmap=cmap, interpolation="nearest",
        vmin=vmin, vmax=vmax,
    )
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label=cbar_label)

# Separadores: sólo entre clusters.
    bloques: list[tuple[int, int]] = []
    if cluster_labels is not None:
        etiquetas = sub_clusters.loc[order]
        inicio = 0
        for i in range(1, len(order) + 1):
            if i == len(order) or etiquetas.iloc[i] != etiquetas.iloc[i - 1]:
                if inicio > 0:
                    ax.axhline(i - 0.5, color=separator_color, linewidth=1,
                               linestyle="--")
                    ax.axvline(i - 0.5, color=separator_color, linewidth=1,
                               linestyle="--")
                bloques.append((inicio, i - 1))
                inicio = i

    # Eje X: el número de cada cluster, centrado bajo su bloque.
    if bloques and cluster_labels_visible:
        # ``imshow`` sitúa el índice ``i`` en el píxel ``[i - 0.5, i + 0.5]``,
        # así que el centro del bloque ``[inicio, fin]`` es ``(inicio + fin) / 2``.
        ax.set_xticks([(ini + fin) / 2 for ini, fin in bloques])
        ax.set_xticklabels(
            [str(i) for i in range(1, len(bloques) + 1)], fontsize=8,
        )
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticks([])

    # Eje Y: sectores, sólo si se piden de forma explícita.
    if sector_labels_visible and sector_labels is not None:
        visibles = sector_labels.reindex(order)
        cambios = [
            i for i in range(1, len(order))
            if visibles.iloc[i] != visibles.iloc[i - 1]
        ]
        ax.set_yticks(cambios, [str(visibles.iloc[i]) for i in cambios],
                      fontsize=7)
    else:
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    return fig


# =============================================================================
# Pair trading
# =============================================================================


def plot_spread(
    spread: pd.Series,
    ou_params=None,
    figsize: tuple[float, float] = (12.0, 5.0),
    title: str = "Spread del par",
) -> None:
    """Spread con la media de equilibrio y bandas ``±k·sigma_eq`` del OU.

    Si ``ou_params`` es ``None`` se dibuja únicamente la serie y su media
    muestral.  Si se proporciona (salida de
    :func:`thesis.pair_trading.fit_ou_process`) se añaden las bandas de
    equilibrio ``±1σ``, ``±2σ`` y ``±3σ``.

    Parameters
    ----------
    spread : pd.Series
        Serie del spread estacionario ``Z_t``.
    ou_params : thesis.pair_trading.OUParameters, opcional
        Parámetros del OU ajustados al spread.
    figsize : tuple
        Tamaño de la figura.
    title : str
        Título de la gráfica.
    """
    if spread.empty:
        print("El spread está vacío: no hay nada que graficar.")
        return

    _, ax = plt.subplots(figsize=figsize)

    ax.plot(spread.index, spread.values, color="darkblue",
            linewidth=1.5, label="Spread (Z$_t$)")

    if ou_params is None:
        mu = float(spread.mean())
        ax.axhline(mu, color="black", linewidth=1.5,
                   label=f"Media muestral ({mu:.4f})")
        subtitle = ""
    else:
        mu = ou_params.mu
        sigma_eq = ou_params.sigma_eq
        ax.axhline(mu, color="black", linewidth=2,
                   label=f"Equilibrio (μ = {mu:.4f})")
        for width, color, alpha in (
            (sigma_eq, "green", 0.20),
            (2 * sigma_eq, "orange", 0.15),
            (3 * sigma_eq, "red", 0.10),
        ):
            ax.fill_between(spread.index, mu - width, mu + width,
                            color=color, alpha=alpha)
            ax.axhline(mu + width, color=color, linestyle="--",
                       linewidth=1, alpha=0.7)
            ax.axhline(mu - width, color=color, linestyle="--",
                       linewidth=1, alpha=0.7)
        subtitle = (f"bandas ±1/2/3 σ$_{{eq}}$ = "
                    f"{sigma_eq:.4f} | vida media "
                    f"{ou_params.half_life:.1f} sesiones")

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Spread")
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                 fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pair_prices(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    figsize: tuple[float, float] = (12.0, 5.0),
    title: str = "Precios del par",
    log_scale: bool = False,
) -> None:
    """Precios ajustados de los dos activos del par en un único eje.

    Parameters
    ----------
    prices : pd.DataFrame
        Frame con una columna por ticker.
    ticker_a, ticker_b : str
        Tickers a graficar (deben ser columnas de ``prices``).
    figsize : tuple
        Tamaño de la figura.
    title : str
        Título de la gráfica.
    log_scale : bool
        Si ``True`` usa escala logarítmica en el eje Y.
    """
    if prices.empty:
        print("No hay precios para graficar. Verifica los tickers o el rango.")
        return

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(prices.index, prices[ticker_a], linewidth=1.6,
            label=ticker_a)
    ax.plot(prices.index, prices[ticker_b], linewidth=1.6,
            label=ticker_b)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Precio (USD, escala log)")
    else:
        ax.set_ylabel("Precio (USD)")

    ax.set_xlabel("Fecha")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pair_overview(
    prices: pd.DataFrame,
    spread: pd.Series,
    ticker_a: str,
    ticker_b: str,
    ou_params=None,
    figsize: tuple[float, float] = (12.0, 8.0),
    log_scale: bool = True,
) -> None:
    """Vista conjunta del par: precios arriba, spread abajo.

    Ambos subplots comparten el eje temporal, de modo que se pueden
    relacionar las divergencias de precio con los movimientos del
    spread.

    Parameters
    ----------
    prices : pd.DataFrame
        Frame con una columna por ticker.
    spread : pd.Series
        Spread estacionario del par.
    ticker_a, ticker_b : str
        Tickers a graficar.
    ou_params : thesis.pair_trading.OUParameters, opcional
        Parámetros del OU; si se pasan, el subplot inferior incluye las
        bandas de equilibrio.
    figsize : tuple
        Tamaño de la figura.
    log_scale : bool
        Si ``True`` (por defecto) usa escala logarítmica en el subplot
        de precios.
    """
    if prices.empty or spread.empty:
        print("Faltan datos (precios o spread) para la vista conjunta.")
        return

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].plot(prices.index, prices[ticker_a], linewidth=1.5,
                 label=ticker_a)
    axes[0].plot(prices.index, prices[ticker_b], linewidth=1.5,
                 label=ticker_b)
    axes[0].set_ylabel("Precio (USD, escala log)" if log_scale
                       else "Precio (USD)")
    axes[0].set_title(f"{ticker_a} vs {ticker_b}", fontweight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    if log_scale:
        axes[0].set_yscale("log")

    axes[1].plot(spread.index, spread.values, color="darkblue",
                 linewidth=1.3, label="Spread (Z$_t$)")
    if ou_params is not None:
        mu = ou_params.mu
        axes[1].axhline(mu, color="black", linewidth=1.5,
                        label=f"Equilibrio (μ = {mu:.4f})")
        for width, color, alpha in (
            (ou_params.sigma_eq, "green", 0.20),
            (2 * ou_params.sigma_eq, "orange", 0.15),
        ):
            axes[1].fill_between(spread.index, mu - width, mu + width,
                                 color=color, alpha=alpha)
    else:
        axes[1].axhline(spread.mean(), color="black", linewidth=1.5,
                        linestyle="--", label="Media muestral")
    axes[1].set_ylabel("Spread")
    axes[1].set_xlabel("Fecha")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Par y spread", fontweight="bold")
    plt.tight_layout()
    plt.show()
