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
    ax = sns.heatmap(corr_matrix, cmap=cmap, center=0, cbar_kws={"label": cbar_label})

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
        k: v
        for k, v in (scatter_kwargs or {}).items()
        if k not in {"c", "cmap", "vmin", "vmax"}
    }
    kwargs = {**base_kwargs, **extra_kwargs}
    feature_windows = list(feature_by_window.keys())
    target_windows = list(target_by_window.keys())

    fig, axes = plt.subplots(
        len(target_windows),
        len(feature_windows),
        figsize=figsize,
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

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
                [
                    feature_by_window[fw].rename(feature_label),
                    target_by_window[tw].rename(target_label),
                ],
                axis=1,
            ).dropna()
            if data.empty:
                continue
            if color_by_date:
                scatter = ax.scatter(
                    data.iloc[:, 0],
                    data.iloc[:, 1],
                    c=mdates.date2num(data.index),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
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
        cbar = fig.colorbar(first_scatter, ax=axes, fraction=0.025, pad=0.02)
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
        data["quantile"] = pd.qcut(data[feature_name], q=n_quantiles, labels=labels)
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
        bin_labels = [f"{(lo + hi) / 2:.2g}" for lo, hi in zip(edges[:-1], edges[1:])]
        data["quantile"] = pd.cut(
            data[feature_name], bins=edges, include_lowest=True, labels=bin_labels
        )
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

        sns.boxplot(
            data=data,
            x="quantile",
            y=target_name,
            ax=ax_box,
            palette=palette,
            width=0.6,
            fliersize=3,
        )
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

        sns.boxplot(
            data=data,
            x="quantile",
            y=target_name,
            ax=ax,
            palette=palette,
            width=0.6,
            fliersize=3,
        )
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
            ax.plot(
                tau,
                corr_matrix[asset],
                linewidth=1,
                alpha=0.9,
                marker="o",
                markersize=5,
                label=asset,
            )
    else:
        for group_name, assets in asset_groups.items():
            color = (group_colors or {}).get(group_name)
            for asset in assets:
                if asset not in corr_matrix.columns:
                    continue
                kwargs = {
                    "linewidth": 1,
                    "alpha": 0.9,
                    "marker": "o",
                    "markersize": 5,
                    "label": group_name,
                }
                if color is not None:
                    kwargs["color"] = color
                ax.plot(tau, corr_matrix[asset], **kwargs)

    ax.plot(
        tau,
        avg_corr,
        color="black",
        linewidth=2.5,
        linestyle="--",
        marker="o",
        markersize=6,
        label=avg_label,
    )
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
        matrix,
        cmap=cmap,
        interpolation="nearest",
        vmin=float(finitos.min()),
        vmax=float(finitos.max()),
    )
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
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
    data = contingency.drop(index=["cluster_size", "cluster_purity"], errors="ignore")
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
    """Muestra únicamente la matriz reordenada y su barra de color.

    El orden es global por clúster, no «sector y dentro del sector»: así
    los bloques del heatmap son exactamente los clusters que evalúa el
    análisis sectorial.  No se dibujan separadores ni etiquetas para
    mantener únicamente la matriz y la barra de color.

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
        cada cluster.
    cluster_labels : pd.Series, opcional
        Etiqueta de cluster por ticker; si falta, la función no dibuja
        ningún separador ni numeración.
    cmap : str
        Paleta; por defecto ``viridis`` para compartir entre métricas.
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
        ordered_matrix,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    ax.set_xticks([])
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

    ax.plot(
        spread.index,
        spread.values,
        color="darkblue",
        linewidth=1.5,
        label="Spread (Z$_t$)",
    )

    if ou_params is None:
        mu = float(spread.mean())
        ax.axhline(mu, color="black", linewidth=1.5, label=f"Media muestral ({mu:.4f})")
        subtitle = ""
    else:
        mu = ou_params.mu
        sigma_eq = ou_params.sigma_eq
        ax.axhline(mu, color="black", linewidth=2, label=f"Equilibrio (μ = {mu:.4f})")
        for width, color, alpha in (
            (sigma_eq, "green", 0.20),
            (2 * sigma_eq, "orange", 0.15),
            (3 * sigma_eq, "red", 0.10),
        ):
            ax.fill_between(
                spread.index, mu - width, mu + width, color=color, alpha=alpha
            )
            ax.axhline(mu + width, color=color, linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(mu - width, color=color, linestyle="--", linewidth=1, alpha=0.7)
        subtitle = (
            f"bandas ±1/2/3 σ$_{{eq}}$ = "
            f"{sigma_eq:.4f} | vida media "
            f"{ou_params.half_life:.1f} sesiones"
        )

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Spread")
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontweight="bold")
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
    ax.plot(prices.index, prices[ticker_a], linewidth=1.6, label=ticker_a)
    ax.plot(prices.index, prices[ticker_b], linewidth=1.6, label=ticker_b)

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
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].plot(prices.index, prices[ticker_a], linewidth=1.5, label=ticker_a)
    axes[0].plot(prices.index, prices[ticker_b], linewidth=1.5, label=ticker_b)
    axes[0].set_ylabel("Precio (USD, escala log)" if log_scale else "Precio (USD)")
    axes[0].set_title(f"{ticker_a} vs {ticker_b}", fontweight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    if log_scale:
        axes[0].set_yscale("log")

    axes[1].plot(
        spread.index,
        spread.values,
        color="darkblue",
        linewidth=1.3,
        label="Spread (Z$_t$)",
    )
    if ou_params is not None:
        mu = ou_params.mu
        axes[1].axhline(
            mu, color="black", linewidth=1.5, label=f"Equilibrio (μ = {mu:.4f})"
        )
        for width, color, alpha in (
            (ou_params.sigma_eq, "green", 0.20),
            (2 * ou_params.sigma_eq, "orange", 0.15),
        ):
            axes[1].fill_between(
                spread.index, mu - width, mu + width, color=color, alpha=alpha
            )
    else:
        axes[1].axhline(
            spread.mean(),
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Media muestral",
        )
    axes[1].set_ylabel("Dispersión")
    axes[1].set_xlabel("Fecha")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # fig.suptitle("Par y spread", fontweight="bold")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Diagnóstico de backtests de pair trading
# =============================================================================


def _entry_exit_masks(results: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Máscaras de entrada y salida a partir de la serie de posiciones."""
    position = results["position"]
    entries = (position != 0) & (position.shift(1).fillna(0) == 0)
    exits = (position == 0) & (position.shift(1).fillna(0) != 0)
    return entries, exits


def plot_hedge_diagnostics(
    results: pd.DataFrame,
    in_sample_beta: float | None = None,
    beta_change_threshold: float = 0.01,
    figsize: tuple[float, float] = (14.0, 10.0),
) -> None:
    """Diagnóstico gráfico del hedge ratio walk-forward.

    Tres paneles:

    1. **Hedge ratio**: beta recalculado cada día frente al beta efectivo
       (escalonado dentro de cada operación).  Si el beta efectivo fuese
       una única constante para toda la serie, indicaría que se está
       reutilizando el beta de la última entrada.
    2. **Intercepto**: evolución de ``alpha``.
    3. **Cambios del beta**: magnitud de las variaciones diarias frente al
       umbral de cambio significativo.

    Parameters
    ----------
    results : pd.DataFrame
        Salida de :func:`thesis.pair_trading.pair_trading_backtest`.
    in_sample_beta : float | None
        Beta estimado con toda la muestra; se dibuja como referencia.
    beta_change_threshold : float
        Umbral a partir del cual se considera significativo el cambio.
    figsize : tuple
        Tamaño de la figura.
    """
    if results.empty:
        print("No hay resultados que graficar.")
        return

    _, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(
        results.index,
        results["beta"],
        color="darkblue",
        linewidth=1.2,
        label="Beta walk-forward (estimación diaria)",
    )
    axes[0].step(
        results.index,
        results["effective_beta"],
        where="post",
        color="darkorange",
        linewidth=1.5,
        label="Beta efectivo (usado por la cartera)",
    )
    if in_sample_beta is not None:
        axes[0].axhline(
            in_sample_beta,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"β in-sample = {in_sample_beta:.4f}",
        )
    axes[0].set_ylabel("Beta")
    axes[0].set_title(
        "Hedge ratio: estimación diaria vs beta efectivo", fontweight="bold"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results.index, results["alpha"], color="purple", linewidth=1.2, label="Alpha"
    )
    axes[1].set_ylabel("Alpha")
    axes[1].set_title("Intercepto walk-forward", fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    beta_change = results["beta"].diff().abs()
    axes[2].bar(
        results.index,
        beta_change.to_numpy(),
        width=1.0,
        color="steelblue",
        alpha=0.7,
        label="|Δβ| diario",
    )
    axes[2].axhline(
        beta_change_threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"umbral = {beta_change_threshold}",
    )
    axes[2].set_ylabel("|Δβ|")
    axes[2].set_xlabel("Fecha")
    axes[2].set_title("Cambios del hedge ratio", fontweight="bold")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_signal_diagnostics(
    results: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    figsize: tuple[float, float] = (14.0, 12.0),
) -> None:
    """Diagnóstico gráfico de las señales y de la ejecución.

    Tres paneles:

    1. **Spread** con marcadores de entrada y salida.
    2. **Z-score** con los umbrales de entrada y salida y la posición
       superpuesta en un eje secundario.
    3. **Días de permanencia** de cada operación.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    entry_z, exit_z : float
        Umbrales usados en el backtest.
    figsize : tuple
        Tamaño de la figura.
    """
    if results.empty:
        print("No hay resultados que graficar.")
        return

    entries, exits = _entry_exit_masks(results)
    _, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(
        results.index, results["spread"], color="navy", linewidth=1.2, label="Spread"
    )
    axes[0].plot(
        results.index[entries],
        results.loc[entries, "spread"],
        marker="^",
        linestyle="none",
        color="green",
        markersize=7,
        label="Entrada",
    )
    axes[0].plot(
        results.index[exits],
        results.loc[exits, "spread"],
        marker="v",
        linestyle="none",
        color="red",
        markersize=7,
        label="Salida",
    )
    axes[0].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Spread")
    axes[0].set_title("Spread con entradas y salidas", fontweight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results.index, results["z_score"], color="navy", linewidth=1.0, label="z-score"
    )
    axes[1].axhline(
        entry_z, color="red", linestyle="--", linewidth=1.2, label=f"±{entry_z}"
    )
    axes[1].axhline(-entry_z, color="red", linestyle="--", linewidth=1.2)
    axes[1].axhline(
        exit_z, color="green", linestyle=":", linewidth=1.2, label=f"salida {exit_z}"
    )
    axes[1].set_ylabel("z-score")
    axes[1].set_title("Z-score y umbrales de decisión", fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    twin = axes[1].twinx()
    twin.step(
        results.index,
        results["position"],
        where="post",
        color="orange",
        alpha=0.6,
        linewidth=1.0,
        label="Posición",
    )
    twin.set_ylabel("Posición")
    twin.set_yticks([-1, 0, 1])

    if "holding_days" in results.columns:
        axes[2].plot(
            results.index,
            results["holding_days"],
            color="teal",
            linewidth=1.2,
            label="Días de permanencia",
        )
    axes[2].set_ylabel("Días")
    axes[2].set_xlabel("Fecha")
    axes[2].set_title("Duración de cada operación", fontweight="bold")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_return_diagnostics(
    results: pd.DataFrame,
    figsize: tuple[float, float] = (14.0, 12.0),
) -> None:
    """Diagnóstico gráfico del retorno, el notional y los costes.

    Tres paneles:

    1. **Descomposición del retorno**: contribución de cada pata, retorno
       bruto, costes y retorno neto.
    2. **Notional y pesos**: notional bruto ``1 + |beta|`` y peso efectivo
       de cada activo.
    3. **Turnover y costes acumulados**.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    figsize : tuple
        Tamaño de la figura.
    """
    if results.empty:
        print("No hay resultados que graficar.")
        return

    _, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(
        results.index,
        results["contribution_a"],
        color="steelblue",
        linewidth=0.9,
        alpha=0.8,
        label="Contribución A",
    )
    axes[0].plot(
        results.index,
        results["contribution_b"],
        color="darkorange",
        linewidth=0.9,
        alpha=0.8,
        label="Contribución B",
    )
    axes[0].plot(
        results.index,
        results["gross_return"],
        color="black",
        linewidth=1.1,
        label="Retorno bruto",
    )
    axes[0].plot(
        results.index,
        results["costs"],
        color="red",
        linewidth=0.9,
        alpha=0.8,
        label="Costes",
    )
    axes[0].plot(
        results.index,
        results["strategy_ret"],
        color="green",
        linewidth=1.3,
        label="Retorno neto",
    )
    axes[0].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Retorno diario")
    axes[0].set_title("Descomposición del retorno diario", fontweight="bold")
    axes[0].legend(loc="best", ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results.index,
        results["gross_notional"],
        color="black",
        linewidth=1.2,
        label="Notional bruto (1 + |β|)",
    )
    axes[1].plot(
        results.index,
        results["weight_a"].abs(),
        color="steelblue",
        linewidth=1.0,
        label="|Peso A|",
    )
    axes[1].plot(
        results.index,
        results["weight_b"].abs(),
        color="darkorange",
        linewidth=1.0,
        label="|Peso B|",
    )
    axes[1].set_ylabel("Exposición")
    axes[1].set_title("Notional bruto y pesos efectivos", fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(
        results.index,
        results["turnover"].to_numpy(),
        width=1.0,
        color="steelblue",
        alpha=0.7,
        label="Turnover diario",
    )
    twin = axes[2].twinx()
    twin.plot(
        results.index,
        results["costs"].cumsum(),
        color="red",
        linewidth=1.4,
        label="Costes acumulados",
    )
    twin.set_ylabel("Costes acumulados")
    axes[2].set_ylabel("Turnover")
    axes[2].set_xlabel("Fecha")
    axes[2].set_title("Turnover y costes acumulados", fontweight="bold")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_equity_diagnostics(
    results: pd.DataFrame,
    figsize: tuple[float, float] = (14.0, 12.0),
    rolling_window: int = 252,
) -> None:
    """Diagnóstico gráfico del equity, drawdown y estabilidad temporal.

    Tres paneles:

    1. **Equity**: equity neta, equity bruta (sin costes) y benchmarks.
    2. **Drawdown** de la estrategia.
    3. **Rolling Sharpe y volatilidad** anualizada.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    figsize : tuple
        Tamaño de la figura.
    rolling_window : int
        Ventana de las métricas rolling, en sesiones.
    """
    if results.empty:
        print("No hay resultados que graficar.")
        return

    _, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    equity = results["equity"]
    gross_equity = np.exp(results["gross_return"].cumsum())
    axes[0].plot(
        equity.index, equity, color="green", linewidth=1.8, label="Estrategia (neto)"
    )
    axes[0].plot(
        gross_equity.index,
        gross_equity,
        color="darkgreen",
        linewidth=1.2,
        linestyle="--",
        label="Estrategia (sin costes)",
    )
    if "bh_equity_A" in results.columns:
        axes[0].plot(
            results.index,
            results["bh_equity_A"],
            color="orange",
            linewidth=1.0,
            linestyle=":",
            label="Buy & Hold A",
        )
        axes[0].plot(
            results.index,
            results["bh_equity_B"],
            color="blue",
            linewidth=1.0,
            linestyle=":",
            label="Buy & Hold B",
        )
    if "bh_equity_SPY" in results.columns:
        axes[0].plot(
            results.index,
            results["bh_equity_SPY"],
            color="black",
            linewidth=1.2,
            linestyle="-.",
            label="Buy & Hold SPY",
        )
    axes[0].axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Capital")
    axes[0].set_title("Curva de capital", fontweight="bold")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].fill_between(
        results.index,
        results["drawdown"].to_numpy(),
        0,
        color="red",
        alpha=0.4,
        label="Drawdown",
    )
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title("Drawdown de la estrategia", fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    rolling_mean = results["strategy_ret"].rolling(rolling_window).mean()
    rolling_std = results["strategy_ret"].rolling(rolling_window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    rolling_vol = rolling_std * np.sqrt(252)
    axes[2].plot(
        results.index,
        rolling_sharpe,
        color="purple",
        linewidth=1.2,
        label=f"Sharpe rolling ({rolling_window}d)",
    )
    axes[2].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    twin = axes[2].twinx()
    twin.plot(
        results.index,
        rolling_vol,
        color="brown",
        linewidth=1.0,
        alpha=0.7,
        label="Volatilidad anualizada",
    )
    twin.set_ylabel("Volatilidad anualizada")
    axes[2].set_ylabel("Sharpe rolling")
    axes[2].set_xlabel("Fecha")
    axes[2].set_title("Estabilidad temporal de la estrategia", fontweight="bold")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_equity(
    results: pd.DataFrame,
    figsize: tuple[float, float] = (14.0, 12.0),
    rolling_window: int = 252,
) -> None:
    """Diagnóstico gráfico del equity, drawdown y estabilidad temporal.

    Tres paneles:

    1. **Equity**: equity neta, equity bruta (sin costes) y benchmarks.
    2. **Drawdown** de la estrategia.
    3. **Rolling Sharpe y volatilidad** anualizada.

    Parameters
    ----------
    results : pd.DataFrame
        Salida del backtest.
    figsize : tuple
        Tamaño de la figura.
    rolling_window : int
        Ventana de las métricas rolling, en sesiones.
    """
    if results.empty:
        print("No hay resultados que graficar.")
        return

    plt.figure(figsize=figsize)

    equity = results["equity"]
    gross_equity = np.exp(results["gross_return"].cumsum())
    plt.plot(
        equity.index, equity, color="green", linewidth=1.8, label="Estrategia (neto)"
    )
    plt.plot(
        gross_equity.index,
        gross_equity,
        color="darkgreen",
        linewidth=1.2,
        linestyle="--",
        label="Estrategia (sin costes)",
    )
    if "bh_equity_A" in results.columns:
        plt.plot(
            results.index,
            results["bh_equity_A"],
            color="orange",
            linewidth=1.0,
            linestyle=":",
            label="Buy & Hold A",
        )
        plt.plot(
            results.index,
            results["bh_equity_B"],
            color="blue",
            linewidth=1.0,
            linestyle=":",
            label="Buy & Hold B",
        )
    if "bh_equity_SPY" in results.columns:
        plt.plot(
            results.index,
            results["bh_equity_SPY"],
            color="black",
            linewidth=1.2,
            linestyle="-.",
            label="Buy & Hold SPY",
        )
    plt.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    plt.set_ylabel("Capital")
    plt.set_title("Curva de capital", fontweight="bold")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.set_yscale("log")


def plot_backtest_dashboard(
    results: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    in_sample_beta: float | None = None,
    figsize: tuple[float, float] = (16.0, 22.0),
) -> None:
    """Dashboard de seis paneles para auditar un backtest de pair trading.

    Los paneles son, de arriba abajo:

    1. Beta walk-forward frente al beta efectivo.
    2. Spread con entradas y salidas marcadas.
    3. Z-score con umbrales y posición superpuesta.
    4. Contribución de cada pata, retorno bruto y retorno neto.
    5. Equity de la estrategia frente a los benchmarks.
    6. Drawdown y turnover diario.

    Parameters
    ----------
    results : pd.DataFrame
        Salida de :func:`thesis.pair_trading.pair_trading_backtest`.
    entry_z, exit_z : float
        Umbrales usados en el backtest.
    in_sample_beta : float | None
        Beta in-sample usado como referencia visual.
    figsize : tuple
        Tamaño de la figura.
    """
    if results.empty:
        print("No hay resultados que graficar.")
        return

    entries, exits = _entry_exit_masks(results)
    _, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)

    axes[0].plot(
        results.index,
        results["beta"],
        color="darkblue",
        linewidth=1.0,
        label="Beta walk-forward",
    )
    axes[0].step(
        results.index,
        results["effective_beta"],
        where="post",
        color="darkorange",
        linewidth=1.5,
        label="Beta efectivo",
    )
    if in_sample_beta is not None:
        axes[0].axhline(
            in_sample_beta,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"β in-sample = {in_sample_beta:.4f}",
        )
    axes[0].set_ylabel("Beta")
    axes[0].set_title("1. Hedge ratio walk-forward vs beta efectivo", fontweight="bold")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results.index, results["spread"], color="navy", linewidth=1.0, label="Spread"
    )
    axes[1].plot(
        results.index[entries],
        results.loc[entries, "spread"],
        marker="^",
        linestyle="none",
        color="green",
        markersize=6,
        label="Entrada",
    )
    axes[1].plot(
        results.index[exits],
        results.loc[exits, "spread"],
        marker="v",
        linestyle="none",
        color="red",
        markersize=6,
        label="Salida",
    )
    axes[1].set_ylabel("Spread")
    axes[1].set_title("2. Spread y operaciones", fontweight="bold")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        results.index, results["z_score"], color="navy", linewidth=0.9, label="z-score"
    )
    axes[2].axhline(entry_z, color="red", linestyle="--", linewidth=1.0)
    axes[2].axhline(-entry_z, color="red", linestyle="--", linewidth=1.0)
    axes[2].axhline(exit_z, color="green", linestyle=":", linewidth=1.0)
    twin = axes[2].twinx()
    twin.step(
        results.index,
        results["position"],
        where="post",
        color="orange",
        alpha=0.6,
        linewidth=0.9,
    )
    twin.set_yticks([-1, 0, 1])
    axes[2].set_ylabel("z-score")
    axes[2].set_title("3. Señales y posición", fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(
        results.index,
        results["contribution_a"],
        color="steelblue",
        linewidth=0.8,
        alpha=0.8,
        label="Contribución A",
    )
    axes[3].plot(
        results.index,
        results["contribution_b"],
        color="darkorange",
        linewidth=0.8,
        alpha=0.8,
        label="Contribución B",
    )
    axes[3].plot(
        results.index,
        results["strategy_ret"],
        color="green",
        linewidth=1.1,
        label="Retorno neto",
    )
    axes[3].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[3].set_ylabel("Retorno diario")
    axes[3].set_title("4. Descomposición del retorno", fontweight="bold")
    axes[3].legend(loc="best", fontsize=8, ncol=3)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(
        results.index,
        results["equity"],
        color="green",
        linewidth=1.6,
        label="Estrategia",
    )
    if "bh_equity_A" in results.columns:
        axes[4].plot(
            results.index,
            results["bh_equity_A"],
            color="orange",
            linewidth=1.0,
            linestyle=":",
            label="Buy & Hold A",
        )
        axes[4].plot(
            results.index,
            results["bh_equity_B"],
            color="blue",
            linewidth=1.0,
            linestyle=":",
            label="Buy & Hold B",
        )
    if "bh_equity_SPY" in results.columns:
        axes[4].plot(
            results.index,
            results["bh_equity_SPY"],
            color="black",
            linewidth=1.2,
            linestyle="-.",
            label="Buy & Hold SPY",
        )
    axes[4].axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    axes[4].set_ylabel("Capital")
    axes[4].set_title("5. Equity y benchmarks", fontweight="bold")
    axes[4].legend(loc="best", fontsize=8)
    axes[4].grid(True, alpha=0.3)

    axes[5].fill_between(
        results.index,
        results["drawdown"].to_numpy(),
        0,
        color="red",
        alpha=0.35,
        label="Drawdown",
    )
    twin = axes[5].twinx()
    twin.bar(
        results.index,
        results["turnover"].to_numpy(),
        width=1.0,
        color="steelblue",
        alpha=0.5,
        label="Turnover",
    )
    twin.set_ylabel("Turnover")
    axes[5].set_ylabel("Drawdown")
    axes[5].set_xlabel("Fecha")
    axes[5].set_title("6. Drawdown y turnover", fontweight="bold")
    axes[5].legend(loc="best", fontsize=8)
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_pair_window_summary(
    summary,
    full_spread: pd.Series | None = None,
    beta_history: pd.Series | None = None,
    figsize: tuple[float, float] = (14.0, 11.0),
) -> None:
    """Compara el estado completo del par con su ventana más reciente.

    Tres paneles:

    1. **Spread**: spread completo en línea fina y spread de la ventana
       reciente en línea gruesa, con las bandas ``±1σ`` y ``±2σ``
       calculadas sobre la ventana reciente.
    2. **Hedge ratio**: beta walk-forward (si se aporta) frente al beta
       de la muestra completa y al de la ventana reciente.
    3. **Z-score de la ventana reciente**: spread normalizado con la media
       y desviación de la propia ventana reciente, con los umbrales de
       entrada y salida.

    Parameters
    ----------
    summary : thesis.pair_trading.PairWindowSummary
        Resumen de la ventana reciente.
    full_spread : pd.Series | None
        Spread de la muestra completa; si se omite se usa el spread
        reconstruido con el hedge ratio de la muestra completa.
    beta_history : pd.Series | None
        Serie con el hedge ratio walk-forward para contrastar.
    figsize : tuple
        Tamaño de la figura.
    """
    if summary is None or summary.spread.empty:
        print("No hay datos de la ventana reciente para graficar.")
        return

    spread = summary.spread

    _, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)

    if full_spread is not None and not full_spread.empty:
        axes[0].plot(
            full_spread.index,
            full_spread,
            color="gray",
            linewidth=0.8,
            alpha=0.7,
            label="Spread muestra completa",
        )
    axes[0].plot(
        spread.index,
        spread,
        color="navy",
        linewidth=1.6,
        label=f"Spread última ventana ({summary.n_obs} obs.)",
    )

    mu = float(spread.mean())
    sigma = float(spread.std())
    for width, color, alpha in ((sigma, "green", 0.18), (2 * sigma, "orange", 0.14)):
        axes[0].fill_between(
            spread.index,
            mu - width,
            mu + width,
            color=color,
            alpha=alpha,
        )
    axes[0].axhline(mu, color="black", linewidth=1.2, label=f"μ = {mu:.4f}")
    axes[0].axhline(mu + 2 * sigma, color="orange", linestyle="--", linewidth=0.9)
    axes[0].axhline(mu - 2 * sigma, color="orange", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Spread")
    axes[0].set_title("Spread: muestra completa vs última ventana", fontweight="bold")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if beta_history is not None and not beta_history.empty:
        axes[1].plot(
            beta_history.index,
            beta_history,
            color="darkblue",
            linewidth=1.0,
            label="Beta walk-forward",
        )
    if summary.full_hedge is not None:
        axes[1].axhline(
            summary.full_hedge.beta,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"β muestra completa = {summary.full_hedge.beta:.4f}",
        )
    axes[1].axhline(
        summary.hedge.beta,
        color="darkorange",
        linewidth=1.6,
        label=f"β última ventana = {summary.hedge.beta:.4f}",
    )
    axes[1].set_ylabel("Beta")
    axes[1].set_title("Hedge ratio: completo vs ventana reciente", fontweight="bold")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    z_recent = (spread - mu) / sigma if sigma > 0 else spread * np.nan
    axes[2].plot(
        spread.index, z_recent, color="navy", linewidth=1.3, label="z-score (ventana)"
    )
    axes[2].axhline(2.0, color="red", linestyle="--", linewidth=1.1, label="±2")
    axes[2].axhline(-2.0, color="red", linestyle="--", linewidth=1.1)
    axes[2].axhline(0.0, color="green", linestyle=":", linewidth=1.1, label="salida")
    axes[2].set_ylabel("z-score")
    axes[2].set_xlabel("Fecha")
    axes[2].set_title("Z-score de la ventana reciente", fontweight="bold")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Banda informative con la vida media de la reversión en ambas muestras.
    if summary.ou is not None or summary.full_ou is not None:
        etiqueta = []
        if summary.full_ou is not None:
            etiqueta.append(f"completa: {summary.full_ou.half_life:.1f} sesiones")
        if summary.ou is not None:
            etiqueta.append(f"reciente: {summary.ou.half_life:.1f} sesiones")
        axes[2].set_xlabel(f"Fecha   (vida media OU — {'; '.join(etiqueta)})")

    plt.tight_layout()
    plt.show()
