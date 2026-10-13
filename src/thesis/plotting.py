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
