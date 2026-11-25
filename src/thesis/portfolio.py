"""Construcción de portfolios long-only, backtest walk-forward y métricas.

El módulo implementa tres esquemas de asignación (``1/N``, ``1/σ``,
``1/σ²``) y dos fuentes de volatilidad: EWMA histórica y predicción
HAR-RV con horizontes escalados por la frecuencia de rebalanceo.

Todas las operaciones de volatilidad y backtest son causales: los pesos
aplicados en ``t`` usan únicamente información disponible hasta ``t-1``.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform

from thesis.config import TRADING_DAYS
from thesis.plotting import _figsize
from thesis.volatility import compute_log_vol_target, log_trailing_realized_vol
from thesis.volatility_models import HARRVModel

VolatilityModel = Literal["ewma", "har_rv"]
PORTFOLIO_NAMES: tuple[str, str, str] = ("1/N", "1/σ", "1/σ²")
COLORS: dict[str, str] = {"1/N": "gray", "1/σ": "blue", "1/σ²": "green"}


def _tex(name: str) -> str:
    """Etiqueta de leyenda compatible con ``text.usetex`` (σ → ``$\\sigma$``)."""
    return name.replace("σ²", r"$\sigma^2$").replace("σ", r"$\sigma$")


def ewma_weights(n: int, lambda_: float = 0.94) -> np.ndarray:
    """Pesos EWMA normalizados (suman 1).

    El dato más reciente (final del array) pesa más.

    Parameters
    ----------
    n : int
        Número de observaciones.
    lambda_ : float
        Factor de decaimiento.

    Returns
    -------
    np.ndarray
        Vector de pesos de forma ``(n,)``.
    """
    if n <= 0:
        raise ValueError("n debe ser positivo.")
    w = lambda_ ** np.arange(n)[::-1]
    return w / w.sum()


def estimate_sigma(
    returns_df: pd.DataFrame,
    use_ewma: bool = True,
    lambda_: float = 0.94,
) -> pd.Series:
    """Volatilidad diaria por activo.

    EWMA: ``var = Σ w_t (r_t - μ_w)² / (1 - Σw²)``.  Con pesos uniformes
    se reduce a la varianza muestral (``ddof=1``).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Retornos logarítmicos diarios (columnas = activos).
    use_ewma : bool
        Si ``True`` usa EWMA; si ``False``, desviación estándar muestral.
    lambda_ : float
        Decaimiento EWMA.

    Returns
    -------
    pd.Series
        Volatilidad diaria por activo.
    """
    if not use_ewma:
        return returns_df.std(ddof=1)

    w = ewma_weights(len(returns_df), lambda_)
    X = returns_df.to_numpy()
    mu = w @ X
    var = (w[:, None] * (X - mu) ** 2).sum(axis=0) / (1.0 - (w**2).sum())
    return pd.Series(np.sqrt(var), index=returns_df.columns)


def build_har_rv_features(
    log_rets_df: pd.DataFrame,
    rebalance_freq: int,
    annualize: int = TRADING_DAYS,
) -> dict[str, pd.DataFrame]:
    """Construye features HAR-RV escaladas por ``rebalance_freq``.

    Para cada activo se generan tres series de log-volatilidad:
    ``rv_h``, ``rv_3h`` y ``rv_9h`` correspondientes a ventanas de
    ``H``, ``3H`` y ``9H`` días, donde ``H = rebalance_freq``.

    Parameters
    ----------
    log_rets_df : pd.DataFrame
        Retornos logarítmicos diarios por activo.
    rebalance_freq : int
        Horizonte de rebalanceo ``H`` en sesiones.
    annualize : int
        Factor de anualización.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario ``{activo: DataFrame con columnas rv_h, rv_3h, rv_9h}``.
    """
    if rebalance_freq <= 0:
        raise ValueError("rebalance_freq debe ser positivo.")

    features: dict[str, pd.DataFrame] = {}
    for col in log_rets_df.columns:
        lr = log_rets_df[col]
        features[col] = pd.DataFrame(
            {
                "rv_h": log_trailing_realized_vol(lr, rebalance_freq, annualize),
                "rv_3h": log_trailing_realized_vol(lr, 3 * rebalance_freq, annualize),
                "rv_9h": log_trailing_realized_vol(lr, 9 * rebalance_freq, annualize),
            }
        )
    return features


def _har_rv_target(
    log_rets_df: pd.DataFrame,
    rebalance_freq: int,
    annualize: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Target de log-volatilidad forward a ``rebalance_freq`` días.

    Crea un DataFrame con columna ``target_log_vol`` por activo usando
    la convención de :func:`thesis.volatility.compute_log_vol_target`.
    """
    targets: dict[str, pd.Series] = {}
    for col in log_rets_df.columns:
        # Reconstruimos precios exactos a partir de log-retornos para que
        # compute_log_vol_target recupere los retornos originales.
        prices = np.exp(log_rets_df[col].cumsum())
        proxy = pd.DataFrame({"Close": prices})
        targets[col] = compute_log_vol_target(proxy, rebalance_freq, annualize)
    return pd.concat(targets, axis=1)


def estimate_sigma_har_rv(
    log_rets_df: pd.DataFrame,
    rebalance_freq: int,
    lookback: int = 252 * 5,
    end_date: pd.Timestamp | None = None,
    min_obs: int = 60,
    annualize: int = TRADING_DAYS,
) -> pd.Series:
    """Predice volatilidad diaria por activo con un HAR-RV por columna.

    En cada activo se ajusta un :class:`thesis.volatility_models.HARRVModel`
    con una ventana expanding que termina en ``t-1-H`` (garantizando que
    todos los targets estén totalmente realizados), y se predice la
    log-volatilidad para ``t-1``.

    Parameters
    ----------
    log_rets_df : pd.DataFrame
        Retornos logarítmicos diarios por activo.
    rebalance_freq : int
        Horizonte ``H`` de rebalanceo en sesiones.
    lookback : int
        Tamaño máximo de la ventana de entrenamiento (expanding acotado).
    end_date : pd.Timestamp | None
        Fecha de corte ``t-1`` para entrenar/predicir.  Si ``None`` se usa
        el último índice disponible.
    min_obs : int
        Mínimo de observaciones de entrenamiento requeridas.
    annualize : int
        Factor de anualización.

    Returns
    -------
    pd.Series
        Predicción de volatilidad diaria ``σ̂ = exp(log σ̂)`` por activo.
    """
    if end_date is None:
        end_date = log_rets_df.index[-1]

    # Todos los cálculos se hacen sobre datos hasta end_date para garantizar
    # que ningún target use retornos posteriores al momento de predicción.
    data = log_rets_df.loc[:end_date]
    features_by_asset = build_har_rv_features(data, rebalance_freq, annualize)
    target_df = _har_rv_target(data, rebalance_freq, annualize)

    sigma_hat: dict[str, float] = {}
    feature_cols = ["rv_h", "rv_3h", "rv_9h"]

    for col in log_rets_df.columns:
        X = features_by_asset[col]
        y = target_df[col]

        # Solo targets totalmente realizados: target_df es NaN en las
        # últimas H filas de ``data`` (cuyo forward cruza end_date).
        mask = X[feature_cols].notna().all(axis=1) & y.notna()
        X_train = X.loc[mask, feature_cols]
        y_train = y.loc[mask].to_numpy()

        # Ventana expanding acotada.
        if len(X_train) > lookback:
            X_train = X_train.iloc[-lookback:]
            y_train = y_train[-lookback:]

        if len(X_train) < min_obs:
            sigma_hat[col] = np.nan
            continue

        model = HARRVModel(feature_cols=feature_cols)
        model.fit(X_train, y_train, maxlags=rebalance_freq)

        # Predicción en t-1 = end_date.
        X_pred = X.loc[[end_date], feature_cols]
        if X_pred.isna().any().any():
            sigma_hat[col] = np.nan
        else:
            log_vol_hat = model.predict(X_pred)[0]
            sigma_hat[col] = float(np.exp(log_vol_hat))

    return pd.Series(sigma_hat, index=log_rets_df.columns)


def build_portfolios(sigma: pd.Series) -> dict[str, pd.Series]:
    """Construye los tres portfolios long-only a partir de ``sigma``.

    Parameters
    ----------
    sigma : pd.Series
        Volatilidad diaria por activo.  Debe ser estrictamente positiva
        y sin NaN.

    Returns
    -------
    dict[str, pd.Series]
        Diccionario con pesos ``1/N``, ``1/σ`` y ``1/σ²``.
    """
    invalid = sigma[(sigma <= 0) | sigma.isna()]
    if not invalid.empty:
        raise ValueError(f"Sigma inválida en: {list(invalid.index)}")

    w_eq = pd.Series(1.0 / len(sigma), index=sigma.index)
    w_ivol = 1.0 / sigma
    w_ivol = w_ivol / w_ivol.sum()
    w_ivar = 1.0 / sigma**2
    w_ivar = w_ivar / w_ivar.sum()

    out = {"1/N": w_eq, "1/σ": w_ivol, "1/σ²": w_ivar}
    for k, w in out.items():
        if not np.isclose(w.sum(), 1.0) or (w < 0).any():
            raise ValueError(f"Pesos inválidos en {k}")
    return out


def run_backtest(
    returns_df: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: int = 21,
    tx_cost: float = 0.005,
    volatility_model: VolatilityModel = "ewma",
    use_ewma: bool = True,
    lambda_: float = 0.94,
    har_lookback: int = 252 * 5,
    har_min_obs: int = 60,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Backtest walk-forward de los tres portfolios.

    Los pesos se calculan con datos hasta ``t-1`` y se aplican al retorno
    del día ``t``.  Entre rebalanceos los pesos derivan con buy & hold;
    al rebalancear se paga ``tx_cost * Σ|w_target - w_drift|``.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Retornos logarítmicos diarios por activo.
    lookback : int
        Ventana de estimación de volatilidad (días).
    rebalance_freq : int
        Días entre rebalanceos.
    tx_cost : float
        Costo por unidad de turnover.
    volatility_model : {'ewma', 'har_rv'}
        Fuente de volatilidad para ``1/σ`` y ``1/σ²``.
    use_ewma : bool
        Para ``volatility_model='ewma'``: si ``False`` usa std muestral.
    lambda_ : float
        Decaimiento EWMA.
    har_lookback : int
        Ventana máxima de entrenamiento HAR-RV.
    har_min_obs : int
        Mínimo de observaciones para ajustar HAR-RV.
    verbose : bool
        Imprime progreso.

    Returns
    -------
    tuple
        ``(retornos_netos, historial_pesos, turnover)``.
    """
    if volatility_model not in ("ewma", "har_rv"):
        raise ValueError("volatility_model debe ser 'ewma' o 'har_rv'.")

    dates = returns_df.index
    n_days, n_assets = returns_df.shape
    R = (np.exp(returns_df) - 1.0).to_numpy()
    names = list(PORTFOLIO_NAMES)

    w_curr = {k: np.zeros(n_assets) for k in names}
    rets: dict[str, list[float]] = {k: [] for k in names}
    hist_w: dict[str, list[np.ndarray]] = {k: [] for k in names}
    hist_to: dict[str, list[float]] = {k: [] for k in names}
    reb_dates: list[pd.Timestamp] = []

    # La frecuencia de rebalanceo HAR necesita al menos 9H de historia para
    # las features, H para el target forward y min_obs observaciones de
    # entrenamiento: t >= 10*H + min_obs - 1.
    if volatility_model == "har_rv":
        min_history = 10 * rebalance_freq + har_min_obs - 1
    else:
        min_history = lookback
    start = max(lookback, min_history)

    if verbose:  # pragma: no cover
        print(
            f"Iniciando simulación sobre {n_days} días "
            f"(modelo={volatility_model}, λ={lambda_})..."
        )

    for t in range(start, n_days):
        rebalance = (t - start) % rebalance_freq == 0

        if rebalance:
            end_date = dates[t - 1]
            if volatility_model == "har_rv":
                sigma = estimate_sigma_har_rv(
                    returns_df.iloc[:t],
                    rebalance_freq=rebalance_freq,
                    lookback=har_lookback,
                    end_date=end_date,
                    min_obs=har_min_obs,
                )
            else:
                window = returns_df.iloc[t - lookback : t]
                sigma = estimate_sigma(window, use_ewma=use_ewma, lambda_=lambda_)
            targets = build_portfolios(sigma)
            reb_dates.append(dates[t])

        for k in names:
            cost = 0.0
            if rebalance:
                w_new = targets[k].reindex(returns_df.columns).fillna(0.0).to_numpy()
                turnover = np.abs(w_new - w_curr[k]).sum()
                cost = tx_cost * turnover
                w_curr[k] = w_new
                hist_w[k].append(w_new)
                hist_to[k].append(turnover)

            gross = float(w_curr[k] @ R[t])
            rets[k].append(gross - cost)
            w_curr[k] = w_curr[k] * (1.0 + R[t]) / (1.0 + gross)

        if verbose and t % 500 == 0:  # pragma: no cover
            print(f"Procesado: {dates[t].date()}")

    idx = dates[start:]
    rets_df = pd.DataFrame(rets, index=idx)
    weights = {
        k: pd.DataFrame(hist_w[k], index=reb_dates, columns=returns_df.columns)
        for k in names
    }
    turnover = pd.DataFrame(hist_to, index=reb_dates)
    return rets_df, weights, turnover


def calculate_metrics(
    series: pd.Series,
    name: str,
    turnover: pd.Series | None = None,
    trading_days: int = TRADING_DAYS,
) -> pd.Series:
    """Métricas clásicas de desempeño anualizadas.

    Parameters
    ----------
    series : pd.Series
        Retornos netos diarios.
    name : str
        Nombre de la estrategia.
    turnover : pd.Series | None
        Turnover por rebalanceo.
    trading_days : int
        Días de negocio al año.

    Returns
    -------
    pd.Series
        Métricas indexadas por nombre.
    """
    cum_ret = (1.0 + series).cumprod()
    years = len(series) / trading_days
    cagr = cum_ret.iloc[-1] ** (1.0 / max(years, 1e-12)) - 1.0
    vol = series.std() * np.sqrt(trading_days)
    sharpe = series.mean() / series.std() * np.sqrt(trading_days)
    max_dd = (cum_ret / cum_ret.cummax() - 1.0).min()
    ann_turnover = turnover.sum() / years if turnover is not None else np.nan
    return pd.Series(
        {
            "Strategy": name,
            "CAGR": cagr,
            "Vol (Ann)": vol,
            "Sharpe": sharpe,
            "MaxDD": max_dd,
            "Turnover/año": ann_turnover,
        }
    )


def plot_equity_drawdown(
    port_rets: pd.DataFrame,
    benchmark: pd.Series | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Gráfico de riqueza acumulada y drawdown.

    Parameters
    ----------
    port_rets : pd.DataFrame
        Retornos netos diarios de los portfolios.
    benchmark : pd.Series | None
        Retornos netos diarios de un benchmark opcional.
    figsize : tuple
        Tamaño de la figura; por defecto ancho ``figure.figsize`` del estilo.

    Returns
    -------
    plt.Figure
        Figura con los dos subplots.
    """
    fig, axes = plt.subplots(2, 1, figsize=_figsize(figsize, 0.67))
    cum = (1.0 + port_rets).cumprod()

    ax = axes[0]
    for k in cum.columns:
        ax.plot(cum.index, cum[k], label=_tex(k), color=COLORS.get(k, None), linewidth=1.5)
    if benchmark is not None:
        bench_cum = (1.0 + benchmark).cumprod()
        ax.plot(bench_cum.index, bench_cum, label="Benchmark", color="red", alpha=0.5)
    ax.set_title("Riqueza Acumulada (neta de costos)")
    ax.set_ylabel(r"Crecimiento (\$1 invertido)")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    for k in cum.columns:
        dd = cum[k] / cum[k].cummax() - 1.0
        ax.plot(
            dd.index,
            dd,
            label=f"Drawdown {_tex(k)}",
            color=COLORS.get(k, None),
            linewidth=1.0,
        )
    if benchmark is not None:
        bench_cum = (1.0 + benchmark).cumprod()
        dd_bench = bench_cum / bench_cum.cummax() - 1.0
        ax.plot(
            dd_bench.index, dd_bench, label="Drawdown Benchmark", color="red", alpha=0.3
        )
    ax.set_title("Perfil de Riesgo: Maximum Drawdown")
    ax.set_ylabel("Caída desde máximo")
    ax.legend()

    return fig


def plot_portfolio_composition(
    weights_df: pd.DataFrame,
    title: str = "Evolución de Pesos",
    figsize: tuple[float, float] | None = None,
    events: list[tuple[str, str, str]] | None = None,
) -> plt.Figure:
    """Stacked area de la evolución de pesos.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Pesos por fecha (filas) y activo (columnas).
    title : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura; por defecto ancho ``figure.figsize`` del estilo.
    events : list | None
        Lista de ``(fecha, etiqueta, color)`` para líneas verticales.

    Returns
    -------
    plt.Figure
        Figura con el área apilada.
    """
    if weights_df.empty:
        raise ValueError("weights_df está vacío.")

    fig, ax = plt.subplots(figsize=_figsize(figsize, 0.75))
    cols = plt.cm.tab20(np.linspace(0.0, 1.0, len(weights_df.columns)))
    ax.stackplot(
        weights_df.index,
        weights_df.T,
        labels=weights_df.columns,
        colors=cols,
        alpha=0.85,
    )

    ax.set_title(title)
    ax.set_ylabel("Asignación de peso")
    ax.set_xlabel("Tiempo")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(weights_df.index[0], weights_df.index[-1])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    # Leyenda bajo el eje: a ancho de texto no cabe una columna lateral.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(labels), 7),
        title="Activos",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    if events is None:
        events = [
            ("2020-03-01", "COVID Crash", "black"),
            ("2022-01-01", "Inflación/Tasas", "red"),
        ]
    for date, label, color in events:
        d = pd.to_datetime(date)
        if weights_df.index[0] < d < weights_df.index[-1]:
            ax.axvline(d, color=color, linestyle="--", linewidth=1.5)
            ax.text(
                d, 0.98, f" {label}", color=color, ha="left", va="top",
                rotation=90, transform=ax.get_xaxis_transform(),
            )

    return fig


def plot_avg_weights(
    weights_dict: dict[str, pd.DataFrame],
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Barras de peso promedio por portfolio.

    Parameters
    ----------
    weights_dict : dict[str, pd.DataFrame]
        Diccionario de pesos por portfolio.
    figsize : tuple
        Tamaño de la figura; por defecto ancho ``figure.figsize`` del estilo.

    Returns
    -------
    plt.Figure
        Figura con barras agrupadas.
    """
    avg = pd.DataFrame({k: w.mean() for k, w in weights_dict.items()})
    sort_col = "1/σ²" if "1/σ²" in avg.columns else avg.columns[0]
    avg = avg.sort_values(sort_col, ascending=False)

    fig, ax = plt.subplots(figsize=_figsize(figsize, 0.5))
    avg.rename(columns=_tex).plot.bar(
        ax=ax,
        color=[COLORS.get(c, None) for c in avg.columns],
        alpha=0.85,
        width=0.8,
    )
    ax.set_title("Composición promedio por portfolio")
    ax.set_ylabel("Peso promedio")
    ax.set_xlabel("Activos")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig


def plot_market_topology(
    returns_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    ax: plt.Axes | None = None,
    title: str = "Dendrograma de correlaciones",
) -> plt.Figure:
    """Dendrograma jerárquico de la matriz de correlaciones.

    La distancia se define como ``sqrt(0.5 * (1 - ρ))`` y se condensa
    correctamente mediante :func:`scipy.spatial.distance.squareform`.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Retornos logarítmicos diarios por activo.
    start_date : str
        Fecha de inicio del régimen.
    end_date : str
        Fecha de fin del régimen.
    ax : plt.Axes | None
        Ejes opcionales.  Si ``None`` se crea una figura nueva.
    title : str
        Título del subplot.

    Returns
    -------
    plt.Figure
        Figura con el dendrograma.
    """
    period_data = returns_df.loc[start_date:end_date]
    if period_data.empty:
        raise ValueError(f"No hay datos para {start_date}:{end_date}")

    return_fig = ax is None
    if return_fig:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    corr = period_data.corr()
    dist = np.sqrt(0.5 * (1.0 - corr))
    linkage_matrix = sch.linkage(squareform(dist, checks=False), method="single")

    sch.dendrogram(
        linkage_matrix,
        labels=period_data.columns,
        ax=ax,
        leaf_rotation=90,
        above_threshold_color="#AAAAAA",
    )
    ax.set_title(title)
    ax.set_ylabel("Distancia ultramétrica")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    return fig
