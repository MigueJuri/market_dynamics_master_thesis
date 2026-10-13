"""Paquete ``thesis``: utilidades compartidas para los notebooks de investigación."""

from thesis.config import DATA_DIR, END, SEED, START, TICKER, TRADING_DAYS
from thesis.event_labeling import (
    add_vertical_barrier,
    apply_barriers,
    compute_overlap_weights,
    ewma_volatility_target,
    get_events,
    get_ewma_crossover_events,
)
from thesis.market_data import load_multiple_prices, load_single_price
from thesis.plotting import (
    plot_correlation_matrix,
    plot_cross_sectional_curves,
    plot_feature_boxplots,
    plot_scatter_grid,
)
from thesis.timeseries import RollingOp, apply_window, log_returns, simple_returns

__all__ = [
    "DATA_DIR",
    "END",
    "RollingOp",
    "SEED",
    "START",
    "TICKER",
    "TRADING_DAYS",
    "add_vertical_barrier",
    "apply_barriers",
    "apply_window",
    "compute_overlap_weights",
    "ewma_volatility_target",
    "get_events",
    "get_ewma_crossover_events",
    "load_multiple_prices",
    "load_single_price",
    "log_returns",
    "plot_correlation_matrix",
    "plot_cross_sectional_curves",
    "plot_feature_boxplots",
    "plot_scatter_grid",
    "simple_returns",
]
