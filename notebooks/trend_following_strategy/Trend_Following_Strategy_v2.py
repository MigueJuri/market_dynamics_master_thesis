# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: tesis
#     language: python
#     name: python3
# ---

# %%
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# Añadimos ``src/`` al path para poder importar ``thesis.*`` desde cualquier
# ubicación relativa dentro del repositorio.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # raíz del repositorio (``thesis/``)
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis.config import DATA_DIR, TICKER, TRADING_DAYS
from thesis.market_data import load_single_price
from thesis.timeseries import log_returns
from thesis.event_labeling import (
    add_vertical_barrier,
    compute_overlap_weights,
    ewma_volatility_target,
    get_events,
    get_ewma_crossover_events,
)

# %%
# ── Load data ──────────────────────────────────────────────────────────────
# Usamos el CSV cacheado de SPY generado por ``load_single_price``.
# Si no existiera, la función lo descargaría con yfinance.

price = load_single_price(TICKER, cache_dir=DATA_DIR)
close = price.astype(float)
log_close = np.log(close)
log_ret = log_returns(close)   # retornos logarítmicos diarios

print(f"Loaded {len(close):,} bars  |  {close.index[0].date()} → {close.index[-1].date()}")


# %%
# ── Parameters ────────────────────────────────────────────────────────────
FAST_SPAN = 40
SLOW_SPAN = 160
BURN_IN   = SLOW_SPAN          # discard early EWMA transient

# Detectamos los eventos de cruce EWMA sobre el log-precio.
# Esto hace que las medias sean proporcionales al nivel de precios y los
# cruces sean simétricos en términos porcentuales.
events_cross = get_ewma_crossover_events(
    log_close, fast_span=FAST_SPAN, slow_span=SLOW_SPAN, burn_in=BURN_IN,
)
print(f"Crossover events detected: {len(events_cross)}")
print(f"  Bullish (+1): {(events_cross['side'] == 1).sum()}")
print(f"  Bearish (-1): {(events_cross['side'] == -1).sum()}")

# ── Quick diagnostic plot ─────────────────────────────────────────────────
fast_log = log_close.ewm(span=FAST_SPAN, adjust=False).mean()
slow_log = log_close.ewm(span=SLOW_SPAN, adjust=False).mean()

# Convertimos las medias logarítmicas a escala de precios para dibujarlas
# junto al precio original; el eje Y usa escala logarítmica.
fast_price = np.exp(fast_log)
slow_price = np.exp(slow_log)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(close.index, close, label='Close', color='steelblue', lw=0.8, alpha=0.7)
ax.plot(fast_price.index, fast_price, label=f'EWMA log fast ({FAST_SPAN})', color='orange', lw=1.2)
ax.plot(slow_price.index, slow_price, label=f'EWMA log slow ({SLOW_SPAN})', color='green', lw=1.2)

bull = events_cross[events_cross['side'] ==  1]
bear = events_cross[events_cross['side'] == -1]
ax.scatter(bull.index, close.loc[bull.index], marker='^', color='limegreen', s=60, zorder=5, label='Bullish X')
ax.scatter(bear.index, close.loc[bear.index], marker='v', color='red',       s=60, zorder=5, label='Bearish X')

ax.set_title('EWMA Crossover Events on Log-Price (log scale)')
ax.set_ylabel('Price (USD, log scale)')
ax.set_yscale('log')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# %%
# ── Triple-barrier parameters ─────────────────────────────────────────────
PT_SL      = [1.0, 1.0]   # profit-take and stop-loss multipliers (symmetric)
MIN_RET    = 1e-4          # minimum target size to keep an event
VOL_SPAN   = 20            # EWM span for daily volatility estimate
NUM_BARS   = 63           # vertical barrier width (bars)

# ── Daily EWMA volatility as the barrier target ───────────────────────────
# La volatilidad objetivo se calcula sobre retornos logarítmicos para que
# las barreras PT/SL sean simétricas en el espacio logarítmico.
trgt = ewma_volatility_target(log_ret, span=VOL_SPAN)

# ── Vertical barrier: t1[i] = timestamp NUM_BARS bars after event i ───────
tEvents = events_cross.index
t1 = add_vertical_barrier(tEvents, close.index, NUM_BARS)

print(f"Log-volatility target — median: {trgt.median():.4f}  "
      f"| mean: {trgt.mean():.4f}")
print(f"Vertical barrier: {NUM_BARS} bars")


# %%
# ── Build triple-barrier event table ───────────────────────────────────────
side_series = events_cross['side'].astype(float)

labeled = get_events(
    close       = log_close,   # log-precios para barreras simétricas logarítmicas
    t_events    = tEvents,
    pt_sl       = PT_SL,
    trgt        = 5 * trgt,
    min_ret     = MIN_RET,
    t1          = t1,
    side        = side_series,
    return_type = "log",
)

print(f"Labeled events: {len(labeled)}")
print(labeled['label'].value_counts().rename({1:'PT (+1)', -1:'SL (-1)', 0:'Vertical (0)'}))
print("\nSample:")
print(labeled.head(8).to_string())


# %%
# ── Position series ───────────────────────────────────────────────────────
# Rule: enter on bullish crossover (+1), exit / flip on bearish (-1).
# When a new signal overlaps an active trade the position updates immediately.
# Bearish crossover → go flat (0), NOT short.

signal_series = events_cross['side'].reindex(close.index).fillna(0).astype(int)
signal_series.iloc[0] = 1  # assume we start in a long position before any crossover


# Replace -1 (bearish) with 0 (flat) for position purposes.
marker = signal_series.replace(-1, 0).astype(float)
marker[signal_series == -1] = 0.0   # explicit exit to flat

# Forward-fill: hold position until next signal.
position = marker.where(signal_series != 0).ffill().fillna(0).astype(int)

# Shift by 1 bar to avoid look-ahead bias (execute on the bar after signal).
entry_pos = position.shift(1).fillna(0).astype(int)

# ── Returns ───────────────────────────────────────────────────────────────
# Los retornos de la estrategia se calculan en el espacio logarítmico para
# coherencia con la construcción de señales y barreras, y luego se convierten
# a retornos simples para la acumulación y los prints.
strategy_log_ret = entry_pos * log_ret
strategy_ret = np.expm1(strategy_log_ret)

# Buy & hold en términos logarítmicos, convertido a simples.
bh_log_ret = log_ret
bh_ret = np.expm1(bh_log_ret)

cum_strat = (1 + strategy_ret).cumprod()
cum_bh    = (1 + bh_ret).cumprod()

total_strat = cum_strat.iloc[-1] - 1
total_bh    = cum_bh.iloc[-1]    - 1
n_trades    = (entry_pos.diff().abs() > 0).sum()

print(f"Strategy total return : {total_strat:+.2%}")
print(f"Buy & Hold total return: {total_bh:+.2%}")
print(f"Number of position changes: {n_trades}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                         gridspec_kw={'height_ratios': [2, 1]})

# ── Top panel: price + EWMA + events ─────────────────────────────────────
ax = axes[0]
fast_log = log_close.ewm(span=FAST_SPAN, adjust=False).mean()
slow_log = log_close.ewm(span=SLOW_SPAN, adjust=False).mean()
fast_price = np.exp(fast_log)
slow_price = np.exp(slow_log)

ax.plot(close.index, close, lw=0.7, color='steelblue', alpha=0.8, label='Close')
ax.plot(fast_price.index, fast_price, lw=1.2, color='orange', label=f'EWMA log({FAST_SPAN})')
ax.plot(slow_price.index, slow_price, lw=1.2, color='green', label=f'EWMA log({SLOW_SPAN})')

# Shade background by active long position.
in_long = False
start_long = None
for ts, pos in entry_pos.items():
    if pos == 1 and not in_long:
        start_long = ts
        in_long    = True
    elif pos != 1 and in_long:
        ax.axvspan(start_long, ts, color='limegreen', alpha=0.10)
        in_long = False
if in_long:
    ax.axvspan(start_long, close.index[-1], color='limegreen', alpha=0.10)

# Mark barrier outcomes on entry points.
for ts, row in labeled.iterrows():
    col = {1: 'limegreen', -1: 'red', 0: 'grey'}[row['side']]
    mk  = {1: '^',          -1: 'v',  0: 'o'}[row['side']]
    ax.scatter(ts, close.loc[ts], marker=mk, color=col, s=55, zorder=6)
    # Vertical barrier line.
    if pd.notna(row['t_hit']):
        ax.axvline(row['t_hit'], color=col, lw=0.4, ls='--', alpha=0.4)

pt_patch  = mpatches.Patch(color='limegreen', label='PT hit (+1)')
sl_patch  = mpatches.Patch(color='red',       label='SL hit (-1)')
vb_patch  = mpatches.Patch(color='grey',      label='Vertical (0)')
long_patch = mpatches.Patch(color='limegreen', alpha=0.3, label='Long period')
ax.legend(handles=[pt_patch, sl_patch, vb_patch, long_patch] +
          ax.get_legend_handles_labels()[0][:3], fontsize=7)
ax.set_title('Price, EWMA Crossovers, and Triple-Barrier Outcomes')
ax.set_ylabel('Price (USD, log scale)')
ax.set_yscale('log')

# ── Bottom panel: equity curves ───────────────────────────────────────────
ax2 = axes[1]
ax2.plot(cum_strat.index, cum_strat, label='Strategy',    color='darkorange', lw=1.2)
ax2.plot(cum_bh.index,    cum_bh,    label='Buy & Hold',  color='steelblue',  lw=1.0, alpha=0.7)
ax2.axhline(1.0, color='black', lw=0.6, ls=':')
ax2.set_title('Equity Curves (no transaction costs)')
ax2.set_ylabel('Cumulative Return')
ax2.set_yscale('log')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()


# %%
# %%
# ── Annualized risk / return metrics ──────────────────────────────────────
# Tasa libre de riesgo anualizada (3 % anual) convertida a retorno logarítmico
# diario para mantener la coherencia con el espacio en el que se calculan
# los retornos de la estrategia.
ANNUAL_RF = 0.03
rf_log_daily = np.log1p(ANNUAL_RF) / TRADING_DAYS

# Excesos de retorno en el espacio logarítmico.
strategy_excess_log = strategy_log_ret - rf_log_daily
bh_excess_log       = bh_log_ret       - rf_log_daily

# Convertimos a retornos simples para el cálculo de Sharpe (media/std sobre
# muestras diarias) y aplicamos la corrección de Bessel (ddof=1).
strategy_excess_simple = np.expm1(strategy_excess_log)
bh_excess_simple       = np.expm1(bh_excess_log)

sharpe_strat = (
    strategy_excess_simple.mean() / strategy_excess_simple.std(ddof=1)
) * np.sqrt(TRADING_DAYS)
sharpe_bh = (
    bh_excess_simple.mean() / bh_excess_simple.std(ddof=1)
) * np.sqrt(TRADING_DAYS)

# Rentabilidad anualizada (CAGR) a partir del retorno total compuesto.
n_days = (close.index[-1] - close.index[0]).days
n_years = n_days / 365.25
ann_strat = (1 + total_strat) ** (1 / n_years) - 1 if n_years > 0 else np.nan
ann_bh    = (1 + total_bh)    ** (1 / n_years) - 1 if n_years > 0 else np.nan

# Volatilidad anualizada sobre retornos simples diarios.
vol_strat = strategy_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
vol_bh    = bh_ret.std(ddof=1)       * np.sqrt(TRADING_DAYS)

# Maximum drawdown sobre la curva de capital.
def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())

max_dd_strat = _max_drawdown(cum_strat)
max_dd_bh    = _max_drawdown(cum_bh)

print(f"Period                       : {close.index[0].date()} → "
      f"{close.index[-1].date()}  ({n_years:.2f} years)")
print(f"Risk-free (annual)            : {ANNUAL_RF:.2%}")
print()
print(f"{'Metric':<28}{'Strategy':>14}{'Buy & Hold':>14}")
print("-" * 56)
print(f"{'Total return':<28}{total_strat:>+14.2%}{total_bh:>+14.2%}")
print(f"{'Annualized return (CAGR)':<28}{ann_strat:>+14.2%}{ann_bh:>+14.2%}")
print(f"{'Annualized volatility':<28}{vol_strat:>14.2%}{vol_bh:>14.2%}")
print(f"{'Sharpe ratio (ann., rf=3%)':<28}{sharpe_strat:>14.4f}{sharpe_bh:>14.4f}")
print(f"{'Max drawdown':<28}{max_dd_strat:>14.2%}{max_dd_bh:>14.2%}")
print(f"{'Number of position changes':<28}{int(n_trades):>14d}{'-':>14s}")

# ── Annualized equity curves plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.625))
ax.plot(cum_strat.index, cum_strat,
        label=f'Strategy (SR: {sharpe_strat:.2f}, CAGR: {ann_strat:.2%})',
        lw=1.2, color='darkorange')
ax.plot(cum_bh.index, cum_bh,
        label=f'Buy & Hold (SR: {sharpe_bh:.2f}, CAGR: {ann_bh:.2%})',
        lw=1.0, color='steelblue', alpha=0.7)
ax.axhline(1.0, color='black', lw=0.6, ls=':')
ax.set_title('Equity Curves (no transaction costs)')
ax.set_ylabel('Cumulative Return')
ax.set_yscale('log')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# %% [markdown]
# To understand them, we first define the outcomes based on the Confusion Matrix:True Positive (TP): The meta-model said "Bet" ($ŷ=1$), and the primary signal hit the profit target ($y=1$).False Positive (FP): The meta-model said "Bet" ($ŷ=1$), but the primary signal hit the stop-loss or timed out ($y=0$).False Negative (FN): The meta-model said "No Bet" ($ŷ=0$), but the primary signal would have hit the profit target ($y=1$).True Negative (TN): The meta-model said "No Bet" ($ŷ=0$), and the primary signal indeed failed ($y=0$).1. Precision (The Reliability)Precision answers: "Of all the trades the meta-model told me to take, how many were actually winners?"In finance, precision is often the most critical metric because it directly relates to your hit rate and the avoidance of losing trades.$$Precision = \frac{TP}{TP + FP}$$High Precision: Means you have very few "False Alarms." When the model says to trade, it is usually right.Low Precision: Means you are taking many trades that end up hitting the stop-loss.2. Recall (The Coverage)Recall answers: "Of all the potential winning opportunities the primary model generated, how many did the meta-model actually catch?"Recall measures your "opportunity capture."$$Recall = \frac{TP}{TP + FN}$$High Recall: Means you are not "missing out" on many winners. You are betting on most of the signals that would have been profitable.Low Recall: Means your filter is very strict. You might be skipping many profitable trades to ensure that the ones you do take are safe.3. F1-Score (The Balance)The F1-Score is the harmonic mean of Precision and Recall. It provides a single number to measure the performance of your filter by penalizing extreme values of either metric.$$F1 = 2 \cdot \frac{Precision \times Recall}{Precision + Recall}$$Why use it? If you have a model with 100% Precision but it only takes 1 trade a year (0.01% Recall), the model is practically useless for generating returns. Conversely, if you take every trade (100% Recall) bu...

# %%
# ── Overlap-weighted F1 ───────────────────────────────────────────────────
# Each event is treated independently (barrier outcome is not updated by
# overlapping signals). Events are weighted by the fraction of their
# barrier window that is NOT overlapped by the next event.
#
# Formally, let event i have window [t_i, t_hit_i].  If event j starts at
# t_j with t_i < t_j < t_hit_i, the unique (non-overlapped) length is:
#
#       unique_i = (t_j - t_i) / (t_hit_i - t_i)
#
# If no overlap exists, unique_i = 1.  This mirrors López de Prado's
# average-uniqueness weighting (AFML §4.3).

weights = compute_overlap_weights(labeled)

# Drop events with no barrier hit (label == 0 from vertical barrier can be
# treated as 'no signal' for F1; include or exclude depending on your
# research design — here we include them as negative class).
y_true = labeled['label'].values          # +1 or -1 or 0
y_pred = labeled['side'].astype(int).values  # signal direction (+1 or -1)

# Binarise: did the signal predict the correct direction?
# Correct = 1 if label == side, else 0.
y_true_bin = (y_true == y_pred).astype(int)
y_pred_bin = np.ones(len(y_true_bin), dtype=int)   # we always take the signal

w = weights.values

precision_unweighted = precision_score(y_true_bin, y_pred_bin, zero_division=0)
recall_unweighted = recall_score(y_true_bin, y_pred_bin, zero_division=0)
f1_unweighted = f1_score(y_true_bin, y_pred_bin, zero_division=0)
precision_weighted = precision_score(y_true_bin, y_pred_bin, sample_weight=w, zero_division=0)
recall_weighted = recall_score(y_true_bin, y_pred_bin, sample_weight=w, zero_division=0)
f1_weighted   = f1_score(y_true_bin, y_pred_bin,
                         sample_weight=w, zero_division=0)

hit_rate       = y_true_bin.mean()
hit_rate_w     = np.average(y_true_bin, weights=w)

print("=" * 50)
print("Triple-Barrier F1 Evaluation")
print("=" * 50)
print(f"  Events evaluated          : {len(labeled)}")
print(f"  Mean overlap weight       : {w.mean():.4f}")
print(f"  Hit rate (unweighted)     : {hit_rate:.4f}")
print(f"  Hit rate (weighted)       : {hit_rate_w:.4f}")
print(f"  Precision (unweighted)    : {precision_unweighted:.4f}")
print(f"  Recall (unweighted)       : {recall_unweighted:.4f}")
print(f"  Precision (weighted)      : {precision_weighted:.4f}")
print(f"  Recall (weighted)         : {recall_weighted:.4f}")
print(f"  F1 (unweighted)           : {f1_unweighted:.4f}")
print(f"  F1 (overlap-weighted)     : {f1_weighted:.4f}")
print()
print("Label distribution:")
print(labeled['label'].value_counts()
      .rename({1:'PT (+1)', -1:'SL (-1)', 0:'Vertical (0)'}))
print()
print("Weight distribution (quantiles):")
print(weights.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).round(4))

# %%
