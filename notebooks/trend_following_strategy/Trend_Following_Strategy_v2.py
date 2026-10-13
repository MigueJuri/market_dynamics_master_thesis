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
from thesis.timeseries import simple_returns
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

print(f"Loaded {len(close):,} bars  |  {close.index[0].date()} → {close.index[-1].date()}")


# %%
# ── Parameters ────────────────────────────────────────────────────────────
FAST_SPAN = 10
SLOW_SPAN = 60
BURN_IN   = SLOW_SPAN          # discard early EWMA transient

# Detectamos los eventos de cruce EWMA usando la función reutilizable.
events_cross = get_ewma_crossover_events(
    close, fast_span=FAST_SPAN, slow_span=SLOW_SPAN, burn_in=BURN_IN,
)
print(f"Crossover events detected: {len(events_cross)}")
print(f"  Bullish (+1): {(events_cross['side'] == 1).sum()}")
print(f"  Bearish (-1): {(events_cross['side'] == -1).sum()}")

# ── Quick diagnostic plot ─────────────────────────────────────────────────
fast_full = close.ewm(span=FAST_SPAN, adjust=False).mean()
slow_full = close.ewm(span=SLOW_SPAN, adjust=False).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(close.index, close,     label='Close',            color='steelblue', lw=0.8, alpha=0.7)
ax.plot(fast_full.index, fast_full, label=f'EWMA fast ({FAST_SPAN})', color='orange',   lw=1.2)
ax.plot(slow_full.index, slow_full, label=f'EWMA slow ({SLOW_SPAN})', color='green',    lw=1.2)

bull = events_cross[events_cross['side'] ==  1]
bear = events_cross[events_cross['side'] == -1]
ax.scatter(bull.index, close.loc[bull.index], marker='^', color='limegreen', s=60, zorder=5, label='Bullish X')
ax.scatter(bear.index, close.loc[bear.index], marker='v', color='red',       s=60, zorder=5, label='Bearish X')

ax.set_title('EWMA Crossover Events')
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
daily_ret = simple_returns(close)
trgt = ewma_volatility_target(daily_ret, span=VOL_SPAN)

# ── Vertical barrier: t1[i] = timestamp NUM_BARS bars after event i ───────
tEvents = events_cross.index
t1 = add_vertical_barrier(tEvents, close.index, NUM_BARS)

print(f"Volatility target — median: {trgt.median():.4f}  "
      f"| mean: {trgt.mean():.4f}")
print(f"Vertical barrier: {NUM_BARS} bars")


# %%
# ── Build triple-barrier event table ───────────────────────────────────────
side_series = events_cross['side'].astype(float)

labeled = get_events(
    close    = close,
    t_events = tEvents,
    pt_sl    = PT_SL,
    trgt     = 5 * trgt,
    min_ret  = MIN_RET,
    t1       = t1,
    side     = side_series,
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
price_ret    = close.pct_change().fillna(0)
strategy_ret = entry_pos * price_ret

cum_strat  = (1 + strategy_ret).cumprod()
cum_bh     = (1 + price_ret).cumprod()

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
fast_full = close.ewm(span=FAST_SPAN, adjust=False).mean()
slow_full = close.ewm(span=SLOW_SPAN, adjust=False).mean()

ax.plot(close.index,     close,     lw=0.7, color='steelblue', alpha=0.8, label='Close')
ax.plot(fast_full.index, fast_full, lw=1.2, color='orange',    label=f'EWMA({FAST_SPAN})')
ax.plot(slow_full.index, slow_full, lw=1.2, color='green',     label=f'EWMA({SLOW_SPAN})')

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
ax.set_ylabel('Price')

# ── Bottom panel: equity curves ───────────────────────────────────────────
ax2 = axes[1]
ax2.plot(cum_strat.index, cum_strat, label='Strategy',    color='darkorange', lw=1.2)
ax2.plot(cum_bh.index,    cum_bh,    label='Buy & Hold',  color='steelblue',  lw=1.0, alpha=0.7)
ax2.axhline(1.0, color='black', lw=0.6, ls=':')
ax2.set_title('Equity Curves (no transaction costs)')
ax2.set_ylabel('Cumulative Return')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()


# %%
# # Geometric daily conversion
# rf_daily = (1 + 0.03)**(1/252) - 1
# # Calculate excess returns
# strategy_excess_ret = strategy_ret - rf_daily
# bh_excess_ret = price_ret - rf_daily

# # Calculate Sharpe (using the original series for std to be computationally strict,
# # explicitly defining ddof=1 for a sample statistic)
# sharpe_strat = (strategy_excess_ret.mean() / strategy_ret.std(ddof=1)) * np.sqrt(252)
# sharpe_bh = (bh_excess_ret.mean() / price_ret.std(ddof=1)) * np.sqrt(252)

# fig, ax = plt.subplots(1, 1, figsize=(10, 5.625))
# set_figure_style()

# ax.plot(cum_strat.index, cum_strat, label=f'Estrategia (SR: {sharpe_strat:.4f})', lw=1.2, color = 'darkorange')
# ax.plot(cum_bh.index,    cum_bh,    label=f'Buy & Hold (SR: {sharpe_bh:.4f})', lw=1.0, color = 'steelblue', alpha=0.7)
# ax.set_title('Curvas de Capital (sin costos de transacción)')
# ax.set_ylabel('Retorno Acumulado')
# ax.legend(fontsize=8)

# plt.tight_layout()





# print(f"\nSharpe Ratios (annualized):")
# print(f"  Strategy:     {sharpe_strat:.4f}")
# print(f"  Buy & Hold:   {sharpe_bh:.4f}")

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
