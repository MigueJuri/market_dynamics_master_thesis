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

fig, ax = plt.subplots(figsize=(6, 3), dpi = 300)
ax.plot(close.index, close, label='Precio', color='steelblue', lw=0.8, alpha=0.7)
ax.plot(fast_price.index, fast_price, label=f'EWMA corta ({FAST_SPAN})', color='orange', lw=1)
ax.plot(slow_price.index, slow_price, label=f'EWMA larga ({SLOW_SPAN})', color='green', lw=1.2, alpha=0.9)

bull = events_cross[events_cross['side'] ==  1]
bear = events_cross[events_cross['side'] == -1]
ax.scatter(bull.index, close.loc[bull.index], marker='^', color='limegreen', s=30, zorder=5, label='Señal de compra')
ax.scatter(bear.index, close.loc[bear.index], marker='v', color='red',       s=30, zorder=5, label='Señal de venta')

# ax.set_title('EWMA Crossover Events on Log-Price (log scale)')
ax.set_ylabel('Price (USD, escala log)')
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
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
ax.plot(cum_strat.index, cum_strat,
        label=f'Estrategia (SR: {sharpe_strat:.2f}, CAGR: {ann_strat:.2%})',
        lw=1.2, color='darkorange')
ax.plot(cum_bh.index, cum_bh,
        label=f'Comprar y mantener (SR: {sharpe_bh:.2f}, CAGR: {ann_bh:.2%})',
        lw=1.0, color='steelblue', alpha=0.7)
ax.axhline(1.0, color='black', lw=0.6, ls=':')
# ax.set_title('Equity Curves (no transaction costs)')
ax.set_ylabel('Retorno acumulado')
ax.set_yscale('log')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# %%
# ── Strategy vs. Buy & Hold: relative performance ─────────────────────────
# Diferencia absoluta de las curvas de capital.  ``cum_strat - cum_bh`` mide
# el "exceso de wealth" generado por la estrategia sobre buy & hold; un valor
# positivo indica que la estrategia va por delante y un valor negativo, que
# queda rezagada respecto al índice.
relative_diff = cum_strat - cum_bh

# Diferencia logarítmica equivalente: ``log(strat) - log(bh)``.  Es
# aproximadamente la diferencia de retornos compuestos vista en escala
# aditiva, y resulta más natural cuando las curvas abarcan varios órdenes de
# magnitud.
relative_log = np.log(cum_strat) - np.log(cum_bh)

# Indicadores resumen del desempeño relativo.
final_diff   = float(relative_diff.iloc[-1])
mean_diff    = float(relative_diff.mean())
max_diff     = float(relative_diff.max())
min_diff     = float(relative_diff.min())
positive_pct = float((relative_diff > 0).mean())

print("Strategy − Buy & Hold (wealth difference)")
print("-" * 56)
print(f"  Final excess wealth  : {final_diff:+.2f}")
print(f"  Mean excess wealth   : {mean_diff:+.2f}")
print(f"  Max excess wealth    : {max_diff:+.2f}")
print(f"  Min excess wealth    : {min_diff:+.2f}")
print(f"  % of bars ahead      : {positive_pct:.2%}")

# ── Plot: relative equity difference ──────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                         gridspec_kw={'height_ratios': [2, 1]},
                         sharex=True)

ax = axes[0]
ax.plot(relative_diff.index, relative_diff, color='darkorange', lw=1.0,
        label='Strategy − Buy & Hold')
ax.axhline(0.0, color='black', lw=0.6, ls=':')
ax.fill_between(relative_diff.index, relative_diff, 0.0,
                where=relative_diff >= 0, color='limegreen', alpha=0.25,
                label='Strategy ahead')
ax.fill_between(relative_diff.index, relative_diff, 0.0,
                where=relative_diff < 0, color='red', alpha=0.25,
                label='Buy & Hold ahead')
ax.set_title('Strategy vs. Buy & Hold — Wealth Difference')
ax.set_ylabel('Cum. wealth difference')
ax.legend(loc='best', fontsize=8)

ax2 = axes[1]
ax2.plot(relative_log.index, relative_log, color='steelblue', lw=1.0,
         label='log(Strategy)  − log(Buy & Hold)')
ax2.axhline(0.0, color='black', lw=0.6, ls=':')
ax2.fill_between(relative_log.index, relative_log, 0.0,
                 where=relative_log >= 0, color='limegreen', alpha=0.25)
ax2.fill_between(relative_log.index, relative_log, 0.0,
                 where=relative_log < 0, color='red', alpha=0.25)
ax2.set_title('Strategy vs. Buy & Hold — Logarithmic Difference')
ax2.set_ylabel('log(wealth ratio)')
ax2.set_xlabel('Date')
ax2.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.show()


# %% [markd

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
