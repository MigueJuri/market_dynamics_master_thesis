---
name: backtest-report
description: Standard backtest metrics, cost assumptions, out-of-sample reporting, and plot conventions.
---

# Backtest reporting conventions

## Required metrics
- Annualized return and volatility (252 trading days).
- Sharpe ratio (risk-free rate = 0 unless specified).
- Maximum drawdown and Calmar ratio.
- Turnover (percent of portfolio traded per period).
- Win rate and profit factor if applicable.

## Cost assumptions
- Default: 5 bps per trade (round-trip 10 bps) unless the plan says otherwise.
- Document borrow cost, slippage, and dividend reinvestment assumptions.
- Report net and gross results separately.

## Out-of-sample rules
- Strategy parameters fitted only on the train window.
- First forecast and position in each test window use data up to the previous bar.
- Walk-forward or expanding-window evaluation; no peeking at the full series.

## Plot conventions
- Equity curves start at 1.0 and are log-scaled when comparing many curves.
- Drawdown shaded under the curve.
- Separate in-sample and out-of-sample regions with a vertical line.
- Use stable filenames: `<strategy>_<dataset>_<metric>.pdf/png`.

## Output discipline
- Each saved figure/table is produced by a single script or notebook cell.
- Name files descriptively and do not create export/sync folders.
