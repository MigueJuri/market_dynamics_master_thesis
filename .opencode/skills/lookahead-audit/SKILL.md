---
name: lookahead-audit
description: Checklist to detect look-ahead leakage in features, labels, normalization, and backtests.
---

# Look-ahead leakage audit

Use this checklist when reviewing any pipeline that learns from time-series data.

1. **Features**
   - Signals must use information available at or before `t-1` unless explicitly stated.
   - Rolling windows use `min_periods` and `shift(1)`; no future-centered windows.
   - No target-derived features unless computed from past targets only.

2. **Normalization / scaling**
   - Fit scalers on the training split only; transform validation/test with the same params.
   - No full-sample standardization before the train/test split.

3. **Labels / returns**
   - Forward returns use `t+1` only; labels do not include contemporaneous target info.
   - Event labels are based on future thresholds only when the experiment explicitly allows it.

4. **Train-test split**
   - Splits are time-ordered (no random shuffle on time index).
   - Group-based CV respects dates, not random rows.

5. **Backtest / equity curve**
   - Position at time `t` is set from signals known at `t-1`.
   - Costs and slippage are applied on the same bar as the trade.
   - No compounding of future returns into past decisions.

6. **Diagnostics**
   - Corrupt the end of the series; verify earlier predictions/features are unchanged.
   - Report which items passed/failed with line references.
