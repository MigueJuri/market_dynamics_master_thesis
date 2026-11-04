# Plan: vol-timing con señal por cruce de medias móviles (EWMA)

> **Estado: revertido.**  Este cambio fue revertido a petición del usuario: la
> señal base volvió a ser la de **momentum** (signo de la media móvil de los
> retornos, calculada dentro de `run_vol_timing_backtest`).  El documento se
> conserva como registro histórico; no describe el estado actual del código.

## Objetivo

Sustituir la señal base de momentum de `run_vol_timing_backtest`
(`src/thesis/volatility_evaluation.py`) por una señal de **cruce de medias
móviles exponenciales (EWMA)** idéntica a la de
`Trend_Following_Strategy_v2.py`, manteniendo el escalado por volatilidad
predicha: si `σ̂` es alta, se reduce la posición.

- Señal base: **long/flat (1/0)** — largo tras cruce alcista, flat tras
  cruce bajista (sin cortos), igual que v2. Ejecución causal (`shift(1)`).
- Vol-timing (sin cambios de lógica): `pos_vol = clip(signal / (σ̂/σ̄), -1, 1)`,
  `pos_base = clip(signal, -1, 1)`.

## Alcance

No se modifica el entrenamiento LSTM/HAR-RV ni las métricas de pronóstico;
solo cambia el backtest (Etapa 5 del cuaderno) y la señal que lo alimenta.

## Cambios

### 1. `src/thesis/event_labeling.py` — nueva función general
`ewma_crossover_position(log_close, fast_span=40, slow_span=160, burn_in=None) -> pd.Series`
- Reutiliza `get_ewma_crossover_events`.
- Reproduce la lógica de v2 (líneas 151-163): `signal_series` reindexado,
  `marker = replace(-1, 0)`, `position = marker.where(signal!=0).ffill().fillna(0)`,
  `entry_pos = position.shift(1).fillna(0)`.
- Devuelve `entry_pos` (serie causal 1/0 alineada al índice de `log_close`).
- Exportar en `src/thesis/__init__.py` (`__all__` + import).

### 2. `src/thesis/volatility_evaluation.py` — `run_vol_timing_backtest`
- Nueva firma: `run_vol_timing_backtest(ret, pred_vol, sigma_bar, signal, cost_bps=0.0)`.
- Eliminar `mom_win` y el cálculo interno de momentum (líneas 249-254).
- `signal` (array 1/0, longitud `len(ret)`) es la posición causal ya alineada.
- Validar `len(signal) == len(ret)` (raise `ValueError`).
- Resto igual: `vol_scale`, `pos_vol/pos_base/pos_bh`, costos y `summarize_returns`
  (mismas claves de salida, sin cambios).

### 3. `notebooks/volatility/Volatilidad_LSTM.py`
- `CONFIG`: sustituir `"mom_win"` por `"fast_span": 40`, `"slow_span": 160`,
  `"burn_in": 160`.
- `train_and_infer`: persistir `log_close = np.log(raw["Close"])` dentro de
  `inference.pkl` (junto a `dates`, `bt_dates`, etc.).
- `smooth_predictions`: re-exponer `log_close` en el dict devuelto.
- `run_backtest_and_strategies`:
  - `signal_full = ewma_crossover_position(log_close, fast_span, slow_span, burn_in)`.
  - `signal = signal_full.reindex(bt_dates).fillna(0.0).to_numpy()` (longitud
    `n_eval-1`, alineada a `ret_te_aligned` y `y_lstm[:-1]`).
  - Llamar `run_vol_timing_backtest(ret_te_aligned, y_lstm[:-1], sigma_bar, signal, cost_bps)`.
- Actualizar imports (`ewma_crossover_position` desde `thesis.event_labeling`).

### 4. Tests
- `tests/test_volatility_evaluation.py`:
  - `test_run_vol_timing_backtest`: pasar `signal=np.ones(200)` (o señal sintética).
  - `test_run_vol_timing_backtest_returns_simple_returns`: con vol constante,
    `pos = signal`; verificar `ret_vol = signal*expm1(ret)`, `ret_base = signal*expm1(ret)`.
  - `test_run_vol_timing_backtest_costs_are_subtracted_from_simple_returns`: pasar `signal`.
  - Añadir test de `ewma_crossover_position` (causalidad: `shift(1)`, valores 0/1).
- `tests/test_volatility_integration.py` (línea ~336): construir `signal` desde
  `real_ohlcv["Close"]` vía `ewma_crossover_position` y pasarlo a
  `run_vol_timing_backtest`.

### 5. Documentación (secundario)
- `notebooks/volatility/README.md` y `docs/volatilidad-lstm-training-process.md`:
  actualizar firma/semántica de `run_vol_timing_backtest` y `mom_win` → `fast_span/slow_span`.

## Criterios de aceptación

- `uv run pytest` (sin integración) en verde, incl. tests actualizados y nuevo test de la señal.
- `uv run pytest -m integration -k real_backtest` en verde.
- `uv run ruff check src tests` sin errores.
- `run_vol_timing_backtest` ya no importa ni calcula momentum; usa la señal EWMA.
- El backtest del cuaderno reporta estrategia "Baseline" = cruce EWMA sin escalar,
  y "Vol-Timed" = misma señal escalada por `σ̂/σ̄` (posición menor si `σ̂` alta).
- Sin look-ahead: la señal usa precios ≤ `t-1` (`shift(1)`), `σ̄` sigue siendo
  mediana del train.

## Riesgos

- **Alineación señal↔retornos**: `signal` debe corresponder al retorno de
  `bt_dates[j]` usando info hasta `dates[j]` (= día previo). Se valida con
  `len(signal)==len(ret)` y reindexando `entry_pos` a `bt_dates`.
- **Causalidad / burn-in**: el cruce EWMA requiere historia previa (160 barras);
  al reindexar a `bt_dates` (test, años después de 1995) el transitorio ya pasó.
- **Cambio de semántica de `ret_base`**: pasa de long/short (±1) a long/flat (0/1);
  los Sharpe base no son directamente comparables con la versión anterior.
- **Compatibilidad de llamadas**: cualquier otro llamador (solo tests y el
  cuaderno, según grep) debe actualizar la firma.
