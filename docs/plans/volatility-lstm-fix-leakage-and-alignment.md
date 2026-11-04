# Plan — Corregir hallazgos Critical/High del LSTM de volatilidad

## Alcance
Corrige los 4 hallazgos Critical/High de la revisión de
`notebooks/volatility/Volatilidad_LSTM.py`, sin tocar arquitectura, features,
horizonte ni pérdida LINEX. Quedan fuera los hallazgos Medium/Low (costos,
DM/HAC, momentum warm-up, determinismo CUDA, `drop_last`) — se documentan como
seguimiento.

## No-objetivos
- No se re-entrena ni se validan métricas finales en este plan (solo se
  implementan los cambios y sus tests).
- No se cambia la semántica de `forecast_horizon` (el target sigue cubriendo
  `r[t+1..t+H]`); solo se alinea el backtest.
- No se edita el texto de la tesis ni se crean figuras.

## Fix 1 (Critical) — Fuga de labels a través de las fronteras de split
**Problema:** `compute_log_vol_target` (`volatility.py:230`) calcula el target
forward **antes** de `temporal_split`, así que las últimas `H` filas de train
usan retornos de validación, y las últimas `H` de validación (early stopping,
`volatility_models.py:228`) usan retornos de **test**.

**Cambios:**
- `src/thesis/volatility.py`: añadir `purge_boundary_labels(df, horizon)` que
  elimina las últimas `horizon` filas (labels cuyo ventana cruza la frontera).
- `notebooks/volatility/Volatilidad_LSTM.py` `prepare_data`: tras
  `temporal_split`, aplicar
  `train_df = purge_boundary_labels(train_df, cfg["forecast_horizon"])` y lo
  mismo para `val_df` (antes de construir `X_tr`/`y_tr`/`scale_splits`).

## Fix 2 (Critical) — Aritmética de cartera simple vs log
**Problema:** `run_vol_timing_backtest` (`volatility_evaluation.py:225-227`)
hace `pos * log_ret - tc`, mezclando log y costos; el notebook grafica
`cumprod(1+ret)` mientras `summarize_log_returns` usa `exp(cumsum)`.

**Cambios en `src/thesis/volatility_evaluation.py`:**
- `run_vol_timing_backtest`: `r_simple = np.expm1(ret_arr)`; calcular
  `ret_vol/ret_base/ret_bh` como retornos **simples** netos de costos.
- Renombrar `summarize_log_returns` → `summarize_returns` (retornos simples):
  - `sharpe_ratio(r_simple)`: `sqrt(252)*mean/std` sobre simples (rf=0).
  - `max_drawdown(r_simple)`: equity `cumprod(1+r)` **anteponiendo 1.0**.
  - `calmar_ratio(r_simple)`: `CAGR/|MDD|` con `CAGR=prod(1+r)^(252/n)-1`.
  - `total_return = prod(1+r)-1`.
- Mantener compatibilidad de salida de `run_vol_timing_backtest` (mismas claves).

**Cambios en el notebook:** `asset_stats = summarize_returns(np.expm1(ret_te_aligned))`
y el plot de backtest usa `np.cumprod(1 + bt["ret_*"])` (ya correcto al ser
simples).

## Fix 3 (High) — Desalineación forecast ↔ retorno operado
**Problema:** el forecast indexado en `j` se origina al cierre de
`dates[j-1]` y predice `r[j+1..j+H]`, pero `ret_te_aligned[j] = te_log_ret[LB+j]`
(`Volatilidad_LSTM.py:403`) opera el retorno `r[j]`, que queda **fuera** de la
ventana predicha (un día de desfase; causal, pero no predictivo).

**Cambios en `train_and_infer` / `smooth_predictions`:**
- `ret_te_aligned = te_log_ret[LB+1:]` (longitud `n_eval-1`) y nuevo
  `bt_dates = te_dates[LB+1:]`; mantener `dates = te_dates[LB:LB+n_eval]` para
  plots/métricas (alineado con `y_true/y_lstm/y_har`).
- Persistir `bt_dates` en `inference.pkl` y exponerlo en `smooth_predictions`.
- `run_backtest_and_strategies`: usar `y_lstm[:-1]` (descartar último forecast
  no operable) junto con `ret_te_aligned` (longitud `n_eval-1`).
- Plot de backtest y `summarize_returns` usan `bt_dates`/`ret_te_aligned`.

## Fix 4 (High) — No suavizar la verdad (evaluación)
**Problema:** `smooth_predictions` (`Volatilidad_LSTM.py:475-477`) suaviza
también `y_true_log`, cambiando el target que se evalúa y creando warm-up en
test.

**Cambios en `smooth_predictions`:**
- `y_true = np.exp(art["y_true_log"])` **sin EWMA**; suavizar solo los forecasts
  `y_lstm`/`y_har` (opcional, solo para trading).
- Devolver además `y_true_raw`/`y_lstm_raw`/`y_har_raw` para que `compute_metrics`
  informe métricas primarias raw-vs-raw.

**Cambios en `compute_metrics`:** calcular métricas primarias con
`y_true` vs `y_lstm_raw`/`y_har_raw`; documentar que las métricas suavizadas
son una vista secundaria para trading (no el estimando declarado).

## Archivos a tocar
- `src/thesis/volatility.py` — `purge_boundary_labels` (nuevo).
- `src/thesis/volatility_evaluation.py` — `run_vol_timing_backtest`,
  `summarize_returns` (renombrado), `sharpe_ratio`, `max_drawdown`,
  `calmar_ratio`.
- `notebooks/volatility/Volatilidad_LSTM.py` — `prepare_data`,
  `train_and_infer`, `smooth_predictions`, `compute_metrics`,
  `run_backtest_and_strategies`, celdas de plot (backtest).
- `tests/test_volatility.py` — `test_purge_boundary_labels`.
- `tests/test_volatility_evaluation.py` — actualizar tests de `summarize_*`,
  `sharpe`, `max_drawdown`, `calmar`, `run_vol_timing_backtest`.
- `tests/test_volatility_integration.py` — aserción de no-dependencia de labels
  a través de fronteras; ajustar aritmética en `test_real_backtest_and_metrics`.

## Criterios de aceptación
- Ningún label de train usa retornos de val, ni ningún label de val usa
  retornos de test (verificable con un test que compare contra un split sin
  purga).
- `run_vol_timing_backtest` devuelve retornos simples netos; `sharpe`, `maxdd`
  y `calmar` calculados con fórmula correcta (drawdown incluye wealth inicial;
  Calmar = CAGR/|MDD|).
- Backtest alineado: `len(ret_te_aligned) == len(y_lstm[:-1]) == len(bt_dates)`;
  `bt_dates[j]` es el día en que se realiza el retorno operado por
  `y_lstm[j]`.
- `compute_metrics` reporta métricas raw-vs-raw como primarias; `y_true` no
  recibe EWMA.
- `uv run pytest` (sin integración) y `uv run pytest -m integration` pasan;
  `uv run ruff check src tests` sin errores nuevos.
- Importar el notebook no descarga datos ni entrena (test existente intacto).

## Riesgos
- **Cambio de longitud (`n_eval-1`)** en el backtest: asegurar que todos los
  consumidores de `ret_te_aligned`/`dates` usen la longitud correcta; los plots
  de métricas siguen con `dates` (n_eval), el plot de backtest con `bt_dates`
  (n_eval-1).
- **Purga insuficiente si hay huecos de calendario:** `purge_boundary_labels`
  asume días hábiles contiguos; para robustez usar la condición
  `t + horizon` sobre el índice en lugar de un `iloc[:-horizon]` si el índice no
  es contiguo (los datos OHLCV validados son diarios, riesgo bajo).
- **Semántica de `summarize_*`:** renombrar rompe importadores externos; grepear
  usos de `summarize_log_returns` en todo el repo (notebook, tests, otros
  scripts) antes de renombrar.
- **Interacción con `smooth-target-before-lstm-training.md`:** Fix 4 (evaluar
  raw) es compatible con suavizar el target antes de entrenar, pero ambos
  planes tocan `smooth_predictions`; coordinar para no duplicar el EWMA del
  target.
- **Reproducibilidad:** no re-entrenar como parte de este plan; los tests
  sintéticos deben cubrir la aritmética sin red.
