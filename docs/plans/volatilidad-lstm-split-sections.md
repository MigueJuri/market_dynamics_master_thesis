# Dividir `Volatilidad_LSTM.py` en secciones re-ejecutables

## Objetivo y alcance

Refactorizar el notebook `py:percent` `notebooks/volatility/Volatilidad_LSTM.py`
(actualmente toda la lógica vive en `run_experiment`) en etapas independientes:
**preparación de datos → entrenamiento → post-proceso (suavizado) → métricas →
backtest/estrategias**. El caso de uso clave: cambiar `smoothing_window` (o
`cost_bps`, `linex_b` en backtest) y re-ejecutar solo la celda correspondiente,
**sin reentrenar** ni re-descargar datos.

**No-objetivos**: no tocar `src/thesis/*` (las funciones reutilizables ya están
bien separadas); no añadir nuevas estrategias de trading; no cambiar la
matemática (métricas, EWMA, vol-timing) ni los valores por defecto de `CONFIG`.
No sincronizar con el repo de la tesis.

## Diseño

El problema raíz: `run_experiment` aplica el EWMA (`smoothing_window`) *dentro*
de la misma función que entrena, y devuelve series ya suavizadas, de modo que
cambiar el span obliga a repetir todo el pipeline. La solución es **persistir
las predicciones en espacio log y sin suavizar**, y separar el suavizado en su
propia celda que solo lee/escribe de disco.

Estado compartido entre celdas: un dict global por etapa (`prepared`,
`trained`, `smoothed`) más un artefacto en disco `inference.pkl` con lo mínimo
para re-ejecutar aguas abajo sin entrenar.

### Nuevos artefactos en disco
- `inference.pkl` (post-entrenamiento): `dates` (DatetimeIndex alineado),
  `y_true_log`, `y_lstm_log`, `y_har_log` (crudos, espacio log, longitud
  `n_eval`), `sigma_bar`, `ret_te_aligned`, `history`.
- Mantener: `lstm_vol_model.pt`, `scaler.pkl`, `target_scaler.pkl`,
  `model_config.json`, `metrics.csv`, `report.md`.

## Archivo y funciones a tocar

Solo `notebooks/volatility/Volatilidad_LSTM.py`. Reorganizar en celdas:

1. **Imports** (sin cambios) + celda **CONFIG** (sin cambios de valores).
2. `prepare_data(config) -> dict` — descarga → `build_volatility_dataset` →
   `temporal_split` → `scale_splits` → `make_sequences` → ajuste HAR-RV.
   Devuelve `raw, data, train/val/test_df, X/Y splits, scalers, secuencias,
   DataLoader, tensores de val/test, har_X_tr/te, y_tr_raw_aligned`.
3. `train_and_infer(config, prepared) -> dict` — construye `VolatilityLSTM`,
   `train_lstm`, carga best checkpoint, predice en test (`X_te_t`), calcula
   `n_eval`, `y_true_log`/`y_lstm_log`/`y_har_log` (log, sin suavizar),
   `sigma_bar`, `ret_te_aligned`, `dates`. **Guarda `inference.pkl`**, modelo y
   escaladores. Devuelve `model, history, har`.
4. `smooth_predictions(config) -> dict` — carga `inference.pkl`, aplica
   `ewm(span=config["smoothing_window"], adjust=False)` a las tres series log,
   `exp()` → espacio sigma. Devuelve `y_true, y_lstm, y_har, sigma_bar,
   ret_te_aligned, dates`. Re-ejecutable sin entrenar.
5. `compute_metrics(config, smoothed) -> dict` — `evaluate_forecasts` (LSTM y
   HAR), `diebold_mariano_test`, escribe `metrics.csv`, imprime checklist.
6. `run_backtest_and_strategies(config, smoothed) -> dict` —
   `run_vol_timing_backtest` + `summarize_log_returns` + `summary` DataFrame +
   prints de Sharpe. Re-ejecutable sin entrenar ni re-metricar.
7. **Celdas de plot** (existentes): leer de `smoothed` en lugar de `results`;
   mantienen "una figura por celda".

## Criterios de aceptación

- Ejecutar celdas 1→3 entrena una sola vez y escribe `inference.pkl`.
- Cambiar `CONFIG["smoothing_window"]` y re-ejecutar **solo** las celdas 4→6
  (y los plots) actualiza métricas/figuras **sin** reescribir
  `lstm_vol_model.pt` ni descargar datos (verificable por timestamps/logs).
- Cambiar `cost_bps` y re-ejecutar solo la celda 6 actualiza el backtest sin
  re-entrenar ni re-metricar.
- Cada etapa es importable sin efectos secundarios (las celdas de ejecución
  siguen bajo `if __name__ == "__main__":`, igual que hoy).
- Los plots (loss, predicho-vs-real, series, adaptabilidad, backtest, ranks,
  abril 2025) siguen generando los mismos PNG.
- `uv run pytest` (sin integración) y `uv run ruff check` sin errores nuevos.

## Riesgos clave

- **Look-ahead**: el suavizado EWMA es causal (`ewm(...).mean()` en el tiempo)
  pero se debe mantener aplicado *solo sobre el período de test*, nunca sobre
  train/val; `sigma_bar` debe seguir derivándose **solo del train** (ya lo
  hace, no romperlo al mover código). No usar `smoothing_window` para entrenar.
- **Alineación de longitudes**: `n_eval = min(len(yte_seq), len(y_pred_lstm),
  len(y_pred_har_te))` y los slices `[LB : LB + n_eval]` son frágiles. Al
  persistir `dates`/`ret_te_aligned`, conservar exactamente el mismo `n_eval`
  que al entrenar para que plots y backtest no se desalineen.
- **Unidades**: persistir `*_log` (espacio log) sin suavizar; `exp()` solo en
  `smooth_predictions`. `sigma_bar` y `ret_te_aligned` se usan en espacio
  sigma; documentar unidades en docstrings de cada celda.
- **Semilla**: `set_volatility_seed(SEED)` debe seguir ejecutándose en la etapa
  de entrenamiento (no en post-proceso) para reproducibilidad.
- **Rutas relativas**: los `*_path` de `CONFIG` se escriben en el CWD; al
  dividir, no cambiar la semántica de rutas ni el orden de escritura.
