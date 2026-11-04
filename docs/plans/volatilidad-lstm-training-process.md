# Plan — Resumen del proceso de entrenamiento (LSTM de volatilidad)

## Objetivo
Producir un documento Markdown que describa la **secuencia completa** del
pipeline de `notebooks/volatility/Volatilidad_LSTM.py`, **excluyendo** la
arquitectura de la red (el usuario ya la conoce). Enfoque en: features,
target, estandarización, split, secuencias, entrenamiento, post-procesado,
evaluación y backtest.

## No-goals
- No documentar la arquitectura de `VolatilityLSTM` (capas, inicialización,
  `forward`, residual, LayerNorm) ni de `ExponentialAsymmetricLoss` (solo
  mencionar que la pérdida es LINEX asimétrica con `b=linex_b`).
- No modificar código fuente, notebooks ni tests.
- No editar el texto de la tesis ni crear figuras.
- No reproducir/entrenar el modelo; es un resumen estático a partir del código.

## Archivos
**Leer (ya inspeccionados):**
- `notebooks/volatility/Volatilidad_LSTM.py` (orquestador, `run_experiment`).
- `src/thesis/volatility.py` (features, target, split, secuencias, escalado).
- `src/thesis/volatility_models.py` (HAR-RV, `train_lstm`, LINEX).
- `src/thesis/volatility_evaluation.py` (métricas, DM, backtest).
- `src/thesis/config.py` (solo constantes `SEED`, `TRADING_DAYS`).

**Escribir (única mutación):** `docs/volatilidad-lstm-training-process.md`.
(Ruta propuesta; alternativa: `docs/ai/volatilidad-lstm-training-process.md`.)

**No tocar:** `src/`, `notebooks/`, `tests/`, `data/`, `outputs/`.

## Contenido del resumen (secuencia exacta a documentar)
Referenciar cada paso a la función que lo implementa:

1. **Configuración** — `CONFIG`: ticker `SPY`, `start_date`, `train_end`
   `2017-01-01`, `val_end` `2022-01-01`, `lookback=5`, `forecast_horizon=5`,
   `smoothing_window=15`, `linex_b=0.8`, lista de 13 `features`, hiperparámetros
   de entrenamiento, `cost_bps=5`, rutas de salida.
2. **Datos** — `download_ohlcv` (yfinance, auto_adjust) → `validate_ohlcv`
   (índice Datetime, sort, ffill, reconstrucción OHLC para Parkinson/GK).
   `count_extreme_moves` (>15%, retenidos).
3. **Features** — `compute_volatility_features`: HAR-RV (`rv_daily/weekly/monthly`,
   ventanas 1/5/22 sobre `log_ret²`), `vol_ratio`, Parkinson (`log(H/L)²`,
   corrección `4·ln2`), Garman-Klass, RSI-14 a [0,1], `rel_volume`,
   `volume_trend`, `intraday_range`, `gap`, `skew_5`, `mean_ret_5`. **Compresión
   logarítmica** de las variables estrictamente positivas (`log(clip(1e-12))`).
   Nota: `vol_20` se calcula pero **no** está en `features` de CONFIG.
4. **Target** — `compute_log_vol_target`: log-vol realizada *forward*
   `log( sqrt( (252/H) · Σ_{i=1..H} r²_{t+i} ) )` con `shift(-H)`. Señalar que
   el target es la etiqueta (usa futuro por diseño, no es leak en features).
5. **Dataset** — `build_volatility_dataset` = features + target, `dropna()`.
6. **Split temporal** — `temporal_split` por fecha (train ≤2017, val ≤2022,
   test >2022), particiones disjuntas y ordenadas.
7. **Estandarización** — `scale_splits`: `StandardScaler` separado para X e y;
   **fit solo en train**, `transform` en val/test. y se escala en espacio
   log-vol.
8. **Secuencias** — `make_sequences`: ventana `[i, i+lookback)`, target
   `y[i+lookback]`; formas `(T-LB, LB, n_features)`. `DataLoader` train con
   `shuffle=True, drop_last=True`.
9. **Baseline HAR-RV** — `align_har_features_with_sequences` (alinea último día
   de la ventana) + `HARRVModel.fit` (OLS sobre `rv_daily/weekly/monthly` con
   HAC Newey-West `maxlags=10`), predice en log-vol crudo.
10. **Entrenamiento** — `train_lstm`: `AdamW(lr=5e-5, weight_decay=1e-3)`,
    `ReduceLROnPlateau` (patience 5, factor 0.5, min_lr 1e-6), pérdida LINEX
    `b=0.8`, `clip_grad_norm_=1.0`, early stopping (`patience=15`,
    `min_epochs=50`, `max_epochs=1000`), checkpoint del mejor val_loss.
    Ratio `n_secuencias/n_params` como chequeo de sobredeterminación.
11. **Post-procesado de predicciones** — `y_scaler.inverse_transform` en
    log-vol → **EWMA** (`smoothing_window=15`, `adjust=False`) sobre real/LSTM/
    HAR → `np.exp` a espacio σ (vol anualizada).
12. **Evaluación** — `evaluate_forecasts` (R², RMSE, MAE, Spearman, Dir_Acc,
    QLIKE, Mincer-Zarnowitz F) + `diebold_mariano_test` (h=5, HAC).
13. **Backtest vol-timing** — `sigma_bar` = mediana de `exp(y_train)` (solo
    train, anti-lookahead); `run_vol_timing_backtest` (momentum 63d con
    `shift(1)`, posición `clip(sign/(σ̂/σ̄), ±1)`, costos 5 bps).

## Criterios de aceptación
- El `.md` cubre los 13 pasos en orden, con nombres de función y valores de
  configuración exactos (fechas, lookback, horizonte, linex_b, hiperparámetros).
- Explica **qué** se estandariza (X e y por separado, `StandardScaler`,
  fit solo en train) y **cuándo** (antes de `make_sequences`).
- Describe la compresión logarítmica de features y la conversión final
  log-vol → σ anualizada (incluye factor 252).
- **No** contiene detalle de arquitectura de la LSTM (capas, init, residual).
- No se edita ningún archivo de código; el único artefacto es el `.md`.
- Lenguaje en español, coherente con el resto del repo.

## Riesgos
- **Leakage**: el mayor riesgo es describir el orden como si el escalador se
  ajustara sobre todo el dataset. Documentar explícitamente `fit` solo en
  train. `compute_log_vol_target` usa datos futuros por diseño (etiqueta), no
  confundir con leak de features. EWMA se aplica a la serie de test completa
  (suavizado de evaluación, no de entrenamiento) — señalarlo.
- **Unidades**: mezcla de espacios (features log, target log-vol anualizada,
  salida final σ anualizada). El resumen debe dejar claro el espacio en cada
  etapa para evitar malinterpretar métricas (RMSE en % de vol anualizada).
- **Numérico**: constantes de clip/epsilon (`1e-12`, `1e-10`, `1e-8`) para
  logs y divisiones; `grad_clip=1.0` y su alerta >5% de batches.
- **Precisión factual**: verificar contra el código (no de memoria) las
  ventanas HAR (1/5/22), anualización 252 y `drop_last=True`.
