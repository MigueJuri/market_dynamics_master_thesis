# Plan: suavizar el target antes del entrenamiento LSTM

## Alcance

Invertir el orden del suavizado en `notebooks/volatility/Volatilidad_LSTM.py`.
Hoy se entrena la LSTM sobre el target crudo y, tras predecir, se suaviza
`y_true`/`y_lstm`/`y_har` con EWMA (`span = smoothing_window`). El cambio:
suavizar **primero** el target (en espacio log-vol) y entrenar la LSTM y el
baseline HAR-RV con ese target suavizado; eliminar el suavizado post-hoc.

## No-objetivos

- No se modifican features, arquitectura, horizonte, pérdida LINEX ni el
  backtest en sí (solo el target que alimenta a los modelos).
- No se toca `src/thesis/` (las utilidades ya sirven; solo cambia la orquestación
  del notebook). Si se detecta necesidad de refactor, se anota como riesgo.
- No se cambia la semántica de `smoothing_window` (sigue siendo el `span` EWMA).

## Archivos y funciones

- **Tocar**: `notebooks/volatility/Volatilidad_LSTM.py`, función
  `run_experiment` (y el comentario/docstring de la celda de configuración).
  - Leer `src/thesis/volatility.py` (`compute_log_vol_target`,
    `build_volatility_dataset`, `temporal_split`, `scale_splits`) y
    `src/thesis/volatility_models.py` (`HARRVModel.fit/predict`) para confirmar
    espacios (log-vol) — ya verificados.

## Cambios concretos

1. **Suavizar el target una vez, en espacio log-vol, antes de dividir.**
   Tras `data = build_volatility_dataset(...)` (≈ línea 183), antes de
   `temporal_split`, aplicar:
   `data["target_log_vol"] = data["target_log_vol"].ewm(span=cfg["smoothing_window"], adjust=False).mean()`
   - EWMA es causal (solo usa pasado), por lo que **no introduce look-ahead** y
     mantiene continuidad en los bordes de las particiones (mejor que suavizar
     cada split por separado, que provocaría arranques en frío en val/test).
   - Se suaviza en log-vol, idéntico al espacio donde hoy se aplica el EWMA
     post-predicción.

2. **Sin más cambios aguas abajo del split.** `y_tr/y_vl/y_te` quedan ya
   suavizados, por lo que:
   - `scale_splits` ajusta el `y_scaler` sobre el target suavizado (correcto).
   - LSTM entrena con target suavizado.
   - `har.fit(har_X_tr, y_tr_raw_aligned)` usa automáticamente el target
     suavizado (`y_tr_raw_aligned = y_tr[LB:...]`), manteniendo comparación justa.
   - `sigma_bar` (línea 312) pasa a ser la mediana de la vol suavizada de train,
     coherente con la estrategia que ahora targetea vol suavizada.

3. **Eliminar el bloque EWMA post-predicción (líneas 274-278)** y la variable
   `span`. Dejar:
   ```python
   y_true_log = y_scaler.inverse_transform(y_te_s[LB:LB+n_eval].reshape(-1,1)).ravel()
   y_lstm_log = y_scaler.inverse_transform(y_pred_lstm[:n_eval].reshape(-1,1)).flatten()
   y_har_log = y_pred_har_te[:n_eval]
   ```
   (HAR predice en log-vol crudo, no requiere inverse_transform.)

4. **Actualizar comentario** en CONFIG/celda inicial: documentar que
   `smoothing_window` ahora se aplica al target antes del entrenamiento.

## Criterios de aceptación

- El EWMA se aplica a `target_log_vol` **antes** de `temporal_split` y del
  `scale_splits`; no queda ningún `.ewm(...)` en la sección de evaluación.
- `har.fit` y `sigma_bar` consumen el target suavizado (verificable con un
  `print`/assert local si se desea).
- `run_experiment()` ejecuta sin errores y produce `metrics.csv`, `report.md`,
  escaladores y artefactos como antes.
- Métricas/fechas de alineación intactas: `n_eval`, `dates` en las celdas de
  figuras no cambian de longitud.

## Riesgos

- **Leakage (bajo)**: EWMA `adjust=False` es causal. Riesgo solo si alguien
  suaviza con `center=True` o suaviza *después* de invertir el escalado; se debe
  mantener el suavizado en log-vol sobre la serie completa antes del split.
- **Unidades**: suavizar en log-vol (no en sigma anualizado). Confirmar que el
  EWMA se aplica a la columna log-vol y que `np.exp(...)` se conserva en
  evaluación para volver a sigma anualizado.
- **Comparación HAR-LSTM**: si no se re-entrena HAR sobre el target suavizado, la
  comparación quedaría sesgada; con este diseño HAR y LSTM comparten el mismo
  target suavizado.
- **Bordes de secuencia**: `make_sequences` descarta `lookback` filas; al suavizar
  la serie completa antes de dividir, train/val/test heredan un suavizado
  consistente (evitar suavizar split por split para no romper la continuidad).
- **Semántica de `smoothing_window`**: ahora modifica la señal objetivo (menos
  ruido, potencialmente más fácil de aprender); esperar métricas diferentes a las
  actuales. Documentar en el reporte generado si se comenta.
