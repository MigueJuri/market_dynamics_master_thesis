# Predicción de Volatilidad con LSTM

Este módulo entrena una red LSTM parsimoniosa para predecir la
volatilidad realizada forward del ETF ``SPY`` y la compara con el
benchmark HAR-RV (Corsi, 2009).

La pregunta central es: **¿una LSTM con features de microestructura y
estimadores de rango mejora la predicción de volatilidad respecto al
modelo HAR-RV canónico?**

---

## Estructura

```text
notebooks/volatility/
├── README.md              ← este archivo
└── Volatilidad_LSTM.py    ← cuaderno Jupytext del experimento

src/thesis/
├── volatility.py          ← features, target, splits y secuencias
├── volatility_models.py   ← HAR-RV, LSTM, pérdida LINEX y entrenamiento
└── volatility_evaluation.py ← métricas, backtest y gráficos
```

---

## Pipeline

1. **Datos**: descarga OHLCV de ``SPY`` desde 1995 con ``yfinance``.
2. **Features causales**: componentes HAR-RV, Parkinson, Garman-Klass,
   RSI, volumen relativo, rango intradía, gap y estadísticos de retornos.
3. **Target**: log-volatilidad realizada anualizada a ``H`` días.
4. **División temporal**: train (hasta 2016-01-01), validation
   (2016-2022) y test (desde 2022-01-01).
5. **Estandarización**: ``StandardScaler`` por partición para la LSTM;
   HAR-RV opera en espacio crudo.
6. **Modelos**:
   - **HAR-RV** ajustado por MCO con errores HAC (Newey-West).
   - **LSTM** con Pre-LayerNorm, conexión residual, dropout y cabeza densa.
7. **Entrenamiento**: AdamW, weight decay, scheduler ReduceLROnPlateau,
   gradient clipping y early stopping.
8. **Evaluación**: R², RMSE, MAE, Spearman, precisión direccional,
   QLIKE, Mincer-Zarnowitz y test de Diebold-Mariano.
9. **Backtest**: estrategia de vol-timing que escala una señal de
   momentum por la volatilidad predicha inversa.
10. **Diagnóstico**: figura de 6 paneles con pérdidas, predicciones,
    adaptabilidad post-shock, rangos y equity.

---

## Funciones generales (`src/thesis`)

### `thesis.volatility`

* ``download_ohlcv(ticker, start, ...) -> pd.DataFrame``
  Descarga y valida datos OHLCV desde Yahoo Finance.

* ``compute_volatility_features(df) -> pd.DataFrame``
  Construye todas las características causales del modelo.

* ``compute_log_vol_target(df, horizon, annualize=252) -> pd.Series``
  Log-volatilidad realizada forward a ``horizon`` días.

* ``build_volatility_dataset(df, horizon) -> pd.DataFrame``
  Combina features y target y elimina ``NaN``.

* ``temporal_split(data, train_end, val_end) -> tuple[DataFrame, ...]``
  División cronológica disjunta.

* ``make_sequences(X, y, lookback) -> tuple[np.ndarray, np.ndarray]``
  Ventanas deslizantes para la LSTM.

* ``align_har_features_with_sequences(features_df, y_seq, lookback) -> pd.DataFrame``
  Alinea el benchmark HAR-RV con las secuencias de la LSTM.

* ``scale_splits(...) -> dict``
  Escala train/val/test y devuelve arrays y escaladores.

### `thesis.volatility_models`

* ``HARRVModel``
  Benchmark HAR-RV con covarianza HAC.

* ``VolatilityLSTM``
  Red LSTM con Pre-LayerNorm, residual y dropout.

* ``ExponentialAsymmetricLoss``
  Pérdida LINEX asimétrica para penalizar subestimaciones.

* ``train_lstm(model, train_loader, X_val_t, y_val_t, config, device) -> dict``
  Bucle de entrenamiento con early stopping y checkpointing.

### `thesis.volatility_evaluation`

* ``evaluate_forecasts(y_true, y_pred, label) -> dict``
  Métricas de pronóstico incluyendo Mincer-Zarnowitz.

* ``diebold_mariano_test(e1, e2, h) -> dict``
  Test de igualdad de precisión predictiva bajo MSE.

* ``run_vol_timing_backtest(ret, pred_vol, sigma_bar, cost_bps) -> dict``
  Backtest de vol-timing con momentum y costos.

* ``calculate_adaptability(y_true, y_pred, horizon, threshold) -> tuple``
  Curva Event-MAE y vida media tras shocks de volatilidad.

* ``generate_plots(...) -> None``
  Figura diagnóstica de 6 paneles.

---

## Características implementadas

### Componentes HAR-RV

```text
RV_t^(d) = r_t^2
RV_t^(w) = (1/5) * Σ_{i=0..4} r_{t-i}^2
RV_t^(m) = (1/22) * Σ_{i=0..21} r_{t-i}^2
```

### Estimadores de rango

**Parkinson (1980)**

```text
σ_P = sqrt( rolling_mean[ log(H/L)^2 ] / (4 * log(2)) )
```

**Garman-Klass (1980)**

```text
σ_GK^2 = 0.5 * log(H/L)^2 - (2*log(2)-1) * log(C/O)^2
```

### Target

```text
y_t = log( sqrt( (252/H) * Σ_{i=1..H} r_{t+i}^2 ) )
```

Todas las features usan información hasta ``t``; el target usa
``r_{t+1}, ..., r_{t+H}``.

---

## Ejecución

Desde la raíz del repositorio:

```bash
# Instalar dependencias (incluye torch CPU, ~200 MB)
uv sync

# Con uv
uv run jupytext --to notebook notebooks/volatility/Volatilidad_LSTM.py
uv run jupyter notebook notebooks/volatility/Volatilidad_LSTM.ipynb

# O ejecutar directamente el script
uv run python notebooks/volatility/Volatilidad_LSTM.py
```

`pyproject.toml` fija `torch` al índice de wheels CPU de PyTorch
(`https://download.pytorch.org/whl/cpu`) para evitar descargar las
dependencias CUDA, que ocupan ~3 GB y no se usan en este proyecto.  El
entorno resultante reporta `torch.cuda.is_available() == False` y el
notebook entrena en CPU.

El notebook guarda los siguientes artefactos en el directorio de
trabajo actual:

| Archivo | Contenido |
| ------- | --------- |
| ``model_config.json`` | Hiperparámetros del experimento. |
| ``lstm_vol_model.pt`` | Pesos del mejor modelo LSTM. |
| ``scaler.pkl`` | Escalador de features. |
| ``target_scaler.pkl`` | Escalador del target. |
| ``metrics.csv`` | Métricas de LSTM y HAR-RV. |
| ``report.md`` | Reporte de resultados (reservado). |
| ``lstm_vol_report.png`` | Figura diagnóstica. |

---

## Interpretación y precauciones

* **Look-ahead bias**: todas las features son causales.  El target es
  forward-looking por diseño y solo se usa como variable dependiente.
* **División temporal**: las particiones son estrictamente disjuntas.
  No se permite solapamiento entre train, validation y test.
* **Estandarización**: la LSTM trabaja en espacio estandarizado; las
  predicciones se invierten antes de calcular métricas.  HAR-RV se
  entrena y evalúa directamente en log-vol crudo.
* **Backtest**: ``sigma_bar`` se calcula sobre el conjunto de
  entrenamiento para evitar look-ahead.  El costo por defecto es 5 bps
  unilateral.
* **Semilla**: ``SEED=42`` se fija en ``thesis.volatility_models`` para
  reproducibilidad.
* **Dependencias**: requiere ``torch`` y ``statsmodels``.  Ver
  ``pyproject.toml``.
* **Consistencia OHLC**: ``validate_ohlcv`` reconstruye ``High``/``Low``
  para que contengan ``Open``/``Close``.  ``yfinance`` puede publicar
  barras intradía inconsistentes (el cierre de la sesión en curso antes
  que su máximo), lo que sesgaría los estimadores de Parkinson y
  Garman-Klass.

---

## Verificación rápida

```bash
# Compilar todos los módulos
python -m py_compile \
    src/thesis/volatility.py \
    src/thesis/volatility_models.py \
    src/thesis/volatility_evaluation.py \
    notebooks/volatility/Volatilidad_LSTM.py

# Tests unitarios (sintéticos, sin red)
uv run pytest tests/test_volatility.py tests/test_volatility_models.py tests/test_volatility_evaluation.py -v

# Tests de integración con datos reales (requiere red)
uv run pytest -m integration -v
```

Los tests de `tests/test_volatility_models.py` usan `pytest.importorskip`
y se omiten automáticamente si `torch` no está disponible en el entorno.

Los tests de integración están marcados con `integration` y se omiten por
defecto (`addopts = "-m 'not integration'"` en `pyproject.toml`) porque
requieren conexión a Yahoo Finance.  Entre ellos se incluye una prueba de
ausencia de *look-ahead*: se entrena el modelo, se corrompe la cola de la
serie y se comprueba que las features y las predicciones anteriores a la
corrupción son idénticas.
