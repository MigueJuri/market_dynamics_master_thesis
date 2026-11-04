# Proceso de entrenamiento (LSTM de volatilidad)

Resumen **estático** del pipeline completo de
`notebooks/volatility/Volatilidad_LSTM.py`: features, target, estandarización,
split, secuencias, entrenamiento, post-procesado, evaluación y backtest.

El documento describe *qué* hace el pipeline y *en qué orden*, no reproduce ni
entrena el modelo.  La **arquitectura de la red** (`VolatilityLSTM`: capas,
inicialización, `forward`, residual, `LayerNorm`) queda deliberadamente fuera
de alcance: aquí solo se indica que la pérdida es LINEX asimétrica con
`b = linex_b`.

Todo lo que sigue se ha verificado contra el código, no de memoria.

---

## Fuentes de código

| Paso | Ubicación | Elementos clave |
| ---- | --------- | --------------- |
| Orquestación | `notebooks/volatility/Volatilidad_LSTM.py` | `CONFIG`, `run_experiment` |
| Features / target / splits | `src/thesis/volatility.py` | `download_ohlcv`, `validate_ohlcv`, `compute_volatility_features`, `compute_log_vol_target`, `build_volatility_dataset`, `temporal_split`, `purge_boundary_labels`, `scale_splits`, `make_sequences`, `align_har_features_with_sequences`, `count_extreme_moves` |
| Modelos | `src/thesis/volatility_models.py` | `HARRVModel`, `ExponentialAsymmetricLoss`, `train_lstm` |
| Evaluación | `src/thesis/volatility_evaluation.py` | `evaluate_forecasts`, `diebold_mariano_test`, `run_vol_timing_backtest`, `summarize_returns` |
| Constantes | `src/thesis/config.py` | `SEED = 42`, `TRADING_DAYS = 252` |

---

## Flujo resumido

```text
CONFIG
  → download_ohlcv + validate_ohlcv          (datos limpios)
  → build_volatility_dataset                 (features + target, dropna)
  → temporal_split                           (train / val / test por fecha)
  → purge_boundary_labels                    (quita los labels que cruzan la frontera)
  → scale_splits                             (StandardScaler, fit solo en train)
  → make_sequences                           (ventanas deslizantes L = 5)
  → HARRVModel.fit                           (baseline en log-vol crudo)
  → train_lstm                               (AdamW + LINEX + early stopping)
  → inverse_transform + EWMA + exp           (log-vol → σ anualizada)
  → evaluate_forecasts + diebold_mariano_test
  → run_vol_timing_backtest                  (vol-timing con σ̂ sobre señal de momentum)
```

---

## 1. Configuración

`CONFIG` (en el cuaderno) fija todos los parámetros del experimento:

| Clave | Valor | Comentario |
| ----- | ----- | ---------- |
| `ticker` | `"SPY"` | ETF del S&P 500. |
| `start_date` | `"1995-01-01"` | Inicio de la descarga. |
| `train_end` | `"2017-01-01"` | Último día **inclusive** de train. |
| `val_end` | `"2022-01-01"` | Último día **inclusive** de validación. |
| `lookback` | `5` | Longitud de la ventana histórica (`L`). |
| `forecast_horizon` | `5` | Horizonte del target y del test DM (`H`). |
| `smoothing_window` | `15` | `span` de la suavización EWMA final. |
| `linex_b` | `0.8` | Parámetro de asimetría de la pérdida LINEX. |
| `features` | 13 columnas (ver abajo) | Entradas de la LSTM y del baseline. |
| `batch_size` | `16` | Tamaño de batch del `DataLoader`. |
| `lr` | `5e-5` | Tasa de aprendizaje de `AdamW`. |
| `weight_decay` | `1e-3` | Decaimiento de pesos. |
| `min_epochs` | `50` | Mínimo de épocas antes de admitir la parada. |
| `max_epochs` | `1000` | Cota superior de épocas. |
| `patience` | `15` | Épocas sin mejora para detener. |
| `grad_clip` | `1.0` | Norma máxima de gradiente por batch. |
| `cost_bps` | `0.0` | Costo de transacción unilateral (puntos básicos). |
| `model_path` | `"lstm_vol_model.pt"` | Checkpoint del mejor modelo. |
| `scaler_path` / `target_scaler_path` | `"scaler.pkl"` / `"target_scaler.pkl"` | Escaladores serializados. |
| `metrics_path` | `"metrics.csv"` | Métricas de test. |
| `report_path` | `"report.md"` | Reporte (reservado). |
| `config_path` | `"model_config.json"` | Copia de la configuración usada. |

Las 13 features, en el orden en que se pasan al modelo:

```text
rv_daily, rv_weekly, rv_monthly, vol_ratio, gap, parkinson_vol, gk_vol,
rsi_14, rel_volume, volume_trend, intraday_range, skew_5, mean_ret_5
```

Además, `run_experiment` fija `set_volatility_seed(SEED)` con `SEED = 42`
(`thesis.config`) para reproducibilidad de `numpy` y `torch`, y elige el
dispositivo `cuda` si está disponible, si no `cpu`.

> Los valores de arquitectura (`hidden`, `dropout`) existen en `CONFIG` pero
> **no** se describen aquí, por decisión de alcance.

---

## 2. Datos

`download_ohlcv(ticker, start=...)` descarga con `yfinance`
(`auto_adjust=True` por defecto, de modo que los precios vienen ajustados por
dividendos y splits) y delega inmediatamente en `validate_ohlcv`, que:

1. exige las columnas `Open, High, Low, Close, Volume`;
2. convierte el índice a `DatetimeIndex` y ordena cronológicamente
   (`sort_index`);
3. aplica `ffill()` sobre los `NaN` y descarta las filas que siguen nulas;
4. lanza `ValueError` si algún precio es no positivo;
5. **reconstruye la consistencia OHLC**: `High ← max(Open, High, Close)` y
   `Low ← min(Open, Low, Close)`.  Esto protege a los estimadores de rango
   (Parkinson y Garman-Klass) frente a barras intradía inconsistentes
   publicadas por `yfinance`.

`count_extreme_moves(raw)` cuenta los movimientos diarios con
`|log-retorno| > 15 %` (umbral por defecto `0.15`).  El cuaderno solo
**informa** del recuento (`INFO: … movimientos diarios > 15% detectados
(retenidos)`): **no se eliminan**, porque son parte del proceso que se quiere
modelar.

---

## 3. Features

`compute_volatility_features(df)` construye un `DataFrame` indexado por fecha
con todas las columnas predictoras.  Todas son **causales**: en el tiempo `t`
solo usan información disponible hasta `t` (inclusive).

| Feature | Definición | Notas |
| ------- | ---------- | ----- |
| `log_return` | `log(C_t / C_{t-1})` | Retorno logarítmico diario. |
| `rv_daily` | `r_t^2` | Componente diario HAR-RV. |
| `rv_weekly` | media móvil de `r^2` en 5 sesiones | Componente semanal HAR-RV. |
| `rv_monthly` | media móvil de `r^2` en 22 sesiones | Componente mensual HAR-RV. |
| `vol_20` | `std(r, 20) · sqrt(252)` | **Se calcula pero no está en `CONFIG["features"]`**: no entra al modelo. |
| `vol_ratio` | `rv_daily / (rv_weekly + 1e-10)` | Razón diaria/semanal. |
| `parkinson_vol` | `sqrt( media20[ log(H/L)^2 ] / (4·ln 2) )` | Estimador de rango con corrección C2. |
| `gk_vol` | `sqrt( media20[ max(0, 0.5·log(H/L)^2 − (2·ln2 − 1)·log(C/O)^2) ] )` | Garman-Klass, con `clip(lower=0)` dentro de la media. |
| `rsi_14` | `(100 − 100/(1+RS)) / 100`, con `RS = media14(ganancias) / (media14(pérdidas) + 1e-10)` | Normalizado a `[0, 1]`. |
| `rel_volume` | `Volume / (media20(Volume) + 1e-10)` | Volumen relativo. |
| `volume_trend` | `media5(Volume) / (media20(Volume) + 1e-10)` | Tendencia de volumen. |
| `intraday_range` | `(High − Low) / Close.shift(1)` | Rango intradía normalizado por el cierre previo. |
| `gap` | `(Open − Close.shift(1)) / Close.shift(1)` | Gap de apertura. |
| `skew_5` | `skew(r, 5)` | Asimetría de retornos recientes. |
| `mean_ret_5` | `media(r, 5)` | Retorno medio reciente. |

**Compresión logarítmica.** Al final de la función se aplica
`log( x.clip(lower=1e-12) )` a las nueve columnas **estrictamente positivas**:

```text
rv_daily, rv_weekly, rv_monthly, vol_ratio, parkinson_vol, gk_vol,
rel_volume, volume_trend, intraday_range
```

Las restantes (`log_return`, `gap`, `rsi_14`, `skew_5`, `mean_ret_5`, y
`vol_20` no usado) pueden ser negativas y **no** se comprimen.  El clip en
`1e-12` evita `log(0)`; las divisiones usan `+1e-10` para no dividir por cero.

---

## 4. Target

`compute_log_vol_target(df, horizon=H)` define la etiqueta:

```text
y_t = log( sqrt( (252/H) · Σ_{i=1..H} r_{t+i}^2 ) )
```

Implementación: `rv_sq = log_ret**2`;
`fwd_rv = rv_sq.rolling(H, min_periods=H).sum().shift(-H)`;
`target = log( clip( sqrt((252/H)·fwd_rv), lower=1e-12 ) )`.

* `A = 252` (`TRADING_DAYS`): el factor `252/H` convierte la suma de cuadrados
  de retornos de los próximos `H` días en una **varianza anualizada**.
* Las últimas `H` observaciones son `NaN` (no hay futuro).
* El target **usa datos futuros por diseño**: es la variable dependiente.
  No es *look-ahead* en las features, que siguen siendo estrictamente
  causales.

---

## 5. Dataset

`build_volatility_dataset(raw, horizon=5)` concatena features y target en un
único `DataFrame` y aplica `dropna()`.  Ese `dropna` elimina tanto el
warm-up de las ventanas móviles (22 sesiones para `rv_monthly`, 20 para
Parkinson/GK/…) como las últimas `H` filas sin futuro.

---

## 6. Split temporal

`temporal_split(data, train_end="2017-01-01", val_end="2022-01-01")`:

```text
train = data[index <= "2017-01-01"]
val   = data[("2017-01-01" < index) <= "2022-01-01"]
test  = data[index > "2022-01-01"]
```

Particiones **estrictamente disjuntas** y ordenadas en el tiempo: el train
nunca ve datos de validación o test, y la validación nunca ve test.

**Purga de frontera** (`purge_boundary_labels`): el target de la fila `t` usa
los retornos de `t+1 ... t+H`, así que las últimas `H` filas de train tienen
labels apoyados en la validación, y las últimas `H` de validación en el test.
Esas filas se eliminan antes de estandarizar y de construir las secuencias:

```text
train = purge_boundary_labels(train, H)   # sin las últimas H filas
val   = purge_boundary_labels(val,   H)   # sin las últimas H filas
```

La purga es posicional (el target usa ventanas móviles sobre filas), por lo que
no depende de que el calendario tenga días hábiles contiguos.

---

## 7. Estandarización

`scale_splits(X_tr, X_vl, X_te, y_tr, y_vl, y_te)` aplica **dos**
`StandardScaler` independientes (de `sklearn.preprocessing`):

```text
X_scaler.fit_transform(X_train)   →  X_scaler.transform(X_val), X_scaler.transform(X_test)
y_scaler.fit_transform(y_train)   →  y_scaler.transform(y_val), y_scaler.transform(y_test)
```

* **Qué** se estandariza: las features `X` (13 columnas) y el target `y`, por
  separado, con su propio escalador.
* **Cuándo**: `fit` **únicamente sobre train**; val y test solo reciben
  `transform`.  Ninguna estadística de validación o test entra en el
  escalador.
* **Dónde**: en **espacio log-volatilidad**.  El target estandarizado es
  `log σ` estandarizado, no `σ`; por eso las predicciones de la LSTM se
  deshacen con `y_scaler.inverse_transform` antes de compararlas con el
  target.
* **Cuándo se aplica**: **antes** de `make_sequences` (paso 8).  Las
  secuencias se construyen ya sobre datos estandarizados.

> HAR-RV es la excepción: se entrena y predice en **log-vol crudo** (sin
> estandarizar), de modo que sus predicciones son directamente comparables
> con el target sin invertir ninguna transformación.

---

## 8. Secuencias

`make_sequences(X, y, lookback=L)` genera pares con ventana deslizante:

```text
secuencia i:  X[i : i+L]        (L filas, la más reciente es i+L-1)
etiqueta   i:  y[i + L]
formas:  X_seq → (T - L, L, n_features)      y_seq → (T - L,)
```

Devuelve arrays `float32`.  Se aplica de forma independiente a train, val y
test, por lo que cada partición pierde sus primeras `L` observaciones.

El `DataLoader` de entrenamiento se construye con
`TensorDataset(Xtr_seq, ytr_seq)` y
`DataLoader(batch_size=16, shuffle=True, drop_last=True)`:

* `shuffle=True`: las secuencias se mezclan en cada época (son muestras
  intercambiables, no una serie que deba recorrerse en orden).
* `drop_last=True`: se descarta el último batch incompleto, de modo que todas
  las Updates usen batches del mismo tamaño.

Validación y test se pasan como tensores completos (`X_vl_t`, `y_vl_t`,
`X_te_t`), sin `DataLoader`.

---

## 9. Baseline HAR-RV

`HARRVModel` es el benchmark canónico (Corsi, 2009):

```text
y_t = β₀ + β_d·RV_t^(d) + β_w·RV_t^(w) + β_m·RV_t^(m) + ε_t
```

* **Alineación**: `align_har_features_with_sequences(features_df, y_seq, L)`
  toma, para cada secuencia, el vector de features del **último día de la
  ventana** (`iloc[L-1 : L-1+len(y_seq)]`), de modo que HAR-RV y LSTM usan
  exactamente la misma información disponible en `t`.
* **Ajuste**: `HARRVModel.fit` ejecuta MCO (`sm.OLS`) con `add_constant`
  sobre `rv_daily`, `rv_weekly`, `rv_monthly` y covarianza **HAC
  (Newey–West)** con `maxlags=10`.  El target de entrenamiento es el
  log-vol **crudo** alineado (`y_tr[L : L + len(ytr_seq)]`), no el
  estandarizado.
* **Predicción**: `har.predict(har_X_te)[:len(yte_seq)]` devuelve
  predicciones en **log-vol crudo**.

---

## 10. Entrenamiento

`train_lstm(model, train_loader, X_val_t, y_val_t, config, device)`:

| Elemento | Valor / comportamiento |
| -------- | ----------------------- |
| Pérdida | LINEX asimétrica (`ExponentialAsymmetricLoss`) con `b = linex_b = 0.8`: `L(e) = exp(b·e) − b·e − 1`, `e = y − ŷ`. Penaliza **exponencialmente** las subestimaciones y linealmente las sobreestimaciones. |
| Optimizador | `AdamW(lr=5e-5, weight_decay=1e-3)`. |
| Scheduler | `ReduceLROnPlateau(mode="min", patience=5, factor=0.5, min_lr=1e-6)`, llamado con la pérdida de validación en cada época. |
| Gradient clipping | `clip_grad_norm_(params, 1.0)` en cada batch; los batches con norma superior se cuentan. |
| Batches con `NaN` | Se saltan (`continue`) antes del paso del optimizador. |
| Validación | `model.eval()` + `torch.no_grad()`; `val_loss` sobre el tensor completo de validación. |
| Checkpoint | Se guarda `model.state_dict()` en `model_path` cada vez que `val_loss < best_val` (y se reinicia el contador de paciencia). |
| Early stopping | Corte cuando `patience_ctr >= 15` **y** `epoch >= min_epochs = 50`; `max_epochs = 1000` es la cota superior. |
| Restaura | Al final se recarga el checkpoint guardado (`load_state_dict`) y el modelo queda en modo `eval()`. |

**Diagnóstico de gradientes.** Al terminar se informa el porcentaje de
batches con clipping:

```text
clip_pct = 100 · clip_count / total_batches
```

y se emite la alerta `*** ALERTA: clipping > 5%. Considerar reducir lr. ***`
si `clip_pct > 5`.

**Chequeo de sobredeterminación** (en el cuaderno, antes de entrenar):
`n_secuencias_train / n_params`; si la razón es `≤ 1` se avisa de que el
modelo está sobredeterminado respecto al número de secuencias.

---

## 11. Post-procesado de predicciones

Con `n_eval = min(len(yte_seq), len(y_pred_lstm), len(y_pred_har_te))`:

1. **Deshacer la estandarización** (`y_scaler.inverse_transform`):
   * real: `y_te_s[L : L+n_eval]` → log-vol real;
   * LSTM: `y_pred_lstm` (salida de la red en espacio estandarizado) → log-vol;
   * HAR-RV: ya está en log-vol crudo, no requiere inversión.
2. **Suavizado EWMA** con `span = smoothing_window = 15` y `adjust=False`
   (forma recursiva `y_t = (1−α)·y_{t−1} + α·x_t`), aplicado **sólo a los
   pronósticos** (LSTM y HAR-RV).  La verdad se conserva **cruda**: el
   suavizado de la serie real cambiaría el target evaluado y añadiría
   warm-up en test.  Las versiones sin suavizar se exponen como
   `y_true_raw`, `y_lstm_raw` y `y_har_raw`, y son las que se usan en las
   métricas primarias.
3. **Conversión a espacio sigma**: `np.exp(...)`, es decir,
   `σ̂ = exp(log σ̂)` = volatilidad **anualizada** (ya incluye el factor 252
   del target).

> El EWMA se aplica sobre la **serie completa de test**: es un suavizado de
> evaluación, no una operación de entrenamiento.  Los primeros puntos del
> test se suavizan con una ventana EWMA aún corta (sesgo inicial de la forma
> recursiva).

---

## 12. Evaluación

`evaluate_forecasts(y_true, y_pred, label)` se ejecuta dos veces
(`"LSTM"` y `"HAR-RV (Baseline)"`) **en espacio sigma** (volatilidad
anualizada), sobre las series post-procesadas:

| Métrica | Definición |
| ------- | ---------- |
| `R2` | `r2_score`. |
| `RMSE` / `MAE` | Error cuadrático medio y absoluto (se reportan en % de volatilidad anualizada). |
| `Spearman_r`, `Spearman_p` | Correlación de rangos entre real y predicho. |
| `Dir_Acc`, `Dir_p` | Coincidencia de signo respecto a la mediana de cada serie, con p-valor exacto de `stats.binomtest` (`H0 = 50 %`). |
| `QLIKE` | `mean( y²/(ŷ²+1e-8) − log(y²/(ŷ²+1e-8)) − 1 )`; menor es mejor. |
| `MZ_alpha`, `MZ_beta` | Regresión Mincer–Zarnowitz `y_true = α + β·y_pred` (covarianza HAC, `maxlags=10`). |
| `MZ_F_pval` | Test F conjunto de `H0: α = 0, β = 1` (pronóstico insesgado). |
| `N` | Número de observaciones efectivas (se descartan `NaN`/`inf`). |

**Diebold–Mariano**: `diebold_mariano_test(e_lstm, e_har, h=5)` con
`e = y_true − y_pred` (residuos en espacio sigma):

```text
d = e1² − e2²,   HAC(Bartlett) con pesos (1 − k/h),  k = 1..h-1
se = sqrt( max(hac_var, 1e-12) / T ),   DM = mean(d) / se
p-valor bilateral con la normal estándar
```

El signo del estadístico indica qué modelo es más preciso bajo el criterio
MSE, y el p-valor si la diferencia es significativa al 5 %.

El cuaderno además imprime un *checklist de validación* sobre las métricas de
la LSTM: `R² > 0.08`, `Dir_Acc > 0.55`, `Spearman r > 0.15`,
`Spearman p < 0.05` y `MZ F p > 0.05` (insesgado).

---

## 13. Backtest de vol-timing

**Referencia `sigma_bar`** (anti-look-ahead): se calcula **solo con el
entrenamiento**:

```text
sigma_bar = median( exp( y_tr[L : L + len(ytr_seq)] ) ) + 1e-8
```

es decir, la mediana de la volatilidad realizada **en train**, en espacio
sigma.  Nunca se usa información de validación o test.

**Retornos de test**: `log(C_t / C_{t-1})` sobre la serie cruda,
reindexados a las fechas de test, con `fillna(0.0)` y recortados a
`[L+1 : L+n_eval]` para alinearlos con los pronósticos.

El forecast `j` se origina al cierre de `dates[j-1]` y predice los retornos de
`dates[j+1 ... j+H]`; por tanto el retorno operable es el de `dates[j+1]` y el
último forecast no es operable (su ventana cae fuera del test).  De ahí las
longitudes `n_eval` (métricas) y `n_eval - 1` (backtest):

```text
dates          = te_dates[L     : L+n_eval]       # n_eval
bt_dates       = te_dates[L + 1 : L+n_eval]       # n_eval - 1
ret_te_aligned = te_log_ret[L + 1 : L+n_eval]     # n_eval - 1
```

`run_vol_timing_backtest(ret_te_aligned, y_lstm[:-1], sigma_bar, cost_bps)`:

```text
señal base: sig = signo( media_móvil( ret, ventana = min(63, n // 4),
                                       min_periods = 5 ).shift(1) )
            posición long/short (-1 / 0 / +1)
escala:    vol_scale = σ̂(t) / sigma_bar
posición:  pos_vol  = clip( sig / (vol_scale + 1e-8), -1, +1 )
           pos_base = clip( sig, -1, +1 )
           pos_bh   = +1  (buy & hold)
costos:    cost = cost_bps · 1e-4,   tc = cost · |Δposición|
retornos:  r_simple = expm1(r)
           ret_vol  = pos_vol  · r_simple − tc_vol
           ret_base = pos_base · r_simple − tc_base
           ret_bh   = r_simple
```

* La señal base es de **momentum**: el signo de la media móvil de los retornos
  (logarítmicos) del test, con ventana adaptativa `min(63, n // 4)` y un
  mínimo de 5 observaciones.  El warm-up de la media móvil deja la posición en
  flat (`0`) durante las primeras barras.
* El `.shift(1)` de la media garantiza que la señal de `t` usa
  **exclusivamente** información anterior a `t` (sin look-ahead).
* El `clip(·, −1, +1)` limita la exposición: cuando la volatilidad predicha
  supera la de referencia (`σ̂ > σ̄`) la posición se reduce, y cuando es
  inferior se amplifica (hasta el límite de ±1).
* Costos: `cost_bps` puntos básicos **unilaterales** por cambio de posición;
  buy & hold no incurre costos.
* Toda la aritmética de la cartera (posiciones, costos y retornos) vive en
  **retornos simples**: no se mezclan log-retornos con costos.

Métricas (`summarize_returns`, sobre retornos simples): retorno total
(`prod(1+r) - 1`), Sharpe anualizado (`sqrt(252) · mean(r) / (std(r) + 1e-10)`),
máximo drawdown (riqueza `cumprod(1+r)` **anteponiendo el capital inicial
1.0**) y ratio de Calmar (`CAGR / |MDD|`, con `CAGR = prod(1+r)^(252/n) - 1`),
para las tres estrategias, más la mejora de Sharpe frente al
baseline y frente a buy & hold.  El cuaderno considera la predicción de
volatilidad útil si la mejora de Sharpe sobre el baseline supera `0.2`.

---

## Espacios y unidades

| Etapa | Espacio | Nota |
| ----- | ------- | ---- |
| Features | mezclado (log para las positivas, Levels para `gap`, `rsi_14`, `skew_5`, `mean_ret_5`) | Compresión logarítmica de 9 columnas. |
| Features estandarizadas | `z-scores` | `X_scaler` ajustado en train. |
| Target crudo | log-volatilidad anualizada | `log( sqrt((252/H)·Σ r²_{t+i}) )`. |
| Target estandarizado | `z-scores` de log-vol | `y_scaler` ajustado en train. |
| Predicción LSTM (red) | log-vol estandarizado | Se invierte con `inverse_transform`. |
| Predicción HAR-RV | log-vol crudo | Sin inversión. |
| Salida final (`y_true`, `y_lstm`, `y_har`) | σ anualizada | `exp(log σ)`; RMSE y MAE en % de vol anualizada. |

---

## Precauciones

### Look-ahead

* `fit` de los dos `StandardScaler` ocurre **exclusivamente sobre train**.
* Todas las features son causales (`shift(1)` en `intraday_range` y `gap`;
  ventanas móviles que incluyen el día corriente).
* El **target** usa datos futuros por diseño: es la etiqueta, no una entrada.
* `purge_boundary_labels` elimina las últimas `H` filas de train y de
  validación, cuyos labels se apoyarían en la partición siguiente.
* La señal de momentum del backtest usa `.shift(1)`.
* `sigma_bar` se calcula solo con train.
* El EWMA final se aplica sobre la serie completa de test y **sólo sobre los
  pronósticos**: es suavizado de evaluación, no de entrenamiento.

### Numérico

| Constante | Uso |
| --------- | --- |
| `1e-12` | `clip(lower=1e-12)` antes de todos los `log` (features, target) y suelo de `sqrt(hac_var)` en el test DM. |
| `1e-10` | Sumas al denominador: `vol_ratio`, RSI (pérdidas), `rel_volume`, `volume_trend`. |
| `1e-8` | QLIKE (`y²/(ŷ²+1e-8)`), `sigma_bar`, `vol_scale + 1e-8`, Sharpe y drawdown. |
| `1.0` | `grad_clip`: norma máxima por batch; alerta si el clipping supera el 5 % de los batches. |
| `252` | `TRADING_DAYS`: anualización del target, del Sharpe y de `vol_20`. |

### Precisión factual (verificada en el código)

* Ventanas HAR-RV: **1 / 5 / 22** sesiones sobre `log_ret²`.
* Anualización: **252** sesiones.
* `DataLoader` de train: `shuffle=True`, `drop_last=True`.
* `temporal_split`: `train ≤ train_end`, `train_end < val ≤ val_end`,
  `test > val_end`, sin solapamiento.
* `make_sequences` devuelve `(T − L, L, n_features)` y `(T − L,)`.
* El escalado ocurre **antes** de construir las secuencias.
* El backtest opera sobre **retornos simples**: `ret_vol = pos·expm1(r) − tc`.
* El drawdown incluye el capital inicial: la riqueza es `cumprod(1+r)` con
  `1.0` antepuesto.
