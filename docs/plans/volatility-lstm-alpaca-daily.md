# Plan: LSTM Alpaca con estimadores diarios (SIP) + caché

## Objetivo
Reescribir `notebooks/volatility/Volatilidad_LSTM_Alpaca.py` para que replique la
estructura y los targets de `Volatilidad_LSTM.py` (lookback=5 días, horizonte=5,
suavizado EWMA solo de pronósticos, métricas raw-vs-raw), usando barras intradía
de Alpaca (feed **SIP**, desde 2015) **únicamente** para construir features y
targets convertidos a **estimadores diarios**. Añadir caché de precios y de
modelo para no re-descargar ni re-entrenar cuando la configuración no cambia.

## Decisiones
- `timeframe="15Min"` por defecto; cambiar a `5Min`/`1Min` es una sola clave de
  CONFIG (la agregación es por fecha de calendario, sin `bars_per_day`).
- `feed="sip"`; `start_date="2015-01-01"`. Splits por defecto
  `train_end="2020-12-31"`, `val_end="2022-12-31"` (configurables).
- `smoothing_window=15` (como el diario); **la verdad nunca se suaviza**.
- `rv_daily` = varianza realizada intradía (suma de `r²` de las barras de la
  sesión, anulando el salto overnight de la 1ª barra). `rv_weekly`/`rv_monthly`
  = medias de 5/22 días de esa RV. Target = RV forward de `H` días anualizada.

## Archivos / funciones

### 1. `src/thesis/volatility_intraday.py`
- Añadir `aggregate_intraday_to_daily(df) -> pd.DataFrame`: agrupa por fecha →
  OHLCV diario (Open=first, High=max, Low=min, Close=last, Volume=sum) +
  `rv_intraday`.
- Añadir `compute_volatility_features_from_intraday_daily(daily)`: reutiliza
  `thesis.volatility.compute_volatility_features` sobre el OHLCV diario y
  sobreescribe `rv_daily`/`rv_weekly`/`rv_monthly`/`vol_ratio` con la RV
  intradía (misma compresión log, mismas 13 columnas).
- Añadir `compute_log_vol_target_from_intraday_daily(daily, horizon, annualize=252)`:
  `log(sqrt((252/horizon) · rv.rolling(horizon).sum().shift(-horizon)))`.
- Añadir `build_volatility_dataset_from_intraday(df, horizon)` =
  aggregate → features → target, `dropna()`.
- Modificar `download_ohlcv_alpaca`: parámetros `cache_dir`/`refresh`; leer/escribir
  `cache_dir/<ticker>_<timeframe>_<feed>_<adjustment>_<start>_<end>.csv`
  (patrón de `market_data.load_single_price`).
- Eliminar (reemplazadas): `compute_volatility_features_intraday`,
  `compute_log_vol_target_intraday`, `build_volatility_dataset_intraday`,
  `DAILY_WINDOWS`. Conservar `infer_frequency_params`, `_build_timeframe`,
  `_utc_timestamp`.

### 2. `notebooks/volatility/Volatilidad_LSTM_Alpaca.py` (reescritura)
- CONFIG: `start_date="2015-01-01"`, `feed="sip"`, `timeframe="15Min"`,
  `lookback=5`, `forecast_horizon=5`, `smoothing_window=15`, `linex_b=0.8`,
  13 features, arquitectura/entrenamiento como el diario, artefactos `_alpaca`.
- `_merge_config`: quitar inferencia de frecuencia; añadir
  `config_fingerprint(cfg)` (hash SHA-256 sobre las claves que definen el
  experimento, excluyendo rutas de salida) y `cache_dir = DATA_DIR/"volatility_alpaca"/fingerprint`;
  reubicar `model_path`, `scaler_path`, `target_scaler_path`, `inference_path`,
  `metrics_path`, `config_path` en `cache_dir` salvo que el usuario las pase
  explícitas.
- `prepare_data`: descarga (con caché) → `daily = aggregate_intraday_to_daily`
  (se usa como `raw` diario) → `build_volatility_dataset_from_intraday`; si
  existe `cache_dir/dataset.csv`, cargarlo y saltar descarga+construcción. Resto
  (split, purge, scale, `make_sequences`, HAR-RV con `maxlags` por defecto)
  idéntico al diario. Persistir el dataset.
- `train_and_infer`: si `inference_path` existe (fingerprint igual), cargar
  `history` de él y saltar el entrenamiento; si no, entrenar y persistir (igual
  al diario, sin `bars_per_day`/`bars_per_year`). `ret_te_aligned` con retornos
  diarios de `raw["Close"]`.
- `smooth_predictions`: **no suavizar la verdad** — `y_true = y_true_raw =
  np.exp(art["y_true_log"])`; aplicar `_ewm` solo a `y_lstm`/`y_har`.
- `compute_metrics`, `run_backtest_and_strategies` (sin `periods_per_year`/
  `bars_per_day`, defaults diarios) y `run_experiment`: idénticos al diario.
- Figuras: las 7 del diario (loss, predictions, timeseries, adaptability,
  backtest, ranks, april2025) con prefijo `lstm_vol_alpaca_`; importar
  `calculate_adaptability`.

### 3. `tests/test_volatility_intraday.py` (reescritura)
- Tests: `aggregate_intraday_to_daily` (OHLCV diario + rv_intraday sin
  overnight), `compute_volatility_features_from_intraday_daily` (mismas columnas
  que el diario; causalidad), `compute_log_vol_target_from_intraday_daily`
  (forward, cola NaN), `build_volatility_dataset_from_intraday` (sin NaN), caché
  de `download_ohlcv_alpaca` (2ª llamada no descarga). Eliminar tests de las
  funciones bar-level reemplazadas.

### 4. `notebooks/volatility/README.md`
- Actualizar "Variante intradía (Alpaca)": estimadores diarios desde barras
  15Min, feed SIP 2015+, lookback/horizonte 5 días, caché de precios/modelo.

## Criterios de aceptación
1. LSTM alimentada con secuencias `(N, 5, 13)` diarias; features/target desde RV intradía.
2. Métricas primarias raw-vs-raw: `y_true` sin EWMA; EWMA solo en pronósticos.
3. `feed="sip"` + `start_date=2015` funcionan; cambiar `timeframe` no toca código.
4. 2ª ejecución con misma config no re-descarga ni re-entrena; config distinta → otro `cache_dir`.
5. Sin look-ahead (features ≤ t, escalador solo train, purga de frontera, `shift(1)`, forecast↔retorno operable).
6. `uv run pytest` verde; `uv run ruff check src tests` limpio; diario intacto.

## Riesgos principales
- **Feed SIP requiere suscripción**; sin acceso falla (documentar).
- **Overnight**: `rv_intraday` excluye el salto overnight mientras `gap`/`skew_5`
  usan cierre→cierre diario; documentar la convención.
- **Fingerprint**: claves incluidas/excluidas estables (rutas fuera; `SEED`,
  arquitectura e hiperparámetros dentro) para no invalidar el caché.
- **Caché vs rutas explícitas** (tests/`tmp_path`): respetar rutas pasadas por el usuario.
- **1Min** (~1M barras) puede golpear rate-limits; por eso 15Min por defecto.
- **Cache-hit**: `train_and_infer` debe devolver `history` desde `inference.pkl`
  para que la figura de pérdida siga funcionando.
