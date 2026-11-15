# Plan: LSTM de volatilidad con datos intraday de Alpaca

## Objetivo
Crear un archivo alternativo a `notebooks/volatility/Volatilidad_LSTM.py` que
descargue barras intraday de **Alpaca** (en vez de Yahoo diario) y entrene la
LSTM alimentándola con **cada barra**, usando lookback de 5 días y target =
volatilidad realizada forward de los próximos 5 días calculada a esa frecuencia.

Decisiones confirmadas:
- Frecuencia: **15 minutos** → `bars_per_day = 26`, `bars_per_year = 6552`.
- Feed: **IEX gratuito** (historia ~5-6 años) → `start` ~2020, splits redefinidos.
- Features: **a nivel de barra** (ventanas diarias reescaladas ×26).
- Alcance: **pipeline completo** (LSTM + baseline HAR-RV + backtest vol-timing).

## Scope
- Nueva lógica reutilizable en `src/thesis/volatility_intraday.py` (sigue la
  convención del repo: el notebook delega en `src/`).
- Nuevo notebook `notebooks/volatility/Volatilidad_LSTM_Alpaca.py` que replica
  las etapas `prepare_data → train_and_infer → smooth_predictions →
  compute_metrics → run_backtest_and_strategies`.
- Cambios **aditivos y retrocompatibles** en `volatility_evaluation.py` para
  anualización intraday. El notebook diario queda intacto.

## Archivos / funciones

### 1. `pyproject.toml` (dependencia)
- Añadir `alpaca-py` a `dependencies` (SDK oficial, gestiona paginación).
- Claves vía env `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (`.env`, ya ignorado).

### 2. NUEVO `src/thesis/volatility_intraday.py`
- `download_ohlcv_alpaca(ticker, start, end, timeframe="15Min", feed="iex",
  adjustment="all") -> pd.DataFrame`: usa `StockHistoricalDataClient`, devuelve
  OHLCV validado reutilizando `validate_ohlcv`. Nota: timestamp de Alpaca =
  **inicio** de barra; documentar.
- `compute_volatility_features_intraday(df, bars_per_day=26)`: espejo de
  `compute_volatility_features` con ventanas en barras. Mapeo (día→barra):
  `rv_daily`=Σr² sobre 26 barras; `rv_weekly`=media 130; `rv_monthly`=media 572;
  `vol_20`=std 520; `parkinson_vol`/`gk_vol`=rolling 520; `rsi_14`=364;
  `rel_volume`/`volume_trend`=520/130; `skew_5`/`mean_ret_5`=130. Conserva las
  mismas 13 columnas y compresión logarítmica (reusar las mismas `log_cols`).
- `compute_log_vol_target_intraday(df, horizon_bars=130, annualize=6552)`:
  RV forward = `Σ r²` sobre `horizon_bars`, `shift(-horizon_bars)`, anualizada.
- `build_volatility_dataset_intraday(df, horizon_bars)` = features + target,
  `dropna()`.
- Reutilizar tal cual: `temporal_split`, `purge_boundary_labels`,
  `make_sequences`, `align_har_features_with_sequences`, `scale_splits`
  (son posicionales, agnósticas a frecuencia).

### 3. `src/thesis/volatility_evaluation.py` (cambios aditivos)
- `sharpe_ratio(r, periods_per_year=252.0)`, `calmar_ratio(...)`,
  `summarize_returns(...)` → añadir `periods_per_year` con default 252.
- `run_vol_timing_backtest(..., periods_per_year=252.0, bars_per_day=1)`:
  ventana de momentum escala a `min(63 * bars_per_day, n // 4)` (63 días
  equivalentes); anualización usa `periods_per_year`.
- `HARRVModel.fit` (en `volatility_models.py`): añadir `maxlags: int = 10`
  opcional; para intraday pasar `maxlags = 5 * bars_per_day` (130).

### 4. NUEVO `notebooks/volatility/Volatilidad_LSTM_Alpaca.py`
- `CONFIG` propio: `ticker=SPY`, `start_date="2020-01-01"`,
  `train_end="2023-01-01"`, `val_end="2024-07-01"`, `bars_per_day=26`,
  `bars_per_year=6552`, `lookback=130`, `forecast_horizon=130`, `features` =
  mismas 13 columnas, `batch_size` ajustable.
- `prepare_data`: llama a `download_ohlcv_alpaca` +
  `build_volatility_dataset_intraday`; el resto del flujo (split, purge, scale,
  `make_sequences`, HAR-RV con `maxlags=130`) idéntico al original.
- `train_and_infer` / `smooth_predictions` / `compute_metrics`: copia de las
  etapas originales (la alineación `ret_te_aligned` ya funciona posicionalmente
  sobre barras). Pasar `h=cfg["forecast_horizon"]` (130) al DM.
- `run_backtest_and_strategies`: llama a `run_vol_timing_backtest` con
  `periods_per_year=6552, bars_per_day=26`.
- Artefactos con prefijo propio (`lstm_vol_alpaca_*.pt/.pkl/.csv/.png`) para **no
  pisar** los del notebook diario.

## Criterios de aceptación
1. `prepare_data` descarga barras 15m de SPY desde Alpaca (requiere claves) y
   genera dataset con 13 features por barra + target anualizado (6552).
2. Secuencias `(N, 130, 13)`; splits temporales disjuntos; purga de `horizon`
   barras en train/val; escalador ajustado solo en train.
3. LSTM entrena con LINEX + early stopping y persiste artefactos `_alpaca`.
4. `compute_metrics` produce R²/RMSE/MAE/Spearman/DM/QLIKE/MZ y escribe
   `metrics_alpaca.csv`.
5. Backtest vol-timing a 15m con Sharpe/CAGR anualizados a 6552 y momentum de
   ~63 días equivalentes (sin hardcodear 252).
6. Sin look-ahead: features ≤ t, normalización solo train, purga de frontera,
   señal de momentum `shift(1)`, alineación forecast↔retorno operable.
7. `uv run pytest` sigue verde; `ruff check` limpio; notebook diario inalterado.

## Riesgos principales
- **Historia corta (IEX ~5-6 años)**: muestra pequeña, sin regímenes de 2008/2020
  completo; los splits 2017/2022 del original no aplican.
- **Credenciales/rate-limit**: sin `ALPACA_API_KEY` falla; paginación de barras y
  límites de la API IEX.
- **Convención de timestamp Alpaca** (inicio de barra) vs cierre diario: verificar
  alineación de `gap`/`intraday_range` y del target.
- **Anualización incorrecta** (252 vs 6552) en Sharpe/CAGR/backtest: mitigado
  parametrizando `periods_per_year`.
- **Overlap de secuencias** (ventana 130, stride 1) → muestras muy correladas;
  HAR-RV con `maxlags=10` subestima errores (pasar 130).
- **Costo computacional**: 130×13 por secuencia y ~30k secuencias; `batch_size`/
  `patience` pueden requerir ajuste.
- **Backtest a 15m**: turnover/costos por barra pueden dominar; señal de momentum
  debe escalarse (63 días = 1638 barras).
