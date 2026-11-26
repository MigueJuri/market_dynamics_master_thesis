# Plan: Integrar `notebooks/portfolio/portfolio_HRP.py` al repo + σ HAR-RV

## Objetivo

Convertir `notebooks/portfolio/portfolio_HRP.py` (asignación long-only 1/N,
1/σ, 1/σ² con backtest walk-forward y topología de correlaciones) en una pieza
integrada. Además, sustituir/complementar la σ histórica EWMA por una
**predicción de volatilidad estilo HAR-RV**: predecir la volatilidad del
próximo período de rebalanceo (`H = REBALANCE_FREQ`), con target en
log-volatilidad y features = log-volatilidad de los horizontes `H·1`, `H·3`,
`H·9` (para `H=21`: 21, 63, 189 días).

## Alcance

**Fuera de alcance**: tocar la LSTM/volatilidad existente (sólo se
generaliza `HARRVModel` de forma retrocompatible), publicar en el ledger, o
añadir estrategias nuevas. Se corrigen sólo defectos estructurales.

## Defectos detectados (auditoría del cuaderno)

1. Efectos al importar (`yf.download`, `run_backtest`, `plt.show()`).
2. Estado global (`LAMBDA_EWMA`, `USE_EWMA`, `LOOKBACK`, `REBALANCE_FREQ`,
   `TX_COST`) como defaults.
3. Bug en `plot_market_topology`: `pdist` sobre matriz ya cuadrada; debe ser
   `sch.linkage(squareform(dist), 'single')`.
4. Datos sin validar; 5. claves griegas `'1/σ'`, `'1/σ²'`; 6. duplicación de
   `market_data`/`timeseries`.

## Diseño del modelo de volatilidad HAR-RV

**Target** (por activo): log de la volatilidad realizada forward anualizada a
`H = rebalance_freq` días — reutilizar `thesis.volatility.compute_log_vol_target
(df, horizon=H)`: `y_t = log(sqrt(A/H · Σ_{i=1..H} r²_{t+i}))`, `A=252`.

**Features** (por activo, causales, en log-vol anualizada): trailing realized
vol sobre `H`, `3H`, `9H`:
`log_vol_w = 0.5·log(A · mean(r²_{t-w+1..t}))`. Nuevo helper reutilizable.

**Regresión** (por activo, OLS con errores HAC):
`y_t = β₀ + β₁·log_vol_H + β₂·log_vol_3H + β₃·log_vol_9H + ε`,
prediciendo `log σ̂`; el σ del portfolio es `exp(log σ̂)`.

**Causalidad del ajuste**: en cada rebalanceo `t`, ajustar el HAR-RV de cada
activo con pares `(features_s, y_s)` cuyo target está **totalmente realizado**
en `t-1` (i.e. `s+H ≤ t-1`); predecir en `t-1` y aplicar a `t`. Sin
look-ahead. Alternativa documentada: split fijo train/test (coeficientes
constantes OOS).

## Estructura propuesta

```
src/thesis/portfolio.py              ← nuevo (pesos + backtest + métricas + plots)
src/thesis/volatility.py             ← + helper log_trailing_realized_vol
src/thesis/volatility_models.py      ← HARRVModel con feature_cols configurable
notebooks/portfolio/portfolio_HRP.py ← orquestador py:percent (reescribir)
notebooks/portfolio/README.md        ← documentación
tests/test_portfolio.py              ← unitarios sintéticos (+ HAR-RV)
tests/test_volatility_models.py      ← + caso HAR con columnas custom (si procede)
README.md (raíz)                     ← estructura
```

## Archivos y funciones

### `src/thesis/volatility.py` (modificar)

- `log_trailing_realized_vol(log_ret, window, annualize=252) -> pd.Series`
  — `0.5·log(annualize·(r²).rolling(window).mean().clip(1e-12))`; causal.

### `src/thesis/volatility_models.py` (modificar)

- `HARRVModel.__init__(feature_cols=None)`: `feature_cols` por instancia,
  default `["rv_daily","rv_weekly","rv_monthly"]` (retrocompatible); `fit` y
  `predict` usan `self.feature_cols`. Sin cambios en el notebook de
  volatilidad.

### `src/thesis/portfolio.py` (nuevo, sin estado global)

- `ewma_weights(n, lambda_=0.94)`, `estimate_sigma(returns_df, use_ewma=True,
  lambda_=0.94)` (baseline histórico, conservar corrección `1/(1-Σw²)`).
- `build_har_rv_features(log_rets_df, rebalance_freq) -> dict[str, pd.DataFrame]`
  — por activo, columnas `rv_h/rv_3h/rv_9h` en log-vol.
- `estimate_sigma_har_rv(log_rets_df, rebalance_freq, ...) -> pd.Series`
  — ajusta un `HARRVModel` por activo (expanding hasta `t-1-H`), predice
  `σ̂ = exp(log σ̂)` para el rebalanceo actual.
- `build_portfolios(sigma) -> {'1/N','1/vol','1/var'}` (valida σ>0/NaN).
- `run_backtest(returns_df, lookback=252, rebalance_freq=21, tx_cost=0.005,
  volatility_model="ewma", use_ewma=True, lambda_=0.94, ...)` — `volatility_model
  ∈ {"ewma","har_rv"}` selecciona la fuente de σ; walk-forward causal y deriva
  buy&hold sin cambios.
- `calculate_metrics(...)`; plots parametrizados `plot_equity_drawdown`,
  `plot_portfolio_composition`, `plot_avg_weights`, `plot_market_topology`
  (`squareform`+`linkage`).

**Export**: no en `__init__.py`; importar `from thesis.portfolio import ...`.

### `notebooks/portfolio/portfolio_HRP.py` (orquestador)

Celda por etapa: import+reload, `CONFIG` (tickers, `REBALANCE_FREQ`,
`TX_COST`, `volatility_model`), datos (`market_data.load_multiple_prices` →
`dropna` → log-retornos), features HAR + ajuste, `run_backtest` con σ EWMA y
HAR-RV, métricas, **una celda por figura**. Guardas `if __name__ == "__main__":`.

### `notebooks/portfolio/README.md`

Pregunta central, pipeline, API, fórmulas (EWMA con pesos de fiabilidad;
HAR-RV en log-vol; 1/σ² = IVP sin correlaciones), ejecución, artefactos,
precauciones (look-ahead del ajuste HAR, `H·9` de warm-up, λ=0.94, tx_cost).

### `tests/test_portfolio.py` (sintéticos, sin red)

- `ewma_weights`/`estimate_sigma`/`build_portfolios`/`calculate_metrics`
  invariantes (como plan original).
- `log_trailing_realized_vol`: causal (corromper cola no altera filas previas),
  ventana de warm-up NaN.
- `build_har_rv_features`: columnas y horizonte correctos; sin look-ahead.
- `estimate_sigma_har_rv`: **truncamiento** — corromper la cola de retornos
  futuros y comprobar que los σ̂ anteriores no cambian; coincide con OLS
  manual sobre un activo sintético de 2-3 features.
- `run_backtest(volatility_model="har_rv")`: formas/índice, suma de pesos=1,
  turnover=Σ|w_target−w_drift|, y test de truncamiento end-to-end.
- (Opcional) integración `@pytest.mark.integration`: 3-4 tickers, ventana corta.

## Criterios de aceptación

- `python3 -m py_compile` de los módulos y del cuaderno OK.
- `uv run pytest` en verde (incluidos los tests HAR-RV de truncamiento).
- Importar `thesis.portfolio`/`thesis.volatility` **no** descarga ni dibuja.
- `HARRVModel` retrocompatible: `test_volatility_models.py` sigue en verde.
- Backtest HAR-RV causal: corromper el futuro no altera pesos/σ̂ previos.
- `uv run ruff check` y `uv run ruff format --check` limpios.

## Riesgos principales

- **Look-ahead en el ajuste HAR-RV**: el riesgo clave. Mitigado por ajuste
  con targets totalmente realizados (`s+H ≤ t-1`) + test de truncamiento.
- **Warm-up**: `9H` (189 d) + `H` (21 d) de historia mínima; los primeros
  rebalanceos pueden carecer de σ̂ válido → documentar y fallar/omitir
  explícitamente.
- **Coste del refit**: OLS por activo en cada rebalanceo (21 activos × ~20/año)
  es barato (3 features) pero O(T·N); aceptable, alternativa split fijo.
- **Fidelidad numérica**: conservar `1/(1-Σw²)` y deriva `w(1+R)/(1+gross)`;
  tests por invariantes, no valores exactos.
- **Red (yfinance)**: tests unitarios sintéticos; integración marcada.

## Pasos de implementación (sesión `/exec`)

1. `volatility.py`: añadir `log_trailing_realized_vol`.
2. `volatility_models.py`: generalizar `HARRVModel` (`feature_cols`).
3. Crear `src/thesis/portfolio.py` (motor EWMA + HAR-RV + métricas + plots).
4. Reescribir `notebooks/portfolio/portfolio_HRP.py` como orquestador.
5. `tests/test_portfolio.py` (+ caso HAR en `test_volatility_models.py` si
   procede).
6. `notebooks/portfolio/README.md` + `README.md` raíz.
7. Verificar (`py_compile`, `pytest`, `ruff`), diffs y preguntar por el commit.
