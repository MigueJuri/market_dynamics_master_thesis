# Plan: Publicar config + métricas desde todos los notebooks al ledger

## Alcance

El mecanismo de publicación (`thesis.results.publish` → adapters en
`results_adapters.py` → `.thesis/results-ledger.md`) ya existe y casi todos los
notebooks tienen una celda final de publicación. El problema: **esas celdas
envían un payload vacío** porque los notebooks no definen `CONFIG` ni
`metrics_df`/`metrics`; las métricas quedan dispersas en `print(...)` y
`savefig(...)`. Este plan hace que cada notebook construya una config
estructurada y un DataFrame de métricas (o dict para LSTM) que alimenten el
adapter correspondiente.

**No-objetivos:** no re-entrenar ni re-ejecutar experimentos; no cambiar la
lógica de cálculo; no escribir adapters por métrica con unidades perfectas (el
adapter genérico marca `candidate` + "revisar unidades", aceptable); no tocar
`Volatilidad_LSTM.py` (ya es el piloto correcto); no tocar `.tex`/figuras.

## Diagnóstico (verificado)

- La celda de publicación en 14 notebooks itera
  `("CONFIG","config","metrics","results","summary","backtest")`:
  - `CONFIG` y `config` colisionan en la clave `config` (inocuo pero redundante).
  - **`metrics_df` nunca se captura** → el adapter genérico recibe payload sin
    `metrics_df` y devuelve `_candidate_record("metrics_df vacío")`.
  - La mayoría de esos notebooks **no define** `CONFIG` ni `metrics_df`.
- `results_adapters.ADAPTERS` tiene entradas para 15 `experiment_id`; el
  genérico (`_generic_dataframe_adapter`) espera `payload["metrics_df"]` +
  `payload["config"]`; el LSTM (`_lstm_daily_adapter`) espera
  `config`/`metrics`(dict model→métricas)/`backtest`/`summary`.
- **Sin celda de publicación ni adapter:** `volatility/Volatilidad_LSTM_Alpaca.py`,
  `volatility/Volatilidad_LSTM_Descripcion.py`, `pair_trading/OUProcess.py`.
- Los notebooks `spy/*`, `btc/BTC.py`, `Volatilidad_LSTM_*.py` usan guardas
  `if __name__ == "__main__":` por etapa (se ejecutan en orden como script o en
  el kernel de Jupyter donde `__name__ == "__main__"`); la celda final debe
  construirse dentro de esa guarda.

## Cambios por archivo

### 1. `src/thesis/results_adapters.py`
- Generalizar `_lstm_daily_adapter` a una factory
  `_lstm_adapter(experiment, code_path, asset_default)` y registrar
  `"volatility/lstm_alpaca"` (misma forma de payload que `lstm_daily`).
- Registrar `"volatility/lstm_description"` y `"pair_trading/ou"` con el adapter
  genérico (capítulos "Volatilidad" y "Pair Trading").
- (Opcional) en `_generic_dataframe_adapter`, usar `config.get("ticker")` ya
  soportado; no cambiar el resto.

### 2. Plantilla estándar de celda de publicación (sustituir la actual en los 14 notebooks)
Reemplazar el bucle `globals()` por un dict explícito que incluya `metrics_df`:
```python
if __name__ == "__main__":
    from thesis.results import publish
    publish(
        experiment_id="<id>",
        payload={"config": CONFIG, "metrics_df": metrics_df},
        chapter="<capítulo>",
    )
```
(En LSTM ya existe y se conserva; añadir `metrics_df` no aplica ahí.)

### 3. Notebooks a modificar (añadir `CONFIG` + `metrics_df`/`metrics`, y celda corregida)

- **feature_analysis/momentum** — `CONFIG` (ticker/start/end, lags, `mom_k`,
  `ret_k`, `n_bootstrap`, `seed`, `alpha`); `metrics_df` de 1 fila con
  `pearson_corr`, máx/mediana de `mom_corr_matrix`, media de `avg_corr`,
  `point/median/ci` bootstrap (valor en τ representativo o resumen).
- **feature_analysis/volatility** — `CONFIG` (ventanas 21/63/252, `vol_window`,
  `target_window`); `metrics_df` con `pearson_corr`, máx de `vol_corr_matrix`,
  media de `avg_corr`.
- **feature_analysis/momentum_volatility** — ídem con `SIGMA_FLOOR`, `tau_p/tau_f`.
- **feature_analysis/moving_average_distance** — `CONFIG` (`WINDOW_TYPE`, lags,
  `tau_p/tau_f`); `metrics_df` con `pearson_corr`, máx de `dist_corr_matrix`,
  media de `avg_corr`.
- **feature_analysis/per** — `CONFIG` (`TAU_F`, variantes, `PE_REFERENCE`,
  `MIN_HORIZON_SESSIONS`); `metrics_df` = `corr_table` ya existe: convertirlo a
  formato ancho (`variante` como columna `model`, métricas `pearson/spearman`
  y variantes `_sub`/`_m`).
- **trend_following/Trend_Following_Strategy_v2** — `CONFIG` (tickers, ventanas
  EWMA, umbrales triple-barrera, `ANNUAL_RF`, `TRADING_DAYS`); `metrics_df` de
  2 filas (Strategy/Buy & Hold) con `total_return`, `ann_return`, `vol`,
  `sharpe`, `maxdd`, `n_trades`, más fila de F1 (`precision/recall/f1`
  ponderado y sin ponderar, `hit_rate`).
- **spy/SPY_caracterizacion** — ya tiene `CONFIG`; añadir `metrics_df` con la
  tabla `summary` de `build_summary_table` + `moments` (media/var/skew/kurt) y
  params de la densidad ajustada (Normal/t).
- **spy/SPY_sectores** — ya tiene `CONFIG`; `metrics_df` con `entropy` medio/
  mín/máx, `hhi` medio/mín/máx, suma de pesos del snapshot.
- **spy/OrderBook** — ya tiene `CONFIG`; `metrics_df` con `max_range`, nº bins
  bid/ask, rango de precio, nº niveles válidos, nº buckets de mid.
- **btc/BTC** — ya tiene `CONFIG`; `metrics_df` con `fit` (α, β/R² por cuantil),
  picos destacados (fecha, resid_ma).
- **pair_trading/1_PairSelectionMutualInformation** — `CONFIG` (min_history,
  `k_neighbors`, `n_clusters`, `SIGNIFICANCIA_ADF`, `MIN_SESIONES_ADF`);
  `metrics_df` con top-MI (top pares), métricas de clustering (`weighted_purity`,
  `normalized_entropy`, `mean_hhi`, `ari`, `ami`, `v_measure`).
- **pair_trading/2_PairSelectionPearson** — ídem con correlaciones top y
  clustering equivalentes.
- **pair_trading/PracticalPairTrading** — `CONFIG` (TICKER_A/B, ventanas,
  `ENTRY_Z/EXIT_Z/COST`, `MAX_ABS_BETA`); `metrics_df`/`summary` con `bt.sharpe`,
  `total_return`, `max_drawdown`, `trades` y la `comparativa` de benchmarks.
- **pair_trading/Spread** — `CONFIG` (TICKER_A/B, START/END); `metrics_df` con
  `hedge_ratio`, OU (`theta/mu/half_life/r_squared`), `z_score`, ADF.
- **pair_trading/OUProcess** — añadir `CONFIG` (T, dt, θ, μ, σ, S0, β, GBM) y
  `metrics_df` mínima (parámetros OU); añadir celda de publicación + adapter
  (`pair_trading/ou`).
- **volatility/Volatilidad_LSTM_Alpaca** — tiene `CONFIG` y `metrics_df`/`backtest`/
  `summary` como retornos de funciones; añadir celda de publicación con
  `payload={"config": CONFIG, "metrics": {"LSTM": metrics["lstm"], "HAR-RV":
  metrics["har"]}, "backtest": backtest["backtest"], "summary": backtest["summary"]}`.
- **volatility/Volatilidad_LSTM_Descripcion** — tiene `CONFIG`; añadir
  `metrics_df` con stats de ACF (primeros lags de retornos y r²) y momentos de
  vol realizada/forward; añadir celda de publicación + adapter.

## Criterios de aceptación

1. Al ejecutar `python3 notebooks/<ruta>.py` (o `uv run`), cada notebook llama
   a `publish()` con un payload que contiene `config` y `metrics_df` (o
   `metrics`/`backtest`/`summary` en LSTM), sin `KeyError`.
2. `publish()` genera ≥1 `ResultRecord` con `metric_value` real (no
   `"adapter_pending"` ni `"metrics_df vacío"`); los 18 notebooks tienen
   `experiment_id` registrado en `ADAPTERS`.
3. `derive_result_id` es estable entre ejecuciones idénticas (segunda ejecución
   no duplica filas).
4. `uv run pytest tests/test_results.py tests/test_results_ledger.py
   tests/test_results_adapters.py` en verde y `ruff check src tests` limpio.
5. No se altera la lógica de cálculo ni las figuras; diffs limitados a
   `CONFIG`/`metrics_df`/celda de publicación + adapters.

## Riesgos principales

- **Captura parcial/heterogénea:** muchas métricas son prints inline sin función
  compartida; el `metrics_df` debe construirse al final referenciando variables
  ya calculadas (sin recalcular). Mitigación: `status=candidate` en adapter
  genérico en vez de inferir unidades.
- **Guardas `__main__`:** en `spy/*` y `btc` las variables de etapa solo existen
  dentro de la guarda; la celda final debe estar en la misma guarda y ejecutarse
  después de las etapas (orden ya garantizado como script/Jupyter).
- **Orden de filas en `metrics_df`:** el adapter genérico usa `fingerprint=
  {"row": int(_), "metric": ...}`; si cambia el orden de filas cambia el
  `result_id` y se duplican filas → construir `metrics_df` con orden estable.
- **`metrics_df` no capturado por la plantilla actual:** es el bug raíz; hay que
  sustituir la plantilla completa (no parchear) en los 14 notebooks.
- **Ledger en este repo:** `LedgerConfig.from_thesis_repo()` resuelve a la raíz
  del proyecto (existe `AGENTS.md`), no al repo de tesis externo; verificar que
  `THESIS_REPO_DIR` apunta donde se espera antes de publicar en producción.
- **Alpaca/Descripcion/OUProcess** eran "excluidos" en el plan anterior; ahora se
  incluyen con métricas mínimas (params de simulación / ACF) — confirmar que es
  lo deseado.
