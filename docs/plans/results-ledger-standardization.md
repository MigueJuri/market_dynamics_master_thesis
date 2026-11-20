# Plan: Ledger de resultados estandarizado (todos los notebooks, por defecto)

## Alcance y no-objetivos

**Objetivo:** mecanismo **por defecto** que, al terminar cualquier notebook canónico,
registre sus métricas de forma idempotente en `<tesis>/.thesis/results-ledger.md`
(fuente que cruza `method-auditor`, subagente de solo lectura). Una única celda/call
al final de cada notebook publica; el registro es validado, deduplicado por
`result_id` y nunca pisa anotaciones manuales.

**No-objetivos:** no re-entrenar ni re-ejecutar experimentos; no extraer salidas
antiguas indiscriminadamente; no publicar notebooks exploratorios/solo-figuras;
no tocar `.tex`, ecuaciones, `\label{}` ni `\cite{}`; no publicar datos crudos,
modelos (`.pt`/`.pkl`), credenciales ni `.env`; no commitear (el plan no implementa).

## Estado de Git (verificado)

- Código: ~11 archivos WIP sin commitear (incl. `Volatilidad_LSTM*.py`). Conservar; no tocar salvo añadir la celda de publicación.
- Tesis: WIP en `.thesis/summaries`, `chapters/*.tex`, `figures/`. Conservar. `.thesis/` sí está versionado.
- Ledger actual: tabla placeholder de 4 columnas (`Capítulo | Métrica | Valor | Fuente`).

## Inventario de notebooks (canónico vs exploratorio)

**Canónicos (publicar):** `volatility/Volatilidad_LSTM.py` (piloto, procedencia OK),
`trend_following_strategy/Trend_Following_Strategy_v2.py`,
`pair_trading/{1_PairSelectionMutualInformation,2_PairSelectionPearson,PracticalPairTrading,Spread}.py`,
`feature_analysis/{momentum,volatility,momentum_volatility,moving_average_distance,per}_analysis.py`,
`btc/BTC.py`, `spy/{SPY_caracterizacion,SPY_sectores,OrderBook}.py`.

**Excluidos (sin métricas o simulación):** `figures/*.py`,
`GeometricBrownianMotion.ipynb`, `pair_trading/OUProcess.py` (simulación),
`volatility/Volatilidad_LSTM_Descripcion.py` (solo figuras descriptivas) y
`volatility/Volatilidad_LSTM_Alpaca.py` (fuera de alcance).

## Archivos y funciones

**Nuevos en `/home/miguel/thesis`:**
- `src/thesis/results.py` — esquema `ResultRecord` + `LedgerConfig` (campos:
  `result_id`, `chapter`, `experiment`, `metric_name/value/unit`, `model`,
  `baseline`, `asset`, `data_source`, `frequency`, `adjustments`, `data_period`,
  `train/val/test dates`, `target_definition`, `horizon`, `sample_size`,
  `protocol`, `code_path`, `execution_date`, `git_rev`, `status`, `artifacts`,
  `notes`). `validate_record()`, `derive_result_id()` (hash estable de
  chapter+experiment+metric+model+asset+fingerprint-config), `redact()`.
  Campos faltantes → `"unknown"` + `status=candidate` + `missing_metadata`.
  **Además: `MetricRegistry` global** con `record()`, `record_many()`,
  `publish()` y `atexit`-hook automático controlado por `THESIS_PUBLISH_RESULTS`
  (default `"1"` en `config.py`): al terminar el proceso publica todo lo
  registrado si hay registro y ruta de tesis resuelta.
- `src/thesis/results_ledger.py` — importador idempotente: parsea el `.md`,
  extrae regiones manuales, merge por `result_id` (dedup), re-renderiza solo el
  bloque gestionado. Sidecar JSON `<tesis>/.thesis/results-ledger.json`.
- `src/thesis/results_adapters.py` — **registro de adaptadores** `ADAPTERS =
  {experiment_id: callable(payload) -> list[ResultRecord]}`. Cada adapter mapea
  la salida estructurada del notebook (dict/DataFrame/dataclass) a records con
  unidades correctas. Se implementan por fases (ver abajo). Los `experiment_id`
  sin adapter → `candidate` + nota "adapter pendiente", no fallan el publish.
- `scripts/publish_result.py` — CLI `python scripts/publish_result.py
  --experiment <id> <payload.json>` (o `--record <record.json>`). Valida,
  calcula `result_id`, invoca importador. Ruta tesis: `--thesis-repo` /
  env `THESIS_REPO_DIR` / `config.py`; sin ruta absoluta embebida.

**Modificados:**
- `src/thesis/config.py` — añadir `THESIS_REPO_DIR: Path` (env, sin default
  hardcodeado) y `PUBLISH_RESULTS_TO_LEDGER: bool = True` (default-on).
- Cada notebook canónico — **una celda final** `if __name__ == "__main__":`
  que agrupa sus métricas en un `payload` (dict/DataFrame/dataclass) y llama
  `thesis.results.publish(experiment_id=..., payload=..., chapter=...)`.
  No se toca la lógica de cálculo. Las métricas que ya retornan funciones
  compartidas (`evaluate_forecasts`, `run_vol_timing_backtest`,
  `evaluate_cluster_sector_alignment`, `pair_trading_backtest`,
  `build_strategy_comparison`, `compute_pair_backtest_diagnostics`,
  `summarize_pair_window`) se reutilizan sin recalcular.
- `<tesis>/.thesis/results-ledger.md` — migrar placeholder a estructura ampliada:
  bloque gestionado `<!-- BEGIN_RESULTS_TABLE -->/<!-- END -->` (regenerado) +
  sección "Anotaciones manuales" (preservada verbatim, referenciada por
  `result_id`).

**Tests:** `tests/test_results.py` (validación, `result_id` estable, rechazo de
secretos, unknown/candidate), `tests/test_results_ledger.py` (segunda publicación
no duplica, no borra anotaciones, migración conserva contenido, legibilidad),
`tests/test_results_adapters.py` (un adapter por familia con payload sintético).

## Formato del ledger (contrato `method-auditor`)

Tabla markdown, una fila por métrica: `result_id | Capítulo | Experimento |
Modelo | Baseline | Métrica | Valor | Unidad | N | Estado | Fuente/Artefacto |
Código+Git | Protocolo`. Conserva las 4 columnas originales como subconjunto.
Bloque gestionado entre comentarios HTML; anotaciones manuales fuera del bloque.

## Fases de implementación (orden)

1. **Core:** `results.py` (esquema + registry), `results_ledger.py`,
   `publish_result.py`, `config.py`, migración del ledger, tests base.
2. **Piloto LSTM diario:** `results_adapters.py` + celda final en
   `Volatilidad_LSTM.py`; validar idempotencia end-to-end.
3. **Extender por defecto:** adaptadores + celda final en el resto de canónicos
   (trend-following, pair-trading, feature-analysis, btc, spy). Los que no tengan
   adapter cerrado se marcan `candidate` explícito.

## Criterios de aceptación

1. `publish()` de cualquier notebook canónico valida y escribe filas sin
   duplicar en una segunda ejecución; anotaciones manuales intactas.
2. `derive_result_id` estable entre ejecuciones para la misma
   (config, métrica, modelo).
3. Ledger migrado legible para `method-auditor` (tabla + columnas originales).
4. `uv run pytest tests/test_results.py tests/test_results_ledger.py
   tests/test_results_adapters.py` en verde (sin red, sin torch) y `ruff check`
   limpio.
5. Comando documentado, sin ruta absoluta embebida; `THESIS_REPO_DIR`/
   `--thesis-repo` resuelven la ruta; sin ruta → aborta con instrucciones.
6. Cada notebook canónico publica con `experiment_id` declarado; los notebooks
   sin adapter cerrado aparecen como `candidate` con `missing_metadata`.

## Riesgos principales

- **Procedencia desfasada:** README LSTM dice `train_end=2016` pero config real
  `2017-01-01`. Los adapters leen solo artefactos/dicts estructurados (no prosa)
  y marcan discrepancias.
- **Leakage/unidades:** vol en espacio sigma anualizada; registrar unidad por
  métrica y protocolo (target log-vol forward 5d, splits 2017/2022, `cost_bps`).
  Correlaciones feature-analysis acotadas [-1,1]; Sharpe con/sin rf según
  notebook (trend-following usa rf=3%; pair-trading sin rf) → documentar en
  `protocol`.
- **Heterogeneidad:** métricas inline (prints) en trend-following/feature-analysis
  no tienen función compartida; el adapter requiere el payload estructurado de la
  celda final. Riesgo de capturar parcialmente → mitigado con `status=candidate`
  + `missing_metadata` explícito en vez de inferir.
- **Integridad del ledger:** importador que reescriba todo borraría anotaciones →
  bloques delimitados + sidecar JSON + merge por `result_id`.
- **Secretos:** redactar/abortar ante `.env`, credenciales, pesos
  (`.pt`/`.pkl`) y rutas `data/` crudas.
- **WIP del usuario:** no tocar archivos sin commitear salvo añadir la celda
  final; limitar diffs a lo necesario.
