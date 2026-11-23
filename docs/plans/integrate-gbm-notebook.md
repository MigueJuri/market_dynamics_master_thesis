# Plan: Integrar `notebooks/GeometricBrownianMotion.py` en `notebooks/spy/`

## Objetivo

Mover el cuaderno suelto `notebooks/GeometricBrownianMotion.py` (test del
GBM de parámetros constantes contra SPY) al dominio `spy/` y convertirlo en
una pieza integrada: lógica reutilizable en `src/thesis/gbm.py`, orquestador
por etapas en `notebooks/spy/GeometricBrownianMotion.py`, tests y documentación.
Sin cambiar la semántica numérica.

## Alcance / no-objetivos

**Incluye**: calibración MLE del GBM (drift geométrico, σ, drift aritmético),
bandas de cuantiles exactas (Lema de Itô), fan chart de densidad, cobertura
empírica, ventana móvil de μ/σ y correlación drift-volatilidad (apalancamiento).

**Fuera de alcance**: modelos de volatilidad estocástica (ARCH/GARCH, Heston),
predicción, trading, editar la tesis. Se corrigen sólo defectos estructurales,
preservando resultados numéricos.

## Estructura propuesta

```
src/thesis/gbm.py                         ← nuevo módulo de dominio (cómputo, sin matplotlib)
notebooks/spy/GeometricBrownianMotion.py  ← orquestador por etapas (git mv del original + reescritura)
tests/test_gbm.py                         ← tests unitarios sintéticos (sin red)
notebooks/spy/README.md                   ← añadir sección "GBM" y actualizar árbol/lista
```

No se tocan `pyproject.toml` (numpy/pandas/scipy/matplotlib/yfinance/arch ya
declarados) ni `src/thesis/__init__.py` (módulo no exportado, igual que
`characterization.py`/`orderbook.py`; import con `from thesis.gbm import ...`).

## Defectos detectados (auditoría)

1. **Efectos al importar**: `load_prices()` (descarga/caché), `np.random.seed`,
   `plt.rcParams.update` y `plt.show()` a nivel de módulo.
2. **Duplicación**: `load_prices` ≈ `market_data.load_single_price`;
   `block_bootstrap_ci` ≈ `characterization.block_bootstrap_ci`.
3. **Estado global**: `TRADING_DAYS`, `BLOCK_SIZE`, `N_BOOT`, `ALPHA`,
   `BAND_LEVELS` leídos del ámbito global dentro de las funciones.
4. **Reproducibilidad frágil**: `np.random.seed(SEED)` no siembra el RNG de
   `arch`; usar `block_bootstrap_ci(..., seed=...)` explícito.
5. **Figuras sin guardar**: sólo `plt.show()`; añadir `savefig` con nombres
   estables y rutas ancladas a la raíz.

## Archivos y funciones

### `src/thesis/gbm.py` (nuevo)

- `BAND_LEVELS = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)`.
- `gbm_mle_params(x, trading_days=252) -> np.ndarray`: `[drift_geo, sigma, mu]`
  anualizados; `mu = drift_geo + sigma²/2` (corrección de Itô). Validación
  fail-fast (finito, ≥2 obs, `trading_days>0`).
- `gbm_quantile_bands(S0, drift_geo, sigma, t_grid, levels=BAND_LEVELS) -> dict`:
  cuantiles exactos `exp(norm.ppf(q, loc=log S0 + drift_geo·t, scale=σ√t))`;
  guarda `σ√t→1e-12` en t=0 (delta de Dirac). Valida `S0>0`, `sigma≥0`, `t_grid≥0`.
- `gbm_density_grid(S0, drift_geo, sigma, t_grid, n_bins=200) -> (log_grid, density)`:
  densidad lognormal por fila normalizada por su máximo (cómputo puro del heatmap;
  el enmascarado y `pcolormesh` viven en el cuaderno).
- `gbm_coverage(test_price, bands, levels=(0.5, 0.8, 0.95)) -> pd.DataFrame`:
  fracción de días OOS dentro de la banda central (50/80/95 %), numérica.
- `rolling_gbm_params(log_ret, window=60, trading_days=252) -> pd.DataFrame`:
  columnas `drift_geo`, `sigma`, `mu` anualizados en ventana móvil (causal).

Reutiliza `market_data.load_single_price`, `characterization.block_bootstrap_ci`
y `config` (`TICKER/START/END/DATA_DIR/TRADING_DAYS/SEED`).

### `notebooks/spy/GeometricBrownianMotion.py` (orquestador, `py:percent`)

Celdas por etapa (convención del repo); el dibujo vive aquí:
0. autoreload (`get_ipython()` guard, `_INTERACTIVE`).
1. path bootstrap (`_HERE/_ROOT/_SRC`) + imports + `CONFIG`
   (`SPLIT_DATE`, `ROLLING_WINDOW`, `BLOCK_SIZE`, `N_BOOT`, `ALPHA`,
   `BAND_LEVELS`, `SEED`, rutas de salida) + `plt.rcParams` de estilo.
2. datos: `load_single_price` + log-retornos + split `asof(SPLIT_DATE)`.
3. calibración: `block_bootstrap_ci(train_ret, gbm_mle_params, seed=...)`.
4. fan chart: `gbm_density_grid` + `gbm_quantile_bands` + heatmap
   (`savefig outputs/spy_gbm_fan_chart.png`).
5. cobertura: `gbm_coverage` (tabla).
6. ventana móvil μ/σ: `rolling_gbm_params` + figura
   (`outputs/spy_gbm_rolling_params.png`).
7. drift vs. σ: scatter/hexbin + Pearson (`outputs/spy_gbm_drift_vs_sigma.png`).
8. tabla resumen + conclusiones.

### `notebooks/spy/README.md`

Actualizar la intro ("cuatro cuadernos") y el árbol; añadir sección "GBM
(Movimiento Geométrico Browniano)" con pregunta central, pipeline, API pública
de `src/thesis/gbm.py`, artefactos y precauciones (ver Riesgos).

### `tests/test_gbm.py`

Sintéticos (sin red):
- `gbm_mle_params` recupera parámetros de un GBM simulado (n grande, semilla fija).
- `gbm_quantile_bands`: en `t=0` todos los cuantiles = `S0`; monótonos en `q`
  (t fijo) y en `t` (q fijo, creciente si q>0.5 / decreciente si q<0.5).
- `gbm_density_grid`: filas normalizadas (máx=1), forma `(n_bins, n_t)`.
- `gbm_coverage`: sobre un GBM sintético, cobertura ≈ nominal (tolerancia).
- `rolling_gbm_params`: causalidad (corromper la cola no altera filas previas).
- Validación: entradas vacías/no finitas, `trading_days≤0`, `S0≤0`, `sigma<0`,
  `t_grid<0`, `levels` fuera de (0,1).
- (Opcional) `@pytest.mark.integration` spot-check con SPY (omitido por defecto).

## Criterios de aceptación

- `python3 -m py_compile src/thesis/gbm.py notebooks/spy/GeometricBrownianMotion.py`.
- `uv run pytest` en verde, incl. `tests/test_gbm.py`.
- `from thesis import gbm` **no** descarga datos, dibuja ni fija `rcParams`.
- Resultados numéricos equivalentes al original sobre el mismo input
  (drift_geo/σ/μ, bandas, cobertura).
- `uv run ruff check src tests` y `ruff format --check` limpios.
- Sin rutas hardcodeadas; `git mv` preserva el historial del cuaderno.

## Riesgos principales

- **Look-ahead**: la calibración usa sólo `train_ret` (≤ `SPLIT_DATE`) y las
  bandas se proyectan OOS → correcto. Las estadísticas de ventana móvil
  (secciones 6-7) son **descriptivas** sobre toda la muestra: documentar que
  no son señales (igual que el cuantil 90 en `characterization`).
- **Numérico**: en `t=0` la banda colapsa a una delta (`σ√t→0`); conservar las
  guardas `1e-12`/`1e-8`. Usar `np.arange` entero para `t_grid` (evitar paso
  flotante). No invertir el orden de filtrado de centinelas (no aplica aquí).
- **Cobertura vs. nominal**: verificar el mapeo `BAND_LEVELS` ↔ nominal
  (0.25/0.75→50 %, 0.10/0.90→80 %, 0.025/0.975→95 %).
- **Semilla**: `characterization.block_bootstrap_ci` siembra `arch` con `seed`
  explícito; el original dependía del `np.random.seed` global → el IC puede
  variar levemente. Aceptable (más reproducible); documentar.
- **Red**: primera ejecución requiere `yfinance`; caché en `data/` (gitignored).
  Los tests de integración deben marcarse y omitirse por defecto.

## Pasos de implementación (sesión `/exec`)

1. `git mv notebooks/GeometricBrownianMotion.py notebooks/spy/`.
2. Crear `src/thesis/gbm.py`.
3. Reescribir `notebooks/spy/GeometricBrownianMotion.py` como orquestador.
4. Escribir `tests/test_gbm.py`.
5. Actualizar `notebooks/spy/README.md` (y `README.md` raíz si enumera módulos).
6. Verificar (`py_compile`, `pytest`, `ruff`), mostrar diffs, preguntar commit.
