# Plan: caracterización general del SPY (retornos y retornos cuadrados)

## Alcance

Integrar `notebooks/SPY_caracterizacion.py` (692 líneas, suelto y sin
versionar: `??`) al repositorio siguiendo la skill `integrate-existing-file`.
El análisis cubre los **hechos estilizados** del SPY sobre **retornos
logarítmicos y retornos al cuadrado**: trayectoria del precio, distribución de
retornos (leptocurtosis, asimetría, ajuste t-Student), momentos con bootstrap
por bloques, ACF de retornos vs. retornos² (bandas robustas Diebold 1986),
clustering de volatilidad (EWMA λ=0.94 + sombreado por cuantil) y bandas
incondicionales.

**Requisito de personalización**: el código debe ser **altamente
personalizable** — poder retocar cualquier figura (colores, bines, spans,
cuantiles, xlims, tamaño) y re-ejecutar **sólo esa celda**, sin reiniciar el
kernel ni re-importar. Por eso **las figuras viven in situ en el cuaderno**
(una celda por figura, código matplotlib directo) y **sólo el cómputo** va a
`src/thesis/characterization.py` (recargable en caliente con `%autoreload 2`).
Se **elimina** el `.py` suelto de `notebooks/` (anti-patrón: dejar original +
copia en `src/`).

## Fase 0 — Preflight (importante)

`git status` muestra WIP **ajeno** (sector composition: `SPY_sectores.py`,
`sector_composition.py`, `test_sector_composition.py` sin trackear; `README.md`,
`pyproject.toml`, `src/thesis/plotting.py`, `pair_trading/*` modificados).
**No** stagear ni commitear esos archivos. De esta tarea sólo:
`notebooks/SPY_caracterizacion.py` (fuente), `notebooks/SPY_caracterizacion.ipynb`
(borrado `D`) y los archivos nuevos. **No tocar `src/thesis/plotting.py`** (ya
está modificado por el WIP de sector composition).

## Fase 1 — Auditoría de la fuente (defectos a corregir)

| Defecto | Corrección |
|---|---|
| `load_prices` duplica `thesis.market_data.load_single_price` | reutilizar `load_single_price` |
| `DATA_DIR = Path("data")` relativo al CWD | `thesis.config.DATA_DIR` |
| `np.random.seed(SEED)` global | `seed` como argumento de `block_bootstrap_ci` |
| `plt.show()`/descarga a nivel de módulo | en celdas `if __name__ == "__main__"`; `src/` sin `plt.show()` |
| bloque duplicado del histograma de `r²` (líneas 540-547, doble `plt.show`) | eliminar |
| `moments.map(...)` sólo-display | mostrar en el cuaderno; lógica en `describe_returns` |

Dependencias ya declaradas (`arch`, `scipy`, `yfinance`): **no** tocar
`pyproject.toml`. EWMA causal (`ewm(adjust=False)` con `x_{t-1}`); cuantil sobre
muestra completa es **descriptivo** (dejarlo en comentario).

## Fase 2 — Estructura

- `src/thesis/characterization.py` (nuevo, **sin** matplotlib): cómputo puro.
- `notebooks/spy/SPY_caracterizacion.py` (nuevo, `py:percent`): orquestador por
  etapas; **las figuras se dibujan aquí, una celda por figura**.
- `notebooks/spy/README.md` y `README.md` raíz (estructura).
- `tests/test_characterization.py` (nuevo).

### `characterization.py` (API propuesta, cómputo puro)

`describe_returns(x, label) -> pd.Series` · `freedman_diaconis_bins(x) -> int`
· `fit_return_distribution(x) -> dict` (normal + t) · `excess_kurtosis(x)`
· `annualized_stats(x, trading_days=252)` · `block_bootstrap_ci(x, func,
block_size, reps, alpha, seed) -> (est, ci)` · `heteroskedastic_robust_acf(
returns, max_lags, alpha) -> (acf, ci_bound)` (Diebold 1986) ·
`sample_acf(x, max_lags)` · `compute_ewma_volatility(log_ret, lambda_ewma,
trading_days) -> pd.Series` · `high_volatility_mask(volatility, quantile,
index) -> pd.Series` · `unconditional_band_excursions(log_ret, ks) ->
pd.DataFrame` · `build_summary_table(...) -> pd.DataFrame`.

## Fase 3 — Cuaderno orquestador (clave de personalización)

1. **Celda 0 — magics (modo dual)** con `get_ipython()` para no romper el modo
   script, activando recarga en caliente de `src/`:
   ```python
   try:
       ip = get_ipython()
       ip.run_line_magic("load_ext", "autoreload")
       ip.run_line_magic("autoreload", "2")
   except NameError:
       pass  # script mode: sin magics
   ```
   Así, editar `characterization.py` se refleja en la siguiente celda **sin
   reiniciar el kernel**.
2. **Celda 1 — `CONFIG` único**: dict con *todos* los parámetros de figuras
   (colores, `smooth_window`, `lambda_ewma`, `quantile`, `max_lags`, `alpha`,
   `bins`, `xlim`, `figsize`, rutas `outputs/`, umbrales). Cambiar un valor y
   re-ejecutar la celda de la figura afectada basta.
3. **Celda 2 — datos**: `load_single_price` + `log_returns` → `price`,
   `log_ret`, `r2` (namespace; no se recalcula en las celdas de figura).
4. **Celdas 3-9 — una celda por figura** (`trayectoria`, `distribución`,
   `bootstrap`, `ACF retornos vs.²`, `histograma de r²`, `regímenes EWMA +
   precio semilog`, `bandas incondicionales`): código matplotlib **directo**,
   lee variables del namespace + `CONFIG`, guarda en `outputs/` y hace
   `plt.show()`. Re-ejecutar una celda no depende de las demás.
5. **Celda 10 — tabla resumen** vía `build_summary_table`.

Cada celda de figura es **idempotente** (depende de variables explícitas, no de
efectos colaterales) y **auto-contenida** en su código de dibujo: no hay
funciones de trazado opacas que obliguen a recargar `src/`.

Figuras en `outputs/`: `spy_price_trajectory.png`, `spy_return_density.png`,
`spy_acf_comparison.png`, `spy_sq_returns_hist.png`, `spy_vol_regimes.png`,
`spy_uncond_bands.png`.

## Fase 4/5 — Documentación y tests

- `notebooks/spy/README.md`: pipeline, fórmulas (EWMA/Diebold), artefactos,
  ejecución, y nota explícita de "cómo retocar una figura: edita su celda y
  re-ejecútala; `%autoreload 2` recarga `characterization.py` al vuelo".
- `README.md` raíz: añadir `characterization.py` al árbol.
- `tests/test_characterization.py` (sintético, sin red):
  - `freedman_diaconis_bins` ≥ 1 y creciente con la dispersión.
  - `describe_returns` gaussiana: skew≈0, exceso curtosis≈0.
  - `sample_acf` ruido blanco dentro de ±z/√n (lag≥1).
  - `heteroskedastic_robust_acf` de ARCH: bandas ≥ Bartlett y ACF de `r²`
    significativa.
  - **Causalidad EWMA**: corromper la cola → pasado idéntico.
  - `high_volatility_mask`: fracción ≈ 1−quantile.
  - `block_bootstrap_ci` devuelve `(est, ci)` con forma correcta.

## Criterios de aceptación

- `uv run python notebooks/spy/SPY_caracterizacion.py` corre completo y genera
  las 6 figuras en `outputs/` (los magics no rompen el modo script).
- En Jupyter, editar una celda de figura (o `CONFIG`) y re-ejecutarla **solo**
  reproduce la figura sin reiniciar el kernel; editar `characterization.py` se
  refleja vía `%autoreload 2`.
- `import thesis.characterization` no descarga ni dibuja.
- Cobertura de **retornos y retornos²** (ACF comparada, distribución con ajuste
  t, histograma de `r²`, regímenes EWMA con precio semilog y cuantil).
- `uv run pytest tests/test_characterization.py` pasa; `ruff check/format`
  limpios; `python3 -m py_compile` sobre los nuevos.
- `notebooks/SPY_caracterizacion.py` eliminado; sin lógica duplicada.

## Riesgos principales

- **Stage de WIP ajeno**: sector composition/pair trading en vuelo; preparar
  sólo los archivos de esta tarea y **no** tocar `src/thesis/plotting.py`.
- **Magics vs modo script**: `%autoreload` es IPython-only; usar la guarda
  `get_ipython()` para que `python notebooks/spy/SPY_caracterizacion.py` siga
  funcionando.
- **Look-ahead del cuantil**: percentil sobre toda la muestra; documentar que es
  descriptivo (no señal).
- **`arch` (MovingBlockBootstrap)**: import pesado, ya declarado; restringir a
  `characterization.py` (no en `__init__.py`).
- **matplotlib fuera del cómputo**: `characterization.py` sin matplotlib; el
  dibujo queda íntegro en el cuaderno.
- **Mover el archivo**: orquestador en `notebooks/spy/`; verificar
  `sys.path` (`_HERE.parents[1]`).
- **Exactitud numérica**: conservar bandas Diebold, Freedman-Diaconis y EWMA
  RiskMetrics sin alterar resultados.
