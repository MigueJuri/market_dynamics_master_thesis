# Reconstruir bandas de confianza del ACF con *moving block bootstrap*

> **Actualización (2026-11-13) — decisión del usuario**: las bandas de la ACF
> (celda 6) se calculan con el **bootstrap i.i.d. simple**, no con el *moving
> block bootstrap*.  `block_bootstrap_acf` se reemplaza por `bootstrap_acf`
> (resampling con reemplazo de observaciones individuales, `block_size` deja de
> existir) y se agrega `bootstrap_ci`, su contraparte i.i.d. de
> `block_bootstrap_ci`.  El *moving block bootstrap* **se mantiene** para los
> momentos de la celda 5 (`block_bootstrap_ci`, sin cambios).
>
> Consecuencia metodológica: el remuestreo i.i.d. destruye la dependencia
> temporal, de modo que la banda pasa a ser la distribución **nula** del
> estimador (equivalente empírico de la banda de Bartlett) y la semetrización
> cambia a la envolvente del intervalo percentil alrededor de cero,
> `h = max(|q_{α/2}|, |q_{1-α/2}|)`.  La banda subestima la incertidumbre real
> en series heterocedásticas (retornos) y persistentes (retornos²); queda
> documentado en el markdown de la celda 6 y en el README del cuaderno.

## Alcance

Reemplazar **solo** las bandas de confianza del ACF (celda 6 de
`notebooks/spy/SPY_caracterizacion.py`):
- retornos: hoy banda robusta de Diebold (1986) → banda *moving block bootstrap*;
- retornos²: hoy banda de ruido blanco de Bartlett ±z/√T → banda *moving block bootstrap*.

Forma elegida: **banda simétrica** (±hₖ) alrededor de 0, compatible con el
`fill_between(-ci, ci)` y el conteo `|ρ̂ₖ| > banda` existentes. No se tocan la
celda 5 (momentos, ya usa MBB), la celda 9 (±kσ) ni las funciones `sample_acf`
/ `heteroskedastic_robust_acf` (siguen existiendo y con sus tests).

## Archivos y funciones a tocar

### 1. `src/thesis/characterization.py` (nueva función)

Agregar `block_bootstrap_acf(...)` (el cómputo vive en `src/`, no en el cuaderno):

```python
def block_bootstrap_acf(
    x, max_lags=50, block_size=10, reps=3_000, alpha=0.05, seed=42,
) -> tuple[np.ndarray, np.ndarray]:
    # (acf, ci_halfwidth), ambos de longitud max_lags+1
```

Implementación:
- `point, interval = block_bootstrap_ci(x, lambda v: sample_acf(v, max_lags), block_size, reps, alpha, seed)`.
- `interval` tiene forma `(2, max_lags+1)` (percentil por lag, vectorizado vía
  `MovingBlockBootstrap.conf_int`, ya soportado por `block_bootstrap_ci`).
- Simetrizar: `half = np.maximum(point - interval[0], interval[1] - point)`;
  forzar `half[0] = 0.0` y `point[0] = 1.0`.
- Devolver `(point, half)`.

### 2. `notebooks/spy/SPY_caracterizacion.py`

- **Import (líneas 116–129)**: agregar `block_bootstrap_acf`; quitar
  `heteroskedastic_robust_acf` y `sample_acf` (quedan sin uso en el cuaderno).
- **`CONFIG` (líneas 146–151)**: agregar `"block_size_acf": 50` (bloque ≥ lag
  máximo 50 para estimar ρₖ en lags altos; el `block_size: 10` de momentos se
  mantiene aparte). Opcional `"n_boot_acf"` si 3 000 réplicas resultan lentas.
- **Markdown celda 6 (líneas 476–500)**: reescribir el párrafo "Sobre las
  bandas de confianza": justificar MBB (preserva dependencia, igual motivación
  que celda 5), tamaño de bloque y que lag 0 no tiene banda.
- **Código celda 6 (líneas 502–565)**:
  ```python
  acf_r, ci_r = block_bootstrap_acf(log_ret.values, max_lags=max_lags,
      block_size=CONFIG["block_size_acf"], reps=CONFIG["n_boot"],
      alpha=alpha, seed=CONFIG["seed"])
  acf_sq, ci_sq = block_bootstrap_acf(log_ret.values**2, max_lags=max_lags,
      block_size=CONFIG["block_size_acf"], reps=CONFIG["n_boot"],
      alpha=alpha, seed=CONFIG["seed"])
  ```
  - `fill_between(-ci_r, ci_r, ...)` con label `IC {1-alpha:.0%} (moving block bootstrap)`.
  - Segundo panel: `fill_between(-ci_sq, ci_sq, ...)`.
  - Ajustar los dos `print` de "fuera de la banda" a `ci_r`/`ci_sq`.
- **Celda 10 (líneas 816–823)**: pasar las bandas nuevas para mantener la tabla
  coherente:
  ```python
  summary = build_summary_table(log_ret, ticker=..., alpha=..., max_lags=...,
      acf_returns=(acf_r, ci_r), acf_squared=acf_sq, squared_margin=ci_sq)
  ```
  `build_summary_table` **no** se modifica: ya acepta `acf_returns` (tuple) y
  `squared_margin` (banda simétrica 1-D). La celda 10 pasa a depender de las
  variables de la celda 6 (igual que hoy depende de `log_ret` de la celda 2).

### 3. `tests/test_characterization.py`

- `test_block_bootstrap_acf_shapes_and_reproducibility`: formas
  `(max_lags+1,)`, `ci[0]==0`, `acf[0]==1`, `ci[1:]>=0`; dos corridas con la
  misma semilla idénticas.
- `test_block_bootstrap_acf_white_noise_within_band`: ruido blanco con `n`
  y `reps` chicos (rápido); fracción de lags fuera de banda ≤ umbral holgado.
- `test_block_bootstrap_acf_rejects_invalid_args`: `max_lags >= n`,
  `block_size<=0`, `alpha` inválido.

## Criterios de aceptación

- `uv run pytest tests/test_characterization.py` en verde (incluye los tests nuevos).
- `uv run ruff check src tests` sin errores.
- `uv run python notebooks/spy/SPY_caracterizacion.py` corre de punta a punta y
  regenera `outputs/spy_acf_comparison.png` con bandas bootstrap (lag 0 sin banda).
- La celda 10 imprime "Lags significativos" coherente con las bandas de la celda 6.
- Sin cambios en las salidas de celdas 3, 4, 5, 7, 8, 9.

## Riesgos

- **Tamaño de bloque vs. lag máximo**: con `block_size=10` las bandas de lags
  altos (≈50) serían poco fiables; se usa `block_size_acf=50` (≥ `max_lags`).
  Validar sensibilidad visualmente.
- **Costo computacional**: 2 × (3 000 réplicas × `np.correlate` sobre ~8 000
  puntos) ≈ decenas de segundos. Si molesta, bajar `n_boot_acf` a ~1 000–2 000.
- **Semántica de la banda en retornos²**: la banda simetrizada es la dispersión
  del estimador alrededor de ρ̂ₖ (que es >0); el conteo `|ρ̂ₖ|>hₖ` sigue siendo
  válido como "incertidumbre", pero no es un test de hipótesis estricto vs 0.
  Documentarlo en el markdown.
- **Dependencia entre celdas**: celda 10 requiere que celda 6 haya corrido
  (NameError si se re-ejecuta sola tras reiniciar el kernel); aceptable, igual
  que la dependencia existente con `log_ret`.
