# Correlación momentum–retorno futuro simétrica en τ con bootstrap nulo

## Alcance

Agregar a `notebooks/feature_analysis/momentum_analysis.py` **una** figura con dos
elementos sobre el mismo eje:

1. **Curva observada**: correlación de Pearson entre momentum y retorno futuro en
   función de la ventana **simétrica** τ. Para cada τ el momentum es la suma de los
   `h = τ/2` retornos logarítmicos *pasados* y el objetivo es la suma de los `h`
   retornos *futuros* (ambos excluyen el día `t`; ventanas disjuntas ⇒ sin look-ahead).
2. **Bandas del 95 %** de un bootstrap **i.i.d. sobre los log-retornos**: en cada
   réplica se remuestrea `log_ret` con reemplazo y se recomputa la curva de
   correlación sobre la serie remuestreada; se grafica el sobre percentil
   `[q_{2.5}, q_{97.5}]` por τ. Es la distribución **nula** (sin dependencia
   temporal), centrada ≈ 0.

El cómputo vive en `src/thesis/correlation.py` (reutilizable y testeable); la celda
del cuaderno solo llama y grafica.

## Archivos y funciones a tocar

### 1. `src/thesis/correlation.py` — función nueva

```python
def bootstrap_momentum_forward_correlation(
    log_ret: pd.Series,
    half_windows: Sequence[int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(point, ci_low, ci_high) por half_window h; la ventana simétrica es τ = 2h."""
```

- Requiere `log_ret` 1-D sin NaN; `half_windows` enteros ≥ 1.
- Para un vector `r` y un `h`:
  - `momentum = pd.Series(r).rolling(h).sum().shift(1)`  (retornos `t-h..t-1`)
  - `fwd     = pd.Series(r).rolling(h).sum().shift(-h)`  (retornos `t+1..t+h`)
  - `ρ = pearsonr(momentum, fwd)` sobre índices válidos (`np.corrcoef` sobre la
    máscara de posiciones no-NaN; devolver `NaN` si <3 puntos o varianza 0).
- `point[h]` con `r = log_ret.values`; luego `n_bootstrap` réplicas con
  `rng = np.random.default_rng(seed)` e índices `rng.integers(0, n, size=n)`
  (`seed` por defecto `thesis.config.SEED`, como el resto del módulo).
- `ci_low/ci_high[h] = percentiles alpha/2 y 1-alpha/2` de las réplicas.
- Vectorizar por columna (arrays `(n_bootstrap, len(half_windows))`); no usar
  `np.arange` con paso flotante (usar `np.linspace`/índices enteros en el grid).

### 2. `notebooks/feature_analysis/momentum_analysis.py` — celda nueva

- Importar `bootstrap_momentum_forward_correlation` desde `thesis.correlation`.
- Grid sugerido: `half_windows = np.arange(5, TRADING_DAYS + 1, 5)` ⇒ τ = 2h
  hasta ~2 años; eje X en años `tau = 2 * half_windows / TRADING_DAYS`.
- Llamar con `log_ret`, `n_bootstrap=1000`, `seed=SEED`.
- Graficar (matplotlib, estilo `PLOT_RC` ya aplicado):
  - `ax.plot(tau, point, label="Observado")`
  - `ax.fill_between(tau, ci_low, ci_high, alpha=0.3, label="IC 95 % (bootstrap i.i.d.)")`
  - `ax.axhline(0, color="gray", lw=0.8, ls="--")`
  - ejes `τ (años)` / `Correlación de Pearson`, título descriptivo, `plt.tight_layout()`,
    `plt.show()`.
- Añadir markdown breve justificando que el bootstrap i.i.d. destruye la
  dependencia temporal ⇒ banda nula (misma motivación que
  `spy-acf-mbb-confidence-bands.md`).

### 3. `tests/test_correlation.py` — archivo nuevo (no existía)

- `test_shapes_and_reproducibility`: formas `(len(half_windows),)`; dos corridas con
  igual `seed` idénticas.
- `test_white_noise_null_centered`: ruido blanco i.i.d. (semilla fija); el `point`
  dentro de `[ci_low, ci_high]` en la mayoría de los τ y banda ≈ 0.
- `test_invalid_args`: `half_windows` vacío/≤0, `n_bootstrap < 1`, `alpha` fuera de (0,1).

## Criterios de aceptación

- `uv run pytest` (suite default) en verde, incluido `tests/test_correlation.py`.
- `uv run ruff check src tests` sin errores.
- `uv run python notebooks/feature_analysis/momentum_analysis.py` corre de punta a
  punta y muestra la figura con la curva observada y las bandas del 95 %.
- Sin look-ahead: momentum usa `t-h..t-1`, objetivo `t+1..t+h` (disjuntos); semilla
  fija para reproducibilidad.

## Riesgos

- **Costo computacional**: `1000` réplicas × grid × `rolling` sobre ~8 000 obs es
  manejable (segundos); si molesta, bajar `n_bootstrap` a ~500.
- **Overlap**: se usan todas las posiciones válidas (ventanas solapadas) en la curva
  observada y en el bootstrap por igual; si se exige inferencia más conservadora,
  submuestrear con paso `h` (opción futura, fuera de alcance).
- **τ impar**: τ se define como `2h` (h entero) para que `τ/2` sea exacto; no hay
  ambigüedad de redondeo.
- **Semántica de la banda**: es un null i.i.d., no una banda de incertidumbre del
  estimador bajo dependencia real (subestima el ancho en series heterocedásticas);
  documentarlo en el markdown de la celda.
