# Instrucciones de trabajo

Repositorio de código cuantitativo para tesis de física/finanzas: análisis de LOB,
volatilidad (HAR-RV, LSTM), backtests de momentum y *vol-timing*, y hechos
estilizados de SPY. El texto de la tesis vive en otro repo; no lo leo ni edito.

## 1. Mapa del proyecto

- `src/thesis/` — módulos de mercado, features, volatilidad, modelos, backtest,
  pares, correlación, series temporales, eventos, visualización y config.
- `tests/` — suite con `pytest`; tests de integración con datos reales marcados.
- `notebooks/` — orquestación y experimentos; `.ipynb` se ignora en Git.
- `data/` y `outputs/` — datos y artefactos generados; **nunca** se versionan.

## 2. Comandos habituales

```bash
source .venv/bin/activate        # o usa `uv run ...`
uv run pytest                    # suite por defecto (sin integración)
uv run pytest -m integration     # tests con red/datos reales
uv run ruff check src tests
python3 scripts/df_peek.py <ruta.csv|parquet>
```

## 3. Reglas de datos

- Nunca leer datos crudos grandes completos (`data/`, `.csv`, `.parquet`, etc.).
- Inspeccionar con `scripts/df_peek.py` o `head -n <archivo>`.
- Los notebooks son `.py` en formato `py:percent`; leer solo las celdas necesarias.

## 4. Reglas de código cuantitativo

- **Sin look-ahead**: las señales usan datos ≤ `t-1`; estadísticas rodantes con
  `shift(1)`; normalización se ajusta solo en train.
- **Splits ordenados en el tiempo**; semillas fijas; vectorizar con pandas/numpy.
- Usar `np.linspace` o grillas con índices enteros; evitar `np.arange` con paso
  flotante.
- Manejar valores centinela del LOB (0, -9999999999, etc.) de forma explícita.
- Documentar unidades; DataFrames ordenados (una variable por columna).

## 5. Salidas

- Nombres descriptivos y estables: `<estrategia>_<dataset>_<metrica>.pdf/png`.
- Cada figura/tabla debe ser producida por un único script o celda.
- **No** crear carpetas de exportación ni scripts de sincronización entre repos.

## 6. Flujos de trabajo

| Flujo | Cómo |
|-------|------|
| Planificar | `/write-plan <slug> <tarea>` → escribe `docs/plans/<slug>.md`, no implementa |
| Ejecutar un plan | `/exec docs/plans/<slug>.md` → implementa, testea, commit + push |
| Cambio pequeño | `/quick <cambio>` → un único cambio puntual, verificación mínima |
| Integrar un archivo externo | skill `integrate-existing-file` (se activa al pedir integrar un script/cuaderno) |

Para localizar código en muchos archivos, usar el subagente `Explore`.

## 7. Reglas de tokens/sesiones

- Una tarea por sesión. Planificar con `/write-plan` e implementar en una
  sesión nueva (`/clear`) con `/exec <plan-file>`: el plan lleva el contexto.
- `/compact` antes de ~80K tokens; limitar la salida de herramientas
  (`rg -n`, `head`, rangos de líneas, `df_peek.py`).
- Ediciones con diffs puntuales, no reescrituras completas de archivos.

## 8. Git y seguridad

- **Backup automático**: al cerrar cada unidad de trabajo, hacer `git add` de
  los archivos intencionados, un `git commit` con mensaje conciso (estilo del
  repo: imperativo, prefijo de área si aplica) y `git push` a `origin`
  inmediatamente después, sin pedir confirmación.
- Diffs visibles antes de continuar; commits agrupados por unidad lógica.
- No commitear secretos, tokens, claves ni archivos de configuración sensibles;
  respetar `.gitignore` (`data/`, `outputs/`, `.env*`, `*.ipynb`, artefactos).
- Nunca `push --force`; no mergear ramas ni crear PRs sin pedirlo.
- Antes de operaciones destructivas, advertir y pedir confirmación.
