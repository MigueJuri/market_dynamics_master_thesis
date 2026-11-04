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
bash scripts/nb_to_py.sh [dir_destino]
```

## 3. Reglas de datos

- Nunca leer datos crudos grandes ni el JSON de `.ipynb` completo.
- Inspeccionar con `scripts/df_peek.py` o `head -n <archivo>`.
- Los notebooks se leen vía `scripts/nb_to_py.sh` (`py:percent`) si es necesario.

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

## 6. Tabla de enrutamiento de modelos

| Agente      | Modelo                           | Cuándo usarlo                     |
|-------------|----------------------------------|-----------------------------------|
| `build`     | `opencode-go/minimax-m3`         | default; ejecutar un plan y tests |
| `planner`   | `opencode-go/deepseek-v4-pro`    | análisis y planificación (escribe `docs/plans/`) |
| `explore`   | `opencode-go/mimo-v2.6-flash`    | localizar definiciones            |
| `reviewer`  | `opencode-go/kimi-k3.1-pro`      | revisión de rigor y leakage       |
| `small_model` | `opencode-go/deepseek-v4.1-flash`| títulos y tareas livianas         |

`plan` (agente nativo de opencode) queda disponible en modo solo lectura: no
escribe nada. Para planificar y persistir el plan se usa `planner`.

DeepSeek requiere región *Global* habilitada en Console → Privacy.

## 7. Reglas de tokens/sesiones

- Una tarea por sesión; planear con `/plan <tarea>` (`planner`).
- Implementar en una **sesión nueva** con `/exec <plan-file>` (`build`).
- No cambiar de modelo en medio de sesión.
- Compactar contexto antes de ~80K tokens; limitar salida de herramientas.
- Pedir diffs, no reescrituras completas de archivos.
- Si algo excede el tamaño de la sesión, delegar a una skill o a `docs/ai/`.

## 8. Git y seguridad

- Diffs visibles antes de continuar; commits agrupados por unidad lógica.
- No commitear secretos, tokens, claves ni archivos de configuración sensibles.
- Preguntar antes de `git push` en cambios grandes; nunca `push --force`.
- Antes de operaciones destructivas, advertir y pedir confirmación.
