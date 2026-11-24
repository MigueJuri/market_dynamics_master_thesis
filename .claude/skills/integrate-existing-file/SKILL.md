---
name: integrate-existing-file
description: Integrates existing scripts, notebooks, or newly added files into this repository by extracting reusable modules into src/, preserving project conventions, adding READMEs and tests, auditing data-flow safety, and validating dependencies. Use when the user asks to integrate, adapt, refactor, modularize, split, or document an existing file, script, notebook, or experiment into the repo structure.
---

# Integrar un archivo existente al repositorio

Procedimiento para tomar un archivo suelto (cuaderno, script o módulo) y
convertirlo en una pieza integrada del repositorio: lógica reutilizable en
`src/`, orquestación ligera en `notebooks/`, documentación y tests.

Las órdenes de este documento asumen la raíz del repositorio como
directorio de trabajo.

## Principio rector

Un archivo integrado no debe quedar como una isla. Al terminar:

- Toda lógica reutilizable vive en `src/<paquete>/` y se **importa**.
- El cuaderno o script es un **orquestador por etapas**: configura, invoca y
  reporta, con una celda por sección del pipeline.
- Cada pieza nueva tiene tests y documentación.
- El árbol de trabajo del usuario queda intacto fuera del alcance.

## Fase 0 — Preflight

Antes de tocar nada:

1. `git status --short` y `git log --oneline -10`. Identifica los cambios
   preexistentes del usuario: **nunca** los incluyas en tu staging ni en el
   commit. Si hay archivos ya preparados (`A`, `M` en la primera columna),
   `git reset HEAD <archivo>` para sacarlos del índice antes de preparar
   los tuyos, y vuelve a comprobarlos al terminar.
2. Lee `CLAUDE.md`, `README.md` y `pyproject.toml`.
3. Explora `src/<paquete>/`: convenciones de nombres, estilo de docstrings,
   tipado, idioma y estructura de los módulos existentes.
4. Revisa `tests/` para copiar el estilo de tests ya presente.
5. Localiza los comandos de verificación **reales** del repo. No los
   asumes: búscalos en `pyproject.toml`, `README.md` o `CLAUDE.md`.

En este repositorio son:

```bash
uv run pytest                                  # suite por defecto
uv run pytest -m integration                   # tests que necesitan red
uv run ruff check src tests
uv run ruff format --check src tests
python3 -m py_compile <archivos>               # 'python' no existe: usa python3
```

Usa `uv run --no-sync` cuando el entorno ya está provisionado: sin él,
`uv` sincroniza y puede reinstalar todo el entorno (incluidos paquetes
grandes) antes de ejecutar el comando.

## Fase 1 — Auditoría del archivo fuente

Clasifica **cada** bloque del archivo por responsabilidad: datos,
features, modelos, entrenamiento, evaluación, visualización, utilidades.

Después busca defectos. El catálogo con señales y correcciones está en
`references/audit-checklist.md`; los más frecuentes en este repo son:

| Defecto | Señal | Corrección |
| --- | --- | --- |
| Efectos al importar | descarga, entrenamiento o `plt.show()` a nivel de módulo | encapsular en una función y proteger con `if __name__ == "__main__":` |
| Estado global | `CONFIG`, `DEVICE`, `sigma_bar` leídos del ámbito global | convertirlos en argumentos explícitos |
| Variable global no definida | usada en un módulo, definida en otro | pasarla como parámetro |
| Orden no determinista | `df[list(un_set)]` | usar tupla o lista explícita |
| Imports muertos | `optuna`, `TimeSeriesSplit` sin uso | eliminarlos |
| Código dentro de docstring | fragmento que rompe la función | sacarlo fuera |
| Dependencia no declarada | import de una librería ausente de `pyproject.toml` | declararla; si el wheel arrastra GB (CUDA), fijar un índice específico |
| Rutas relativas al CWD | `open("model.pt")` | `pathlib.Path` anclado a la raíz del repo |
| Datos sin validar | NaN, precios no positivos, columnas ausentes | validador explícito que falle rápido |
| Posible *look-ahead* | feature que consulta el futuro | test de truncamiento o de corrupción |

## Fase 2 — Decidir la estructura

**Pregunta 1:** ¿el archivo es un cuaderno/script (orquestación) o ya es
biblioteca? Un cuaderno nunca debe ser la única copia de la lógica.

**Pregunta 2:** ¿qué bloques independientes contiene?

Particiona **por responsabilidad, no por función**. Cada módulo debe:

- describirse en una frase;
- caber en el contexto de un agente (≈ 300-500 líneas);
- no depender de estado global.

Heurística:

| Tamaño / complejidad | Estructura |
| --- | --- |
| < 300 líneas, una responsabilidad | 1 módulo |
| 300-800 líneas, 2-3 responsabilidades | 2-3 módulos |
| > 800 líneas o > 3 responsabilidades | dividir por dominio |

Agrupación típica en este repo: `data` + `features` en un módulo,
`models` + `training` en otro, `evaluation` + `plots` en un tercero.

Antes de crear un módulo, **busca funcionalidad ya existente** en
`src/<paquete>/` y reutilízala. Duplicar `log_returns`, `apply_window` o
`load_single_price` es un defecto, no un bug.

Si `src/<paquete>/__init__.py` existe, exporta desde ahí solo los módulos
que **no** arrastren dependencias pesadas u opcionales: importar el paquete
raíz no debe fallar por una dependencia opcional ausente.

## Fase 3 — Implementación

1. Extrae las funciones generales a `src/<paquete>/<dominio>.py`.
2. Haz explícitas las dependencias: nada de variables globales ni de
   imports mutantes.
3. Conserva el comportamiento numérico del original. Si corriges un
   defecto, documéntalo en el commit.
4. Deja el cuaderno como orquestador **por etapas**, en formato `py:percent`
   (jupytext): una celda por sección del pipeline (datos, features, modelo,
   evaluación, figuras), de modo que cada sección pueda re-ejecutarse de
   forma independiente.  Una función `run_experiment()` es sólo un atajo
   programático opcional, nunca la única vía de ejecución del cuaderno.
   Ver «Cuadernos interactivos» más abajo.
5. Actualiza `pyproject.toml` si aparecen dependencias nuevas.
6. Muestra el diff después de cada modificación (ver Fase 6).

### Cuadernos interactivos (`py:percent`)

Un cuaderno de experimento se escribe para que el autor pueda **iterar in
situ**: cambiar un parámetro o una figura y re-ejecutar sólo lo afectado, sin
volver a correr el pipeline entero (y sin reentrenar ni volver a descargar
datos).  Reglas:

- **Formato**: cabecera jupytext `percent` (`# %%` como separador de celdas,
  celdas markdown con títulos `##`), como en
  `notebooks/volatility/Volatilidad_LSTM.py`.  El archivo sigue siendo
  ejecutable como script (`python notebooks/<dominio>/<script>.py`).
- **Una celda por etapa**: cada sección del pipeline (carga de datos,
  preparación, ajuste, evaluación, cada figura) va en su propia celda.  Las
  etapas son funciones en `src/`; la celda las invoca y **deja el resultado en
  el espacio de nombres** (`btc_df`, `reg_df`, `fit`, ...) para poder
  explorarlo y re-ejecutar las celdas siguientes.
- **Ejecución idempotente por celda**: la celda de una etapa debe poder
  re-ejecutarse sin repetir las anteriores.  Si una etapa depende de otra,
  que la dependencia sea explícita (el resultado de la etapa previa vive en
  una variable) y no un efecto colateral oculto.
- **Guardas de script**: las celdas de ejecución se envuelven en
  `if __name__ == "__main__":` si el archivo también se usa como script o
  módulo importable.  En Jupyter la condición es verdadera, de modo que las
  celdas siguen ejecutándose de forma interactiva.
- **`run_experiment()` es opcional**: puede existir como atajo para tests y
  scripts (encadena las etapas y devuelve los resultados), pero no debe
  sustituir a las celdas ni ser la única forma de ejecutar el cuaderno.
- **Figuras in situ**: las figuras del experimento se dibujan en el cuaderno,
  una por celda, para poder personalizarlas sin tocar `src/` ni reiniciar el
  kernel.  Sólo la lógica reutilizable va a `src/<paquete>/<dominio>.py`.

### Personalización sin reiniciar el kernel

**Regla de oro:** cambiar el aspecto de una figura (título, etiquetas,
colores, tamaño, ejes, anotaciones) o un parámetro del experimento **nunca**
debe obligar a reiniciar el kernel, re-entrenar el modelo ni volver a leer los
datos.  Todo lo que se personaliza con frecuencia debe ser un **argumento**,
no una constante escrita dentro del cuerpo de la función.

Reglas:

1. **Cada figura es una función parametrizada.** En `src/<paquete>/<plots>.py`
   (o en el módulo de su dominio), una función por figura, con argumentos
   explícitos para todo lo que se toquen a menudo:

   ```python
   def plot_equity_curve(equity, *, title="Equity curve", ylabel="PnL",
                         colors=None, figsize=(10, 5), ax=None, **kwargs):
       ...
       return fig, ax
   ```

   Reglas de firma: argumentos de estilo con valor por defecto `None` (y
   resueltos dentro con un valor por defecto *documentado*), `ax=None` para
   poder componer en subplots, y `**kwargs` para extender sin romper la API.
   Devuelve siempre `(fig, ax)`; no llames a `plt.show()` dentro de `src/`.
2. **Los parámetros de estilo viven en el cuaderno, no en `src/`.** La celda de
   figura declara sus propios ajustes (p. ej. `STYLE = {...}`,
   `COLOR_UP = "..."`, `figsize=(10, 5)`) y los pasa a la función.  Así el
   usuario ajusta el aspecto editando *una celda* y re-ejecutando *esa celda*.
3. **Estilo global de matplotlib en una celda de estilo**, al principio del
   cuaderno: `plt.style.use("seaborn-v0_8-whitegrid")` y/o
   `plt.rcParams.update({...})`.  Nunca fijes `rcParams` a nivel de módulo en
   `src/` (afecta a todo el proceso y no se puede deshacer).
4. **La celda de figura sólo dibuja.** No lee disco, no entrena, no
   recalcula series: consume variables ya presentes en el espacio de nombres
   (`df`, `equity`, `fit`, `btc_df`, ...).  Si una figura necesita un cálculo
   nuevo, ese cálculo va en una etapa previa y su resultado se cachea en una
   variable; así se reusa al cambiar el estilo.
5. **Una figura por celda.** Cada figura en su propia celda, para poder
   re-ejecutar sólo esa.  Si una figura se guarda en disco
   (`fig.savefig("outputs/...", dpi=200, bbox_inches="tight")`), el `savefig`
   va al final de la misma celda, con nombre estable
   `<estrategia>_<dataset>_<metrica>.pdf`, y termina con `plt.close(fig)`
   para no acumular figuras.
6. **Recargar `src/` sin reiniciar el kernel.** Si el usuario cambia una
   función de `src/`, se recarga el *módulo* (no la función importada) y se
   re-ejecutan sólo las celdas dependientes:

   ```python
   import importlib
   import thesis.plots as plots
   importlib.reload(plots)
   ```

   Por eso el cuaderno debe **importar módulos** (`import thesis.plots as
   plots`) y llamar a `plots.plot_equity_curve(...)`, no usar
   `from thesis.plots import plot_equity_curve`; una función importada con
   `from` sigue apuntando al objeto viejo tras el `reload`.  Pon el `import` +
   `reload` en una celda propia, para que el usuario pueda re-ejecutarla.
7. **El cuaderno debe ser re-ejecutable por partes.** Comprueba que cambiar
   el estilo de la última figura y re-ejecutar sólo su celda no dispara
   entrenamiento ni descarga de datos; y que tras un `reload` de un módulo,
   sólo hay que re-ejecutar las celdas que lo usan.

## Fase 4 — Documentación

Si el experimento es autocontenido, crea `notebooks/<dominio>/README.md`
con: objetivo y pregunta central, pipeline paso a paso, API de cada función
pública (firma + qué hace), fórmulas matemáticas del dominio, cómo ejecutar,
artefactos generados, y una sección de interpretación y precauciones
(*look-ahead*, división temporal, coste de la estandarización, semillas).

Actualiza también la sección de estructura del `README.md` raíz.

## Fase 5 — Tests

Escribe tests en tres capas:

1. **Unitarios** con datos sintéticos, sin red. Incluyen las invariantes
   matemáticas del dominio.
2. **Integración** con datos reales, marcados con `@pytest.mark.integration`
   y **omitidos por defecto** vía `addopts = "-m 'not integration'"` en
   `pyproject.toml`. Se ejecutan con `uv run pytest -m integration`.
3. **End-to-end**: el orquestador completo con un config reducido
   (pocas épocas, `tmp_path` para los artefactos).

En series temporales, añade una prueba de **ausencia de look-ahead**: entrena
el modelo, corrompe la cola de la serie y comprueba que las features y las
predicciones anteriores a la corrupción son idénticas.

Patrones de código listos para copiar en `references/testing-patterns.md`.

## Fase 6 — Verificación y Git

**Verificación (obligatoria antes de cerrar):**

```bash
python3 -m py_compile <archivos nuevos o modificados>
uv run pytest
uv run pytest -m integration        # si la tarea lo requiere
uv run ruff check src tests
uv run ruff format --check src tests
```

Si `ruff format` reporta cambios, aplícalos y vuelve a pasar `ruff check`.

**Protocolo Git (respeta `CLAUDE.md`):**

1. Muestra el diff después de cada modificación.
2. Para archivos nuevos: `git add` y muestra `git diff --cached`.
3. Antes de commitear, revisa `git status`, `git diff` y
   `git log --oneline -10`. Prepara **solo** los archivos de la tarea.
4. No incluyas secretos, datos ni artefactos generados.
5. Agrupa toda la unidad lógica en un único commit y haz `git push` a
   `origin` inmediatamente, sin pedir confirmación (`CLAUDE.md` §8).
   Nunca uses `--force`.

**Estilo de mensaje:** asunto imperativo en español, sin punto final, y un
cuerpo que explique el *porqué* (problema → solución → decisiones → tests),
como el resto del historial del repo.

## Anti-patrones

- Dejar el archivo original intacto y añadir una copia en `src/`.
- Mover todo a `src/` y dejar el cuaderno vacío de explicación.
- Compactar todo el cuaderno en una única función `run_experiment()`: obliga a
  re-ejecutar el pipeline completo para cambiar un parámetro o una figura y
  elimina la posibilidad de iterar celda a celda.
- Un módulo gigante con todo mezclado.
- Fragmentar en archivos de una sola función.
- Figuras con estilo hardcodeado en el cuerpo (título, colores, `figsize`)
  en lugar de argumentos: obliga a editar `src/` y recargar el módulo para
  cambiar el aspecto.
- Cambiar una figura re-ejecutando el pipeline completo (reentrenar o
  releer datos) en vez de re-ejecutar sólo la celda de la figura.
- `plt.show()`, `plt.rcParams.update(...)` o `plt.style.use(...)` a nivel de
  módulo en `src/`.
- Importar funciones con `from <paquete>.<módulo> import <función>` en el
  cuaderno: impide que `importlib.reload` surta efecto.
- Declarar dependencias que el código no usa.
- Tests que dependen de red en la suite por defecto.
- Commitear cambios preexistentes del usuario junto a los tuyos.
- Ejecutar el pipeline completo como efecto secundario de un `import`.
