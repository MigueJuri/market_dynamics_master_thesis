---
name: write-plan
description: Analiza una tarea y escribe un plan compacto en docs/plans/<slug>.md sin implementar nada.
argument-hint: <slug> <tarea>
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Write(/docs/plans/**), Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git ls-files:*), Bash(rg:*), Bash(python3 scripts/df_peek.py:*)
---

Tarea: $ARGUMENTS

El primer argumento es el slug del plan; si falta o no parece un slug
(`kebab-case`), pregunta uno antes de escribir.

Analiza solamente: lee el código necesario (con `Grep` y luego `Read` con
rango de líneas) y **no** implementes ni edites nada fuera de
`docs/plans/<slug>.md`, que es la única escritura permitida.

El plan, en ≤1.500 tokens, debe incluir:

- **Alcance y no-objetivos.**
- **Archivos y funciones** a leer/tocar (rutas concretas; reutilizar lo que ya
  existe en `src/thesis/`).
- **Pasos** de implementación en orden.
- **Criterios de aceptación** verificables (tests, comandos, salidas).
- **Riesgos**: look-ahead, unidades, problemas numéricos, splits temporales.

Al terminar, muestra la ruta del plan y recuerda ejecutarlo en una sesión
nueva con `/exec docs/plans/<slug>.md`.
