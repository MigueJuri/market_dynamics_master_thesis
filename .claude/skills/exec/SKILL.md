---
name: exec
description: Implementa un plan de docs/plans/, corre tests y lint, y hace commit + push.
argument-hint: docs/plans/<slug>.md
disable-model-invocation: true
---

Plan a ejecutar: `$ARGUMENTS`

1. Lee el plan y `git status --short`. Anota los cambios previos del usuario:
   **nunca** entran en tu staging.
2. Implementa **solo** lo que dice el plan, con cambios mínimos y ediciones
   puntuales, respetando `CLAUDE.md` §4–5. Si el plan es ambiguo o está mal,
   detente y pregunta en vez de improvisar.
3. Verifica:
   ```bash
   uv run pytest
   uv run ruff check src tests
   ```
   Corrige los fallos que introdujiste; reporta los preexistentes sin tocarlos.
4. Comprueba los criterios de aceptación del plan uno por uno.
5. Backup (`CLAUDE.md` §8): `git add` solo de los archivos de la tarea, un
   commit conciso (imperativo, prefijo de área) y `git push` a `origin`.
6. Resume: qué cambió, resultado de los tests y lo que quedó pendiente.
