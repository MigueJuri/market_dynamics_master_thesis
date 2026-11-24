---
name: quick
description: Un único cambio de código pequeño y puntual con mínimo gasto de tokens; verifica y hace commit + push.
argument-hint: <cambio a realizar>
disable-model-invocation: true
---

Cambio: $ARGUMENTS

Haz exactamente ese cambio. No refactorices, renombres ni "mejores" nada más.

Disciplina de tokens:
- Localiza con `Grep`; luego `Read` solo la región necesaria (offset/limit).
- Un `Edit` mínimo; nunca reescribas ni pegues archivos completos.
- No pegues contenido sin cambios en la respuesta.

Verifica con lo mínimo que demuestre el cambio: `uv run pytest tests/<archivo>`
si existe; si no, `python3 -m py_compile <archivo>` o `uv run ruff check <archivo>`.
Suite completa solo si tocas módulos compartidos de `src/thesis/`.

Cierre (`CLAUDE.md` §8): `git add` solo del/los archivo(s) cambiados, commit
conciso y `git push` a `origin`. Respuesta final en 1–3 líneas.
