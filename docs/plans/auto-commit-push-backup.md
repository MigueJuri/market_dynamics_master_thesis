# Plan: Auto-commit + auto-push para backup permanente en GitHub

**Slug:** `auto-commit-push-backup`
**Enfoque:** instrucciones + permisos (sin hooks), según elección del usuario.

## Alcance

Hacer que los agentes realicen `git commit` y `git push` automáticamente (sin
pedir confirmación) al cerrar cada unidad de trabajo, de modo que GitHub
(`origin` = `https://github.com/MigueJuri/market_dynamics_master_thesis.git`)
actúe como backup permanente. Se reescribe la política Git de `AGENTS.md` y se
ajustan los permisos `bash` para que `add/commit/push` no caigan en "ask".
No se instalan hooks ni cron.

## Archivos a tocar

1. **`AGENTS.md` — §8 "Git y seguridad"**
   Reemplazar las reglas actuales ("diffs visibles antes de continuar",
   "preguntar antes de `git push`") por una política de backup automático:
   - Commit automático por **unidad lógica** al terminar cada tarea/cambio
     (mensajes concisos, estilo del repo: imperativo, prefijo de área si aplica).
   - `git push` automático a `origin` inmediatamente después de cada commit
     (o al final de la sesión), sin preguntar.
   - Conservar invariantes de seguridad: nunca commitear secretos/tokens/.env,
     nunca `push --force`, advertir antes de operaciones destructivas,
     y respetar `.gitignore` (ya excluye `data/`, `outputs/`, `.env*`, `*.ipynb`,
     artefactos). No mergear ramas ni crear PRs sin pedirlo.

2. **`opencode.jsonc` — `permission.bash`**
   Añadir al allow-list (manteniendo `"*": "ask"` y el resto de reglas):
   `git add*`, `git commit*`, `git push*`, `git pull*`, `git fetch*`.
   Añadir después una regla `git push --force*` → `deny` (el último match gana).
   No tocar el resto del archivo (modelos, compaction, watcher, read/edit).

3. **`.opencode/agents/build.md` — prompt**
   Añadir un paso final explícito: tras `uv run pytest` en verde, hacer
   `git add` de los archivos intencionados, commit conciso y `git push`.
   (El agente ya tiene `bash: allow`, no requiere cambio de permisos.)

4. **`.opencode/agents/quick.md` — prompt + frontmatter `permission.bash`**
   Añadir el mismo paso commit+push tras verificar el cambio, y agregar al
   allow-list del frontmatter: `git add*`, `git commit*`, `git push*`.

## Primer paso inmediato (backup del estado actual)

El árbol está sucio en `chore/opencode-config` (varios modificados, 1 borrado
`notebooks/SPY_caracterizacion.ipynb` y varios untracked en `src/`, `tests/`,
`docs/plans/`, `notebooks/`). La primera sesión de `build` debe:
1. Revisar `git status`/`git diff` para NO subir nada sensible.
2. Commitear el trabajo pendiente en unidades lógicas.
3. `git push` a `origin/chore/opencode-config` para materializar el backup.
No mergear `chore/opencode-config` → `main` sin pedirlo.

## Criterios de aceptación

- `AGENTS.md` §8 describe commit automático por unidad lógica + push automático,
  y conserva las reglas de no-secretos / no-force-push / no-destructivos.
- `opencode.jsonc` valida (JSONC válido, `$schema` intacto) y `git commit/push`
  ya no requieren aprobación para el agente por defecto/`quick`.
- `build` y `quick` incluyen el paso de commit+push en su prompt.
- Tras la primera ejecución: working tree limpio (salvo ignorados) y
  `git status` muestra `origin/chore/opencode-config` sincronizado
  (`ahead 0`).

## Riesgos principales

- **Commit de WIP/código roto o secretos** por automatización. Mitigación:
  mantener la regla de no-secretos, revisar `git diff` antes de `git add` y
  apoyarse en el `.gitignore` existente.
- **Regresión de permisos por orden de reglas** en `opencode.jsonc`: el último
  match gana; colocar las reglas nuevas antes del `"*": "ask"` y el
  `push --force` deny después de `git push*`.
- **Conflictos/divergencia de ramas** (`chore/opencode-config` vs `main`):
  no mergear ni `pull --rebase` sin pedirlo; si hay conflicto, detenerse y avisar.
- **Push sin credenciales/red**: fallo benigno; reintentar al final de la sesión.
- Los cambios de config no se aplican en caliente: reiniciar opencode para que
  tomen efecto los nuevos permisos/prompts.
