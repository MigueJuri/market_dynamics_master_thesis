# Plan: quick

## Scope

Add a new primary agent `quick` that makes a single, small code change fast and
with minimal token spend, using the capable model `opencode-go/minimax-m3`.
"Minimal tokens" is achieved through the agent's **prompt discipline and
permissions**, not a cheaper model: it reads only the needed region, edits with
minimal exact strings, shows a short diff, and runs only targeted verification.

## Non-goals

- No changes to `src/`, `tests/`, `notebooks/`, `data/`, `outputs/`.
- No new skill, command, or plugin (a `/quick` command is optional, see below).
- No change to `opencode.jsonc`: `quick` is a fresh name, so file-form
  auto-discovery (`.opencode/agents/quick.md`) suffices — same as
  `explore.md` / `reviewer.md`, which are not in the `agent` map.
- Do not touch the existing `build`, `planner`, `explore`, `reviewer` agents.

## Files to touch

### 1. Create `.opencode/agents/quick.md` (new)

```markdown
---
description: Quick minimal-token code edits; one targeted change, show diff, run only needed tests
mode: primary
model: opencode-go/minimax-m3
permission:
  edit:
    "*": allow
  bash:
    "*": ask
    "git status*": allow
    "git diff*": allow
    "git log*": allow
    "rg *": allow
    "grep *": allow
    "uv run pytest*": allow
    "uv run ruff check*": allow
    "python3 -m py_compile*": allow
    "python3 scripts/*": allow
  read:
    "*": allow
    "data/**": ask
    "*.csv": ask
    "*.parquet": ask
    "*.npy": ask
    "*.npz": ask
    "*.pt": ask
    "*.pth": ask
    "*.ipynb": ask
---

You are the quick agent. Make exactly the small, well-scoped change the user
asks for. Do not refactor, rename, or "improve" anything beyond it. One change
per session.

Token discipline (the point of this agent):
- Locate code with `rg`/`grep` first, then `read` only the needed region
  (offset/limit). Never read whole files you do not need.
- Use a single targeted `edit` with a minimal exact oldString/newString. Never
  rewrite or paste an entire file.
- Show a short `git diff` after the edit. Do not paste unchanged file contents
  into chat.
- Verify with the smallest thing that proves the change: a targeted
  `uv run pytest tests/<file>` if one exists, else `python3 -m py_compile <file>`
  or `uv run ruff check <file>`. Do not run the full suite unless the change
  touches shared modules or the user asks.

Correctness (inherited repo rules):
- No look-ahead: signals use data <= t-1; rolling stats use shift(1);
  normalization fit on train only; time-ordered splits; fixed seeds.
- Vectorize with pandas/numpy; use np.linspace, not float-step np.arange.
- Handle LOB sentinels (0, -9999999999) explicitly; document units.
- Never read raw data files or .ipynb JSON wholesale.
```

### 2. Update `AGENTS.md` §6 routing table (add one row)

| Agente | Modelo | Cuándo usarlo |
|---|---|---|
| `quick` | `opencode-go/minimax-m3` | cambios de código pequeños y rápidos con mínimo gasto de tokens |

## Acceptance criteria

1. `.opencode/agents/quick.md` exists with valid frontmatter
   (`description`, `mode: primary`, `model`, `permission`) and a prompt body.
2. `quick` appears as a selectable primary agent after restart, and its prompt
   enforces: rg/grep-then-read, minimal `edit`, short `git diff`, targeted
   verification only.
3. `quick` can `edit` source and run targeted pytest/ruff/py_compile without
   prompts; `bash` for anything else still asks (broad default `*: ask`).
4. `read` of `data/**`, `*.csv`, `*.parquet`, `*.npy`, `*.npz`, `*.pt`,
   `*.pth`, `*.ipynb` requires confirmation (guards large reads).
5. `AGENTS.md` §6 lists `quick` (and its row matches the file's `model`).
6. `uv run pytest` still passes (config/docs only; verify no source touched).

## Key risks

- **Config drift:** the routing table row and the file's `model`/`mode` must
  agree, or agents are routed to a dead state. Criterion 5 guards this.
- **Over-broad edit permission:** `edit: "*": allow` trades safety for speed;
  `quick` is intended for trusted, scoped edits only — keep data reads gated
  (criterion 4) and document that `quick` is not for broad refactors.
- **Test coverage loss:** defaulting to targeted tests can miss regressions in
  shared modules. The prompt's "run the full suite when touching shared modules"
  clause is the guard.
- **Restart required:** config is loaded once at startup; the agent won't appear
  until opencode is restarted. Must be called out in the PR/summary.
- **Leakage / numerical issues:** none introduced by config change; the prompt
  re-states repo rules so `quick` edits stay compliant.

## Optional follow-up (do only if requested)

- A `/quick` command (`.opencode/commands/quick.md`, `agent: quick`,
  `subtask: true`) so `quick` can be invoked inline for a task.
