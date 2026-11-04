# Plan: repo-consistency-agent-efficiency

## Scope

`Guide.md` was copied from a two-repo setup (coding repo + thesis writing repo)
and does not describe this repository. Bring it in line with what actually
exists here, and remove the duplication that costs tokens on every read.

- **Depurate** `Guide.md`: drop every agent, skill, command, model, workflow and
  troubleshooting row that does not exist in this repo (the whole "writing repo"
  half).
- **Add** the skills that do exist here but were missing from the list.
- **Consolidate**: one source of truth for agent routing; keep the
  always-loaded `AGENTS.md` lean.

## Non-goals

- No changes to `src/`, `tests/`, `notebooks/`, `data/`, `outputs/`.
- No change to the model routing table (already aligned in `AGENTS.md` §6).
- No new skills, agents, commands or scripts.
- Do not delete `Guide.md` without a home for its model/allowance tables.

## Ground truth to read first

- `Guide.md` (189 lines) — the file to edit.
- `AGENTS.md` §6 and §7 — routing table and session rules (already updated in
  this session: the repo planner is now `planner`, not `plan`).
- `.opencode/agents/*.md`, `.opencode/skills/*/SKILL.md`, `.opencode/commands/*.md`
  — what actually exists.
- `opencode.jsonc` — permissions and `instructions`.

## Files to touch

### 1. `Guide.md` — remove what does not exist

| § | Remove |
|---|---|
| 1 | "Writing repo" routing table; the "GPT 6 Luna" clause of the `small_model` note |
| 2 | `scripts/tex_errors.sh` and "the full `.bib`" from rule 6; the Qwen/Haiku/Grok/GPT price-tier list in rule 5 (keep "compact or restart at about 80K tokens") |
| 3 | model rows Qwen3.7 Plus, GLM-5.3-Flash, GPT 6 Luna |
| 4 | agent sections `writer` and `texfix`; the "writing" clause inside `reviewer` |
| 5 | the "Writing repo" skills block (`thesis-latex-style`, `latex-build-debug`, `figures-tables`) |
| 6 | command rows `/write`, `/tex`, `/review` |
| 7 | "Writing a thesis section"; "Moving results between repos" |
| 8 | the two writing prompt examples (keep the `/exec` one) |
| 10 | rows "Writer invents numbers" and "`texfix` loops"; the MiMo-V2.6-Pro suggestion in "Premium model exhausted" |

Also reword the intro (line 3): this guide describes **this** repo only, not a
two-repo setup.

### 2. `Guide.md` — add what does exist

In §5, after `backtest-report`:

- `integrate-existing-file`: extract reusable modules from scripts/notebooks
  into `src/`, preserve conventions, add tests, audit data-flow safety.
- `customize-opencode` (built-in): config, agents, skills, plugins, MCP servers
  and permissions of opencode itself.

### 3. `Guide.md` — consolidation

- §1 coding table stays the single source of truth for *who does what*;
  `AGENTS.md` §6 keeps only the model mapping. Do not repeat the "Use for"
  column verbatim in both files.
- §4: distinguish the repo planner (`planner`, writes `docs/plans/<slug>.md`)
  from the built-in `plan` agent (read-only, writes nothing).
- §3 (models) and §9 (budget) live in `Guide.md` only; they are not in
  `AGENTS.md`.
- Do **not** add `Guide.md` to `instructions` in `opencode.jsonc`: that would
  load ~120 extra lines on every request. Reference it with `@Guide.md` on
  demand.

### 4. `AGENTS.md` — no change needed

§6/§7 were updated in this session (the `plan` → `planner` rename, the
`small_model` row, and the note about the read-only built-in `plan` agent).
Re-read them before editing so the two documents do not drift.

## Acceptance criteria

1. `rg -i 'writer|texfix|tex_errors|thesis-latex|latex-build|figures-tables|qwen|glm-5|gpt 6|/write|/review' Guide.md` returns nothing.
2. `Guide.md` lists exactly the 4 repo skills plus the built-in `customize-opencode`.
3. Every agent and command named in `Guide.md` exists in `.opencode/agents/` or `.opencode/commands/` (the built-in `plan` excepted).
4. The routing table is defined in one place only (`Guide.md` §1 for "use for", `AGENTS.md` §6 for models).
5. `Guide.md` stays under 120 lines.
6. `uv run pytest` still passes (no source touched, but verify).

## Key risks

- **Agent/config drift:** a guide that names agents which do not exist sends
  agents into a dead workflow. Criterion 3 is the guard.
- **Losing cost data:** pruning §3/§9 must keep the rows for models actually
  used here (MiniMax M3, DeepSeek V4 Pro, Kimi K3.1 Pro, MiMo-V2.6-Flash,
  DeepSeek V4.1 Flash, Kimi K2.7 Code).
- **Context cost:** `Guide.md` must stay out of `instructions`, or every request
  pays for it.
- **Leakage / numerical issues / units:** none — documentation only, no
  quantitative code is touched.
