# OpenCode Go: models, agents and skills usage guide

Reference for this repository only (quantitative code, analysis and backtests). Model quality ratings below are reasoned from prices, allowances and OpenCode's own descriptions, not benchmarks. Verify on your own tasks and adjust the routing tables.

## 1. Routing at a glance

| Role | Agent | Use for |
| --- | --- | --- |
| Default builder | `build` | Implement from a plan, run tests, routine edits |
| Planner | `planner` | Design, decomposition, debugging strategy; writes `docs/plans/<slug>.md` |
| Searcher | `explore` (subagent) | Find code, definitions, call sites |
| Reviewer | `reviewer` (subagent) | Leakage, statistics, numerical issues |
| Hard-bug fallback | `build` (Kimi K2.7 Code) | Only after MiniMax fails the same bug twice |

This table is the single source of truth for *who does what*. The model bound to each agent lives in `AGENTS.md` §6; per-model cost data lives in §3 below. Small tasks (titles, summaries) go to `small_model` (the model alias in `opencode.jsonc`).

## 2. The rules that save the most

1. **Cost is mostly cached reads.** Every request re-reads the whole session context. Short sessions beat long ones.
2. **One task per session.** New task, new session.
3. **Never switch model or primary agent mid-session.** The cache is per model. Switching (including Tab between primary agents) re-reads everything at full input price. Open a new session instead.
4. **Stable prefix.** Don't edit `AGENTS.md`, skills or config mid-session. Changes take effect cleanly in the next session.
5. **Keep contexts small.** Compact or restart at about 80K tokens anyway: quality and cost both degrade with long contexts.
6. **Control tool output.** Use `rg -n`, `head`, line ranges, `scripts/df_peek.py`. Never `cat` notebooks, data, logs or PDFs.
7. **Name files with `@path`** instead of letting the agent search.
8. **Ask for diffs or targeted edits**, not full-file rewrites. Output tokens cost the most per token.

## 3. Models

Allowances are monthly per model; the 5-hour cap is 20% of it and the weekly cap 50%. Request counts are OpenCode's estimates for typical agent traffic.

| Model | Allowance / est. requests per month | Use it for | Avoid it for |
| --- | --- | --- | --- |
| MiniMax M3 | $60 / \~16,000 | Daily coding in this repo | Tasks it fails twice (escalate) |
| Kimi K2.7 Code | $60 / \~6,750 | One hard debugging or refactor task | Everyday edits (about 3x the cost of MiniMax) |
| DeepSeek V4 Pro | $15 / \~5,200 | Planning, decomposition (about $3 per 5-hour window) | Routine edits, long sessions |
| Kimi K3.1 Pro | $15 / \~735 | Reviews of finished work | Anything iterative; it runs out fast |
| MiMo-V2.6-Flash | $60 / \~150,000 | Exploration and searching | Writing or reasoning-heavy work |
| DeepSeek V4.1 Flash | $80 / \~173,000 | `small_model`, bulk cheap tasks | Complex reasoning |

- DeepSeek models require Global region enabled in Console → Privacy, and peak pricing (about 2x) applies 01:00-04:00 and 06:00-10:00 UTC on weekdays. Daytime at UTC-3 is off-peak.
- DeepSeek's zero-retention agreement is valid through October 31, 2026 and is renewed monthly. If that matters, change `planner` to MiMo-V2.6-Pro.
- Do not use Muse Spark Contributor or Iceman Contributor Free for thesis work: they use your prompts for training.
- Free models (Beam Free) are a fallback after you hit limits, not a default.

## 4. Agents

Start a session with the agent you want, state the task in one or two sentences, and name files with `@path`.

| Agent | Purpose | Give it | Pitfall |
| --- | --- | --- | --- |
| `planner` (primary) | Write a plan to `docs/plans/<slug>.md`: files to touch, functions, acceptance criteria, ≤1,500 tokens | Goal, constraints, relevant `@files` | Don't ask it to implement; read and edit the plan first, a wrong plan propagates |
| `build` (primary) | Implement a plan and run tests | Only the plan file (`/exec docs/plans/<slug>.md`), plus failing output | Fails the same bug twice → stop, open a new session on Kimi K2.7 Code with the plan and the failing output |
| `explore` (subagent) | Locate things; returns paths and line ranges, not file contents | `@explore`, or let the primary agent call it | Runs in its own context, so it doesn't break the session cache |
| `reviewer` (subagent, read-only) | Rigor check of finished work: leakage, statistics, units, numerical issues | One file or function and what to check | Report ≤30 lines; use it after the work is done, it has the smallest allowance |

OpenCode's built-in `plan` agent is read-only and writes nothing; use `planner` to persist a plan.

## 5. Skills

Skills sit in `.opencode/skills/<name>/SKILL.md`. Only the name and description are in context; the body loads when the agent decides it is relevant. You can also force it: "use the `lookahead-audit` skill on `@path`".

- `lookahead-audit`: checklist for future leakage in features, rolling windows, normalization, labels, equity-curve compounding. Use before trusting any backtest or forecaster result.
- `lobster-format`: LOBSTER column layout, price scaling, event types, sentinel values, session filtering, bin-averaging pitfalls. Use when touching LOB code.
- `backtest-report`: metrics, cost assumptions, out-of-sample reporting, plot conventions. Use when producing results for the thesis.
- `integrate-existing-file`: extract reusable modules from scripts/notebooks into `src/`, preserve conventions, add tests, audit data-flow safety.
- `customize-opencode` (built-in): config, agents, skills, plugins, MCP servers and permissions of opencode itself.

Maintenance: keep each skill short; fix a skill when you find a repeated mistake, rather than adding the rule to `AGENTS.md`.

## 6. Commands

| Command | What it does |
| --- | --- |
| `/plan <task>` | `planner` writes `docs/plans/<slug>.md` |
| `/exec <plan-file>` | `build` implements only that plan and runs tests |
| `/leak <path>` | `reviewer` + `lookahead-audit` |

## 7. Workflows

| Situation | Steps |
| --- | --- |
| New analysis or feature | 1) new session `/plan <task>`, read and edit the plan; 2) new session `/exec docs/plans/<slug>.md` (MiniMax M3); 3) `/leak @path` on anything that produces results; 4) commit, naming outputs descriptively and stably |
| Debugging | 1) state the failure with the exact error output; 2) MiniMax M3 attempts a fix, and if it fails twice open a new session on Kimi K2.7 Code with the plan and failing output; optionally `/plan` first for a DeepSeek V4 Pro diagnosis |

## 8. Prompt patterns

- Good: "`/exec docs/plans/lob-features.md`" (the plan carries the context.)
- Bad: pasting the whole notebook output into the prompt.

## 9. Budget management

- Each model's 5-hour cap is 20% of its monthly allowance, weekly 50%. For $15 models that is about $3 per 5 hours and $7.50 per week.
- Premium models (DeepSeek V4 Pro, Kimi K3.1 Pro): plans and reviews only.
- Check the console, or `opencode.allowance` in responses, after the first week and compare actual spend with the tables above.
- Extra Usage (pay-as-you-go fallback) is off by default. After limits you can use free models.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Cost climbing in a long session | Large cached context re-read every turn | Compact or start a new session with the plan file |
| Agent re-explores the repo | Vague plan or missing `@path` | Tighten the plan; name files |
| Model not found | List changed | `curl https://opencode.ai/inference/go/v1/models`, update IDs |
| DeepSeek request denied | Global region not enabled | Console → Privacy → Global |
| Premium model exhausted | $15 cap hit | Wait for the window, switch to a cheaper model, or free models |

## 11. Monthly checklist

- Re-check model IDs and allowances against the Go docs.
- Keep each `AGENTS.md` under 100 lines; move detail to skills.
- Review spend per model and adjust the routing tables.
- Confirm DeepSeek's zero-retention status if privacy matters.
- Delete or fix skills that the agents misuse.
