---
description: Write a compact implementation plan to docs/plans
agent: planner
subtask: true
model: opencode-go/deepseek-v4-pro
---

Task: $ARGUMENTS.

Write a compact implementation plan to `docs/plans/<slug>.md`. Use `$1` as the
slug if provided; otherwise ask for one. Include:
- Scope and non-goals
- Files and functions to read/touch
- Acceptance criteria
- Key risks (leakage, numerical issues, units)

Keep the plan ≤1,500 tokens. Do not implement.