---
description: Execute a plan file with the build agent
tags: []
agent: build
subtask: true
model: opencode-go/minimax-m3
---

Read the plan file `$ARGUMENTS` (or `$1`) and implement only what it says. Keep
changes minimal; show the diff after every edit. When finished, run
`uv run pytest` and report any failures. Do not read raw data files or `.ipynb`
JSON wholesale.