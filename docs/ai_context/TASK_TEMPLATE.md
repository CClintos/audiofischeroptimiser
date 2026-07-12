# TASK_TEMPLATE.md

Use this structure when giving Codex a bounded implementation task.

```text
Task:
<one concrete change>

Read:
- AGENTS.md
- REPO_MAP.md
- docs/ai_context/PROJECT_MAP.md
- <smallest relevant code files only>

Do not inspect:
- <unrelated files or subsystems>

Goal:
- <desired behaviour>

Constraints:
- <safety rules that matter for this task>

Acceptance:
- <observable checks or validation requirements>
```

## Good Example

```text
Task:
Adjust optimiser scoring for <specific behaviour>.

Read:
- AGENTS.md
- REPO_MAP.md
- docs/ai_context/DSP_RULES.md
- _optimizer.py
- _tunefit.py
- objective_module/afpx_objective.py

Do not inspect:
- afpx.py
- pct6.py
- unrelated output folders

Acceptance:
- benchmark or equivalent before/after comparison
- concise explanation of audible trade-off
- no unrelated write-path changes
```
