@AGENTS.md
@.claude.local.md

# Claude Code — vllm-mlx

Shared contract is `AGENTS.md`, imported above. Claude-only below.

- `.claude/rules/engine-and-tests.md` loads automatically when you open a file under
  `vllm_mlx/` or `tests/`. Codex has to be pointed at it; `AGENTS.md` does that.
- Skills: `/benchmark-tps` (warmup + measurement traps), `/smoke-test-registry`
  (fast serve→swap→evict check), `/diagnose-wedge` (wedged-server triage).
- `/compact` at task boundaries, never mid-task.