# Maintaining the fork

This runbook supplements the [fork maintenance plan](../superpowers/plans/2026-10-01-fork-maintenance.md)
and its [design](../superpowers/specs/2026-10-01-fork-maintenance-design.md).
The inventory below is a source classification at fork `82996d1f89b35f47c34dfd8276c5765bf54af64e`
against upstream `f5d7e00a5c23d7dd478bc07af03cc53fea75e6be`. Refresh both
revisions and the evidence when integrating another upstream change. A clean
textual merge does not establish behavioral compatibility.

## Retained fork behavior

| Area and source | Contract and regression evidence | Classification and disposition |
| --- | --- | --- |
| Reasoning effort: `vllm_mlx/server.py`, `vllm_mlx/api/models.py`, `vllm_mlx/utils/effort.py` | Request/default precedence, per-template vocabulary, `/v1/models` listing and HTTP 400 detail; `test_reasoning_effort.py`, `test_chat_template_kwargs.py`, `test_responses_api.py` | Fork policy. Extract only the pure resolution rule to `vllm_mlx/effort_policy.py`; retain template probing, cache, listing and HTTP translation with their current owners. |
| Residency and aliases: `vllm_mlx/model_registry.py`, `run.sh`, `models.json` | Resident count, priority, budget, live status, elastic single-model swaps and catalog aliases; `test_model_registry.py`, `test_cli.py`, registry smoke procedure in `AGENTS.md` | Fork policy. Retain. Upstream registry idle eviction is related but does not establish equivalent fork behavior; no deletion is qualified. |
| Admission and cancellation: `vllm_mlx/engine/simple.py`, `vllm_mlx/engine_core.py` | Queue capacity, cancellation and stop/drain ownership; `test_simple_engine.py`, `test_simple_engine_cancel_serialization.py`, `test_simple_engine_thread_pinning.py` | General correctness fix. Retain with worker and cache lifecycle. Check restart-before-drain coverage before future edits. |
| Worker and cache affinity: `vllm_mlx/mlx_streams.py`, `vllm_mlx/memory_cache.py`, `vllm_mlx/mllm_batch_generator.py`, engine files | Persistent worker streams, cross-thread model materialization and prefix-cache hit safety; `test_engine_core_thread_streams.py`, `test_engine_core_stream_safety.py`, `test_memory_cache_thread_affinity.py` | General correctness fix. Retain. Upstream worker behavior overlaps, but the merged MLX 0.32 contract has no proven redundant fork patch. |
| Scheduler recovery and MTP: `vllm_mlx/scheduler.py`, `vllm_mlx/mllm_scheduler.py` | Orphan/stall recovery, cache clearing, per-request sampling and rollback; `test_scheduler_resilience.py`, `test_mllm_scheduler_resilience.py` | General correctness fix. Retain in the scheduler's ownership boundary. |
| Stream errors: `vllm_mlx/server.py`, `vllm_mlx/engine/base.py` | Busy responses, SSE terminal/error signaling and disconnect cleanup; `test_server_engine_busy.py`, `test_responses_api.py`, `test_server.py` | General correctness fix. Retain the upstream-owned stream helper in place; moving it would create a second merge surface without proven benefit. |
| API key fallback: `vllm_mlx/cli.py`, `.gitignore`, `.env.example` | Explicit `--api-key` takes precedence over `VLLM_MLX_API_KEY`; ignored local `.env` variants, placeholder-only example | Fork policy/security. Retain. Followup `82fd70a` adds explicit CLI-over-environment cases to `test_cli.py`; QA validation and integration remain pending. |
| Launcher and guidance: `run.sh`, `scripts/setup.sh`, `models.json`, `AGENTS.md`, `.claude/skills/`, fork docs | Catalog serving, safe smoke/benchmark/incident procedures and corrected documentation | Fork tooling/docs. Retain; documentation corrections do not imply source equivalence. |

No upstream-equivalent fork implementation has been demonstrated removable by
both source comparison and a passing behavior check. Preserve the upstream
SSE, lifecycle, idle eviction and worker code at their existing paths while
reviewing each merged change semantically.

## Prepare an integration checkout

1. Check the checkout and remotes. `origin` is the fork
   (`tirans/vllm-mlx`); `upstream` is `waybarrios/vllm-mlx`.
   Record `git status --short --branch`, `git rev-parse HEAD`,
   `git rev-parse origin/main upstream/main`, and `git remote -v` without
   changing unrelated working trees.
2. Fetch `origin main` and `upstream main`. Create a distinct integration
   branch and worktree from the refreshed fork tip, for example
   `git worktree add -b fork/integrate-YYYYMMDD /private/tmp/vllm-mlx-integrate origin/main`.
   Give each concurrent writer its own worktree and owned files.
3. Compare `upstream/main...HEAD` with the inventory and review upstream
   changes before merging. If upstream advanced, merge with automatic rerere
   staging disabled for this command:

   ```bash
   git -c rerere.autoupdate=false merge --no-ff upstream/main
   ```

   Inspect `git status`, every conflict and `git diff --cc`; a reused rerere
   resolution still needs human review before `git add`. Keep normal commit
   history. Never force push or use a blanket conflict-resolution driver.
4. Before implementation or validation, identify the interpreter and verify
   `PYTHONPATH="$PWD"` resolves `vllm_mlx.__file__` inside the integration
   worktree. Check `platform.machine()`, Python, pytest, MLX and MLX-LM
   versions, port 8000 listeners, and `vm_stat`/swap pressure. Worktrees do
   not isolate GPU memory or the shared service. Use the existing verified
   environment; installation or a shared-server restart follows the
   session's authorization boundary.

The 2026-10-01 P1 DevOps preflight used a separate worktree at `bd2e1bb`:
Darwin arm64, Python 3.13.13, pytest 9.1.1, MLX 0.32.0, MLX-LM 0.31.3,
MLX-VLM 0.6.13. `PYTHONPATH="$PWD"` resolved to that worktree. `lsof`
found no TCP port-8000 listener; `vm_stat` succeeded. This sandbox denied
`sysctl hw.memsize`/`vm.swapusage`, and the shared venv had no `pip` module.
These observations are a dated preflight, not a test or a future host guarantee.

## Validation gates

The additive [fork workflow](../../.github/workflows/fork-regressions.yml)
runs the pure `test_effort_policy.py` on Linux 3.10/3.13 with only pytest,
requires collected tests with no skips, and checks that they did not import
server/MLX modules. Its separate Apple Silicon
3.11/3.13 job installs the existing `[dev,vision,harmony]` extras, verifies
arm64 and MLX 0.32 stream support, then runs the explicit fork fixture files
with `-ra`, offline Hugging Face/Transformers settings, and a failure on any
required fixture skip. The existing `ci.yml`
remains an independent gate and already runs the model-loading
`test_engine_core_stream_safety.py` separately; that file is deliberately
outside this fixture job. Neither fork job proves the full local suite or a
real-model completion.

On an available Apple Silicon host, run the plan's focused fork regression
command first. Keep `tests/test_effort_policy.py` in the list after its
implementation lands. Run the full suite in **both** file orders from the
verified environment, with logs outside Git:

```bash
PYTHONPATH="$PWD" python - <<'PY'
import subprocess
import sys
from pathlib import Path

files = sorted(str(path) for path in Path("tests").rglob("test_*.py"))
assert files, "No test files found"
log_dir = Path("/private/tmp/vllm-mlx-maintenance-validation")
log_dir.mkdir(exist_ok=True)
codes = []
for label, order in (("forward", files), ("reverse", list(reversed(files)))):
    log_path = log_dir / f"{label}.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-ra", *order],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(label, result.returncode, log_path)
    codes.append(result.returncode)
raise SystemExit(0 if all(code == 0 for code in codes) else 1)
PY
```

Compare the explicit file set and collected node count with normal
`python -m pytest --collect-only -q` before treating it as full coverage.
Record each command, source SHA, interpreter, exit status, pass/fail/skip counts,
first failure and log path. `tests/conftest.py` skips `slow` without
`--run-slow` and server integration without `--server-url`. Report these as
**NOT_RUN**; a required fork fixture that unexpectedly skips closes the gate.
Real-model and live-server checks are separate, conditional work. Check host
resources and server ownership before opting in; only a completed response
qualifies the live behavior, not `/v1/models` or a listening port.
When the Llama test model is already cached and the host is available, the
stream-safety gate is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  python -m pytest -q -ra tests/test_engine_core_stream_safety.py
```

That file calls `mlx_lm.load`; inspect its result and skip reasons. A missing
model that causes a skip is **BLOCKED**, not a passing stream-safety check.

## Credentials and publication

The CLI uses `args.api_key or os.environ.get("VLLM_MLX_API_KEY")`; a supplied
nonempty CLI key wins. This project does not automatically load `.env` in the
CLI. Keep local `.env`/`.env.*` ignored and `.env.example` placeholders only.
Do not place populated keys in commands, tracked logs, issue text or CI output.
Retain raw test logs outside Git and report sanitized failure summaries.

After review and full candidate validation, refresh both remotes again. If
either relevant main branch advanced, integrate normally, inspect conflict
resolutions, and repeat affected gates. Scan outgoing commits and content for
credentials, check `git diff --check`, and verify the accepted candidate SHA.
The integration owner alone moves the accepted branch to fork `main` with a
normal push and verifies `git ls-remote origin refs/heads/main` matches the
delivered commit. No hosted workflow here merges or publishes changes.
