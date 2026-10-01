# Fork Maintenance Implementation Plan

> For agentic workers: use `superpowers:subagent-driven-development` for the
> role-based execution requested by the user. Follow the scope and gates below;
> the superseded five-extraction proposal is not an implementation instruction.

**Goal:** Reduce unnecessary fork divergence and qualify upstream integrations
before publishing them.

**Architecture:** Keep upstream-owned implementations at their existing paths.
Extract only fork-added reasoning-effort policy into a pure helper with a small
server adapter. Add fork-owned validation and documentation around integration.

**Tech stack:** Python >=3.10, existing pytest/MLX stack, Git worktrees, existing
GitHub Actions CI conventions. No new production dependency is required.

**Spec:** [Updated design](../specs/2026-10-01-fork-maintenance-design.md).

**Execution status (2026-10-01):** IN_PROGRESS. G1 and G2 are complete;
P3A policy extraction is in progress, and P3B workflow/runbook preparation is
complete. Final integrated review, candidate QA, hosted CI and publication are
pending. The schedule remains dependency-based.

## Global constraints

- Preserve public behavior and published history; no force push.
- Keep MLX worker/cache ownership, upstream SSE/admission/scheduler code, model
  discovery/cache, and endpoint call order with their current owners.
- Only format touched files; no broad cleanup or dependency upgrades.
- One writer per file, one worktree per agent, one integration owner.
- At most four concurrent agents including the integration owner.
- Only one MLX/GPU test process at a time. Worktrees do not isolate host resources.
- Do not restart shared services or load models against a busy host.
- Use persistent worker threads for MLX thread-affinity checks.
- Coder writes implementation and tests; QA executes checks and reports evidence
  without editing source. Reviewers are read-only.
- Never record populated credentials, request bodies, or environment dumps in
  tracked evidence. Retain sanitized summaries; keep raw logs outside the repo.

## Roles and execution queue

Models below are configured assignments from `~/.agents/role-workflows.md`.
At the original plan update, only the architect had been dispatched. Execution
assignments use the model, effort, checkout and revision recorded with their
evidence.

| Role | Configured model / effort | Ownership |
|---|---|---|
| Integration owner | Current parent session | Plan, branch pinning, integration, evidence gates, publication |
| Architect | GPT-6 Astra / high | Read-only design and fork-difference classification |
| Coder | GPT-6 Sol / high | Policy, narrow server adapter, meaningful tests |
| DevOps | GPT-6 Sol / high, using coder role | Fork workflow and maintenance docs; bounded repository configuration |
| QA | GPT-6 Luna / medium, using tester role | Baseline, targeted and integrated verification, failure evidence |
| Reviewer | GPT-6 Astra / high | Independent semantic and maintenance review of frozen diffs |

| Phase | Work and assigned roles | Dependency | Exit gate | Current status |
|---|---|---|---|---|
| P0 | Owner updates plan; Architect checks boundaries | Latest critical review | Revised scope and contracts written | COMPLETE |
| P1 | Architect inventories deltas; DevOps checks environment; QA establishes baseline after preflight | P0 | G1: source/environment pinned, inventory, baseline result | COMPLETE |
| P2 | Coder repairs baseline failures or removes proven redundant delta; Reviewer and QA check each commit | P1 findings | G2: stable baseline; necessary fork changes retained | COMPLETE |
| P3A | Coder extracts effort policy and adds characterization tests | G2 | Focused commit and targeted results | IN_PROGRESS |
| P3B | DevOps adds fork workflow and maintenance runbook | G2 | Workflow/doc commits, focused review and guard checks | COMPLETE; hosted CI NOT_RUN |
| P4 | Reviewer reviews the finished integrated diff; Owner combines accepted commits | P3A/P3B | G3: no unresolved correctness or scope findings | PENDING |
| P5 | QA validates frozen integrated candidate | G3 | G4: required regression and suite checks pass | PENDING |
| P6 | Owner refreshes remote state, scans outgoing changes and publishes | G4 | G5: final revision accepted and remote verified | PENDING |

### Dated baseline and workflow evidence (2026-10-01)

The original focused baseline had **291 passed, 15 failed**; its first-failure
evidence remains outside Git in
`/private/tmp/vllm-mlx-maintenance-baseline/focused-elevated.log`. The bounded
test-isolation/fixture repair at `fc83413` passed the focused gate with **422
passed, 4 deselected** in 6.75 seconds. Across 149 test files, the full suite
passed in forward order (**3587 passed, 24 skipped, 27 deselected**, 42.99 seconds)
and reverse order (same counts, 41.65 seconds). Logs are
`focused-repair2.log`, `forward-repair2.log` and `reverse-repair2.log` in that
same external directory. Skipped and deselected contracts are not live-model
acceptance. The reviewed repair was integrated at `4ce111c`; its `vllm_mlx/`
and `tests/` trees are identical to `fc83413`. No fork implementation was
removed as redundant.

The P3B workflow/runbook commits are `196b76d` and `ab14fc1`. A local check
of the workflow's extracted Linux guard returned exit codes 0 for a passing
test, 1 for pass plus skip, 5 for collection skip, 5 for zero collection, and
1 for a failing test. This validates the guard logic locally; hosted Linux and
Apple Silicon jobs remain **NOT_RUN** until their actual CI results exist.
These baseline results precede the P3A policy extraction and do not establish
G4 acceptance for its integrated candidate.

P1 can use Owner + Architect + DevOps + QA; DevOps host preflight must finish
before QA starts resource-consuming checks. P3 uses Owner + Coder + DevOps.
P4/P5 can use Owner + Reviewer + QA once all writers have stopped. GPU test runs
remain serialized even when read-only review overlaps.

## Review focus

1. Top-level/default/template precedence, explicit None and falsey values:
   characterization cases in P3 and endpoint checks in P5.
2. Unsupported versus unknown vocabulary and unloaded model discovery: existing
   effort/listing tests plus added missing cases in P3.
3. Cancellation while queued or while a stopped worker drains: existing engine
   regressions and a controlled restart-before-drain regression in P2 if absent.
4. Mid-stream engine faults, buffered output, and disconnect cleanup: existing
   server busy/error/structured-output checks in P1/P5.
5. A green Linux job hiding MLX omissions or a reused merge resolution discarding
   a local fix: explicit Apple Silicon fork suite in P3/P5 and review in P6.

## P1 — Pin baseline and inventory

**Owners:** Architect (analysis), DevOps (preflight/docs), QA (execution), Owner
(source snapshot and integration branch). Each uses its own worktree.

**Files:** Read `AGENTS.md`, `.claude/rules/engine-and-tests.md`, `pyproject.toml`,
`.github/workflows/ci.yml`, the fork diff, and relevant tests. DevOps creates
`docs/development/fork-maintenance.md` containing the architect's classifications
and later the maintenance runbook. No production code changes in this phase.

- [ ] Owner records clean/dirty state, local/fork/upstream SHAs and remotes, fetches
  origin/upstream, and creates an integration branch from the current fork tip.
  If upstream advanced, merge on that branch and review every resolved conflict
  before choosing the baseline; do not start extractions during conflict resolution.
- [ ] Create role worktrees at the pinned revision. Use distinct external paths
  such as `/private/tmp/vllm-mlx-maintenance-qa` and role-specific branches for
  writers. Give every worker exact ownership and tell them not to revert other work.
- [ ] DevOps checks architecture, Python/dependency versions and the existing venv,
  shared port-8000 process state, host memory, model-cache requirements, and test
  imports. Avoid dumping environment variables or process arguments containing keys.
  Record the interpreter and runtime without installing or upgrading implicitly.
- [ ] Verify worktree imports resolve to that worktree. Existing editable installs
  can otherwise import main while tests appear to exercise a candidate. Use
  `PYTHONPATH="$PWD"` from the role worktree and inspect module `__file__` there.
- [ ] Architect inventories each retained behavior: source paths, base/fork SHAs,
  behavioral contract, existing tests, upstream equivalence evidence, disposition.
  Required rows: effort validation/listing; residency/count/priority; admission
  and cancellation; worker/cache affinity; scheduler recovery/MTP; stream errors;
  API key environment fallback; launcher/catalog; documentation-only corrections.
- [ ] QA records collected tests and runs focused fork regressions, then full
  forward/reverse file order on an available Apple Silicon host. Preserve command,
  exit status, counts, skipped/deselected contracts and the first failure log.

Focused starting command, from the pinned QA worktree with the verified venv:

```bash
PYTHONPATH="$PWD" /Users/tiran/Documents/github/vllm-mlx/.venv/bin/python -m pytest -q \
  tests/test_reasoning_effort.py tests/test_chat_template_kwargs.py \
  tests/test_responses_api.py tests/test_server_engine_busy.py \
  tests/test_memory_cache_thread_affinity.py tests/test_engine_core_thread_streams.py \
  tests/test_engine_core_stream_safety.py tests/test_simple_engine.py \
  tests/test_simple_engine_thread_pinning.py tests/test_simple_engine_cancel_serialization.py \
  tests/test_scheduler_resilience.py tests/test_mllm_scheduler_resilience.py \
  tests/test_model_registry.py tests/test_cli.py
```

The existing `tests/conftest.py` skips `slow` tests without `--run-slow` and
server integration tests without `--server-url`. G1/G4 require the default suite
in both orders and the explicit fork fixture contracts. Those expected live/slow
skips must be listed as NOT_RUN, not counted as live acceptance. If a required
fork fixture unexpectedly skips, G4 stays closed. Separate real-model checks are
conditional on a demonstrated need for the changed behavior and host permission;
this pure policy extraction does not inherently require loading model weights.

Full suite order driver; run only after host/model preflight and verify that the
explicit file set matches normal pytest collection. The venv path is a current
candidate, not a claim it has already been checked:

```bash
PYTHONPATH="$PWD" /Users/tiran/Documents/github/vllm-mlx/.venv/bin/python - <<'PY'
import subprocess
import sys
from pathlib import Path

files = sorted(str(p) for p in Path('tests').rglob('test_*.py'))
assert files, 'No test files collected'
log_dir = Path('/private/tmp/vllm-mlx-maintenance-baseline')
log_dir.mkdir(exist_ok=True)
codes = []
for name, order in [('forward', files), ('reverse', list(reversed(files)))]:
    with (log_dir / f'{name}.log').open('w') as log:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', *order],
            stdout=log, stderr=subprocess.STDOUT,
        )
    print(name, result.returncode, log_dir / f'{name}.log')
    codes.append(result.returncode)
raise SystemExit(0 if all(code == 0 for code in codes) else 1)
PY
```

**G1:** Baseline evidence is fresh and bound to source/environment; failures are
retained and routed to P2. A failed baseline opens repair work only, not extraction.
If host resources block required tests, report BLOCKED and continue only independent
inventory/pure checks. Do not mark a subset as full acceptance.

## P2 — Resolve baseline findings before extraction

**Owners:** Owner/Architect scope a bounded work packet; Coder edits; Reviewer
reviews; QA reproduces and verifies. Do not assign the same source file to DevOps.

- [ ] For each baseline failure, retain the first failing node and evidence,
  identify the affected behavior, and classify environment versus source causes.
- [ ] Write a focused repair packet with exact files and acceptance case before
  assigning it; fix source regressions in separate commits from extraction.
- [ ] Check whether restart-before-drain is covered behaviorally: an old iterator
  remains active after stop times out; restart begins before that iterator is
  released; closure and cache cleanup finish on the old worker before new load.
  If missing, Coder adds an event-controlled fake-worker regression without real
  weights; QA verifies it. Do not introduce timing-only sleeps as proof of order.
- [ ] Delete a fork implementation only when Architect identifies an equivalent
  upstream implementation and QA demonstrates its contract. No redundancy has
  yet been proven, so no specific deletion is preapproved by this plan.
- [ ] Reviewer and QA accept each repair/removal. Establish the updated baseline
  revision and repeat affected checks; repeat suite orders when code or observed
  failure scope justifies it before opening G2.

**G2:** No unexplained baseline regression; no unproven removal. If P1 passes and
no redundant change is proven, P2 is explicitly NO_ACTION_REQUIRED.

## P3A — Extract fork reasoning-effort policy

**Owner:** Coder, GPT-6 Sol/high. Its own worktree starts at the G2 revision.

**Owned files:** create `vllm_mlx/effort_policy.py` and
`tests/test_effort_policy.py`; modify only `_apply_reasoning_effort` and its import
in `vllm_mlx/server.py`; add meaningful characterization cases to
`tests/test_reasoning_effort.py`, `tests/test_chat_template_kwargs.py`,
`tests/test_responses_api.py`, or `tests/test_anthropic_adapter.py` where a contract
is not already covered. Do not move existing test bodies solely to change paths.

**Import boundary:** Use the top-level module because `vllm_mlx/__init__.py` is
lazy, while `vllm_mlx/api/__init__.py` eagerly imports API models and utilities.
Do not modify either initializer merely to accommodate the extraction.

**Interfaces:**

```python
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class EffortResolution:
    template_kwargs: Mapping[str, object] | None
    replace_kwargs: bool = False
    dropped_default: bool = False
    error_detail: str | None = None

def resolve_reasoning_effort(
    *,
    resolved_template_kwargs: Mapping[str, object] | None,
    request_effort: str | None,
    request_template_kwargs: Mapping[str, object] | None,
    supported_efforts: tuple[str, ...] | None,
    model_name: str,
) -> EffortResolution:
    ctk = resolved_template_kwargs
    explicit = request_effort or (request_template_kwargs or {}).get('reasoning_effort')
    effort = explicit or (ctk or {}).get('reasoning_effort')
    if effort is None:
        return EffortResolution(ctk)
    replace = request_effort is not None
    if replace:
        ctk = dict(ctk or {})
        ctk['reasoning_effort'] = request_effort
        effort = request_effort
    if supported_efforts is None:
        if explicit is None:
            filtered = {k: v for k, v in (ctk or {}).items() if k != 'reasoning_effort'}
            return EffortResolution(filtered, True, True)
        detail = (
            f'model {model_name!r} does not support reasoning_effort '
            '(its chat template ignores the parameter); remove it or use a '
            'model whose /v1/models entry lists reasoning_efforts'
        )
        return EffortResolution(ctk, replace, error_detail=detail)
    if supported_efforts and effort not in supported_efforts:
        detail = (
            f'reasoning_effort {effort!r} is not supported by model '
            f'{model_name!r}; supported values: {", ".join(supported_efforts)}'
        )
        return EffortResolution(ctk, replace, error_detail=detail)
    return EffortResolution(ctk, replace)
```

The adapter retains the current no-effort fast path, discovers support only after
that guard, and calls this function. It then applies `replace_kwargs`, logs
`dropped_default` with the existing message, and raises the existing HTTP 400
using `error_detail`. Applying replacement before raising preserves current
mutation ordering. The helper returns validation errors as data for that reason.
Do not add cache ownership or reinterpret falsey values.

- [ ] Add missing characterization cases against the current adapter first.
  Existing behavior tests should pass before extraction; do not force them red
  by changing expected behavior.
- [ ] Add pure helper contract tests (red until the helper exists), including
  precedence, unsupported/unknown vocabularies, invalid defaults, unrelated
  kwargs, falsey request-template values, and input immutability. Example:

```python
from vllm_mlx.effort_policy import resolve_reasoning_effort

def test_default_effort_is_removed_without_mutating_input():
    original = {'reasoning_effort': 'low', 'enable_thinking': True}
    result = resolve_reasoning_effort(
        resolved_template_kwargs=original, request_effort=None,
        request_template_kwargs=None, supported_efforts=None, model_name='plain',
    )
    assert result.template_kwargs == {'enable_thinking': True}
    assert result.replace_kwargs and result.dropped_default
    assert result.error_detail is None
    assert original == {'reasoning_effort': 'low', 'enable_thinking': True}
```

- [ ] Implement the helper and thin adapter. Keep support cache, listing,
  `_resolve_chat_template_kwargs`, template introspection, and endpoint call sites
  in place. Pure tests import neither server nor MLX.
- [ ] Run new tests plus existing effort/chat kwargs/Responses/Anthropic tests;
  verify exact status/detail, absent-key shape, top-level precedence, and no
  support-probe call on a request with no effort.
- [ ] Compare the adapter diff against the pinned baseline/upstream. Stop and
  revise if extraction requires changing lifecycle/caching, copying upstream
  implementations, or broad unrelated edits.
- [ ] Commit as a focused behavior-preserving change, retaining test output for QA.

## P3B — Add fork regression CI and maintenance runbook

**Owner:** DevOps assignment using coder, GPT-6 Sol/high; separate worktree.

**Owned files:** new `.github/workflows/fork-regressions.yml` and
`docs/development/fork-maintenance.md` begun in P1. No edits to production code,
tests, or upstream `.github/workflows/ci.yml`. Consume the agreed new
`tests/test_effort_policy.py` path from P3A; integration waits for both commits.

- [ ] Add push/PR validation following repository workflow conventions, with
  read-only contents permission and normal hosted runners. Do not add scheduled
  auto-merges, privileged untrusted PR execution, or deployment credentials.
- [ ] Linux job: Python 3.10 and 3.13, install pytest, run only the dependency-light
  `tests/test_effort_policy.py`. Fail on accidental server/MLX import dependencies.
- [ ] Apple Silicon job: use existing project installation conventions and verify
  actual arm64/MLX availability before running the fork fixture files from P1.
  Keep the model-loading `test_engine_core_stream_safety.py` in the separate
  cached-model gate. Keep stub-based Linux tests in a separate process from real
  MLX. Missing architecture/dependencies must fail visibly, not silently skip
  the job.
- [ ] Add the new policy tests to the Apple job as well, so both jobs bind to the
  integrated implementation. Preserve logs and counts for skipped/deselected tests.
- [ ] Document fixture checks separately from full local suite-order validation
  and real-model checks. A hosted job passing a subset does not satisfy G4 alone.
- [ ] Document the inventory, standard remotes, integration worktree/branch process,
  explicit local `rerere.autoupdate=false`, conflict review, targeted/full test
  gates, normal push, and remote verification. Link this plan as design history.
- [ ] Include API key environment precedence and `.env` handling without populated
  keys or credential-bearing command arguments. Keep raw logs outside Git.
- [ ] Validate workflow syntax and inspect its permissions/triggers. QA executes
  its test commands in matching available environments; unavailable hosted results
  remain NOT_RUN until actually received. Commit workflow/docs together.

## P4 — Independent review and integration

**Owners:** Reviewer GPT-6 Astra/high, read-only on a frozen candidate worktree;
Owner integrates. Writers have finished before review begins.

- [ ] Supply Reviewer with baseline/upstream/candidate SHAs, exact diff, inventory,
  spec, test commands/results, and changed dependency information.
- [ ] Review behavior, ownership boundaries, error translation, imports, workflow
  safety, skip handling, and whether the upstream-owned change surface is actually
  reduced. Small line count alone is not enough.
- [ ] Route findings back to the owning Coder or DevOps worktree. Review repaired
  diffs and repeat affected tests. Do not let QA silently repair source.
- [ ] Integrate accepted commits into the owner branch with normal history.

**G3:** No unresolved correctness findings or unauthorized scope expansion.

## P5 — QA acceptance on the integrated candidate

**Owner:** QA GPT-6 Luna/medium in a fresh worktree at the integrated commit.

- [ ] Verify candidate import path/environment and repeat targeted policy and
  endpoint checks against the integrated tree, not an earlier worker checkout.
- [ ] Run the explicit fork regression set, including thread ownership, cache
  materialization, queue cancellation, restart drainage, registry limits, SSE
  error signaling, CLI/environment key precedence, and compatibility fallback.
- [ ] Run the full suite in forward/reverse file order on available hardware as
  in P1, using a new candidate-specific log directory. Check skip/deselection
  details against required contracts; no tests marked PASS without execution.
- [ ] Verify diff whitespace, no conflict markers, and no untracked credential
  files in the candidate. Confirm CI commands cover newly added fork tests.
- [ ] Report PASS/FAIL/BLOCKED/NOT_RUN with exact SHA, environment, commands,
  exit codes, counts, evidence paths, and remaining live-model limitations.

**G4:** Required checks pass and required contracts were executed. Blockers may
allow independent work to continue, but cannot be reclassified as acceptance.
No throughput, real-model, or production-serving claim follows from fixture tests.

## P6 — Publish accepted integration

**Owner:** Integration owner. This phase uses the session's existing commit/push
authorization for the fork; it does not authorize upstream PR messages or changes
to repository security settings.

- [ ] Refresh origin/upstream and compare against the pinned source. If either
  relevant main branch advanced, integrate it normally, review new resolutions,
  and reopen affected acceptance gates before publishing. Avoid rewriting history.
- [ ] Review all outgoing commits and final content with redacted secret scanning;
  verify local `.env` variants are ignored and `.env.example` is placeholder-only.
  Pattern scans are partial evidence; manually resolve any findings.
- [ ] Confirm the accepted candidate, inventory/runbook, and test evidence agree.
  Incorporate it into main with normal history, preserving unrelated local work.
- [ ] Push only to the fork's origin/main using a normal push. A non-fast-forward
  rejection returns to integration; never force the update.
- [ ] Verify remote main matches the delivered local commit and report working
  tree state, changes, evidence, and remaining limitations.

**G5:** Published fork commit matches the accepted source and remote confirmation.

## Plan-update evidence

The architect reviewed source at `82996d1` in an isolated worktree using its
configured GPT-6 Astra/high role. It confirmed the single-extraction interface,
the no-effort introspection boundary, and missing fork test coverage in current
CI. The integration owner checked the plan against the updated spec and current
source. No coding agent, QA test run, workflow deployment, or publication occurred
during this planning phase.
