# Fork maintenance design

Status (2026-10-01): implementation, independent review and local candidate
validation complete at `1036ca3` (G4). Publication of `ac58df4` to the fork
was verified (G5). Subsequent CI compatibility repairs passed both hosted
workflows at `1ff0a2e`; the plan contains the follow-up evidence.

This replaces the five-extraction proposal previously stored at
`/private/tmp/vllm-mlx-fork-merge-refactor-spec.md`. The implementation schedule is
[the companion plan](../plans/2026-10-01-fork-maintenance.md).

## Outcome

Maintain a small, explainable difference from upstream and detect incompatible
upstream changes before publishing the fork. Textually clean merges are useful
but do not prove that threading, cancellation, or API behavior remains correct.
No claim of conflict-free future merges is an acceptance criterion.

## Verified starting point

- Fork/main: `82996d1f89b35f47c34dfd8276c5765bf54af64e`.
- Last fetched upstream/main: `f5d7e00a5c23d7dd478bc07af03cc53fea75e6be`.
- Upstream is an ancestor of the fork: 37 fork-side commits, no pending upstream
  commits at the last fetch; the net difference touches 53 files.
- `origin` points to `tirans/vllm-mlx`; `upstream` points to
  `waybarrios/vllm-mlx`. Local main tracks origin/main.
- At the original design update, the last merge had 15 conflicted paths and
  received static checks and source review. Runtime baseline acceptance was
  NOT_RUN at that snapshot; the later G2 evidence is recorded in the plan.
- Local `rerere.enabled` is true; `rerere.autoupdate` was unset when inspected.
  The integration checkout subsequently set `rerere.autoupdate=false`.

These are a dated snapshot. Execution must record fresh branch and environment
identities before using them as evidence.

## Decisions replacing the original proposal

| Proposed extraction | Updated decision | Reason |
|---|---|---|
| Reasoning effort | One narrow pure policy extraction | Fork-added validation can leave a small adapter in the existing server function. |
| Registry capacity/ranking | Deferred | State accounting belongs under the manager lock; a separate ranking helper needs evidence of benefit. |
| Generation admission | Deferred | Lock, continuation context, counters, cancellation, and diagnostic ownership need a separate design. |
| Streaming terminal/error handling | Keep existing upstream helper in place | Upstream already edits this implementation; relocating it can create modify/delete conflicts. |
| Scheduler compatibility/sampling | Keep in place | These are different concerns tied to generator construction and scheduler state. |

## Work sequence

1. Pin the source/environment and establish a fresh tested baseline.
2. Inventory retained fork behavior and classify each difference as equivalent
   upstream behavior, a general correctness fix, fork policy, or fork tooling/docs.
3. Repair baseline regressions in separate focused commits. Retire a redundant
   fork change only when equivalence is demonstrated by source and behavior.
4. Extract the reasoning-effort rule without changing cache, introspection,
   endpoint call order, or public errors.
5. Add fork-owned regression CI and a maintenance runbook, then independently
   review and validate the combined candidate.
6. Integrate and publish only with acceptance evidence bound to the final source
   revision and a scan of outgoing content/history for credentials.

## Required behavior

- Preserve existing top-level effort precedence after intermediate kwargs merging.
- Preserve the distinction between unsupported vocabulary (`None`) and unknown
  vocabulary (`()`), and preserve current falsey request-template value behavior.
- An absent effort does not trigger model introspection or add a kwargs key.
- Unsupported server defaults are removed; unsupported request effort and invalid
  vocabulary values retain their current HTTP 400 details.
- Preserve unrelated kwargs, existing successful output shapes, and adapter
  mutation ordering before a validation error.
- Keep template probing/cache/listing and API error translation with their current
  owners; the new helper has no engine, HTTP, MLX, filesystem, or global state access.
- Preserve pinned worker ownership, materialization at cache/thread boundaries,
  queue cancellation, streaming failure signaling, registry limits/priority, and
  MTP fallback. Existing tests for those behaviors are integration gates.
- Keep API key CLI-over-environment precedence, ignored local `.env` files, and
  placeholder-only `.env.example` content.

## Validation and operational boundaries

The role-based execution includes meaningful characterization checks, existing
API regression tests, and the repository-required full suite in forward and
reverse file order. The original design update ran no tests. The later repaired
baseline passed G2 at `fc83413`, with its counts and logs in the companion
plan. The integrated candidate at `1036ca3` passed independent review,
targeted local fixture checks and both full suite orders. Counts, skips and
evidence paths and the verified publication receipt are in the plan. Hosted
CI subsequently passed on `1ff0a2e` after the documented compatibility repairs.

Linux dependency-light tests, fixture tests on Apple Silicon, and real-model
checks must be reported separately. Existing CI enumerates tests explicitly and
omits several fork-specific files. A new fork workflow supplements that CI.
Unrelated test failures, unavailable hardware, and skipped contracts cannot be
turned into a passing acceptance result.

One agent uses each worktree and one writer owns each file. One integration owner
combines commits. Worktrees do not isolate GPU memory: model-loading checks wait
for an available host; shared port-8000 services cannot be restarted without the
repository-required permission. Do not source `run.sh` to test shell logic.

Preserve published history. Upstream updates enter an integration branch,
receive semantic review and tests, then move into main using normal commits and
a normal push. Review all reused rerere resolutions. No force push or automatic
conflict-resolution driver is part of this design.

## Evidence of maintenance benefit

Compare the fork diff with the same pinned upstream before/after the change.
Report changes to upstream-owned functions, new dependencies, duplicated code,
and remaining integration points. Accept the extraction only if the server
adapter becomes smaller and no upstream implementation is copied or relocated.
This establishes a maintainability improvement, not proof of fewer future conflicts.
A historical merge experiment, if later needed, must use a matching earlier fork
base and genuinely later upstream edits; remerging an existing ancestor is invalid
evidence.
