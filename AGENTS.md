# vllm-mlx — agent instructions

## Commands

- `PYTHONPATH=. .venv/bin/python -m vllm_mlx.cli serve ...` — canonical CLI invocation
  (see `.claude.local.md` if the `vllm-mlx` console script is inert on your machine).
- `./run.sh --help` — the model catalog (`models.json`): name, size, context, port, whether
  the weights are cached. `./run.sh <name>` serves one; `./run.sh --multi a b` serves several
  on one port; `./run.sh --env <name>` prints the client env vars.
- Servers started by `run.sh` are named by their **alias**, not the HF repo id — it passes
  `--served-model-name`, and `server.py:1827` rejects any other name with a 404 (on both
  `/v1/chat/completions` and `/v1/messages`).
- Bare `./run.sh` (no model given) serves the catalog `"default": true` model in **elastic
  single mode**: registry-backed (`ModelManager`, `model_registry.py`) with
  `max_resident_models: 1` — exactly one model resident, but a request naming a different
  catalog alias unloads the current one and loads the requested one (downloading first if
  needed) instead of 404ing. `./run.sh <name>` (no `--pin`) does the same for that alias.
  Two models concurrently resident (this repo's prior default) was suspected of capping GPU
  utilization at ~80% (not confirmed — unverified whether it was residency itself vs.
  something else, e.g. `gpu_memory_utilization`/`max_num_seqs`), which is why bare invocation
  went back to one resident model regardless — `--multi <names...>` still gives several
  models concurrently resident when that tradeoff is wanted deliberately.
- `--pin <name>` is the old, hard-pinned single-model mode: no registry, no catalog, 404 on
  any name but the one it was started with. Kept as the explicit "backup / resource-limited"
  option — zero registry bookkeeping overhead, and `--auto-unload-idle-seconds` only applies
  on this path (elastic single/multi have no idle-timeout unload — a resident model stays
  loaded until a different requested model displaces it).
- Elastic single/multi's parsers (`--reasoning-parser`/`--tool-call-parser`) are fixed at
  process startup from whichever alias was named on the command line (or `.multi`'s shared
  pair for 2+ names) — process-global, per `RegisteredModel` having no per-model parser field.
  If a client causes a swap to a *different* catalog model whose ideal parser differs, the
  swapped-in model runs under the original parser, not its own, until the process restarts.
- `models.json` model entries can set `priority` (default `0`, higher = evicted later) and
  `.multi.max_resident_models` (count cap alongside `memory_budget_gb`). Eviction picks the
  lowest-priority idle/active resident first; idle time only breaks ties within a tier.
  `/v1/models` now lists every catalog entry, loaded or not, with a `loaded` bool per entry —
  not just what's currently resident.
- `.multi.memory_budget_gb` is normally left unset in `models.json`: `run.sh` derives it at
  launch from THIS machine's actual RAM (`RUN_SH_RAM_BUDGET_FRACTION`, default `0.75` of
  `sysctl hw.memsize`) via `auto_memory_budget_gb`/`effective_memory_budget_gb`, so the same
  catalog doesn't silently over- or under-commit on a machine other than the one it was
  tuned on. Set the key explicitly to pin a fixed value again — an explicit value always
  wins. Bare `./run.sh --multi` (no names) also fits `default_multi` to that same budget by
  each alias's `size_gb`, in list order, and warns about (or on total failure, `die`s over)
  anything it had to drop — an explicit `--multi a b` is never trimmed this way.
- `run.sh`'s own startup warmup (`start_reporter`'s two priming calls to the initially
  preloaded alias) can race a real client request in elastic mode: if a different model is
  requested while warmup is still in flight, the warmup's second call can swap the resident
  model back once it fires. Self-resolving (one extra reload), not data-affecting — wait for
  the `warm` log line before benchmarking a fresh swap.
- `./scripts/serve-qwen3-thinking.sh` is now a shim for `./run.sh reasoning-30b`.
- `./run.sh --stop --all` kills every listener on a configured port. Check what is running
  first — it does not ask.
- `./run.sh --multi a b -- --extra-flag` — everything after `--` passes through to the
  underlying `serve` (e.g. `-- --disable-prefix-cache`).
- The server log is `/tmp/vllm-mlx.log`. Relaunch with `>>`, never `>`: truncating it on
  restart destroys the forensic window you will want ten minutes later.
- To smoke-test `run.sh`/registry changes fast, follow `.claude/skills/smoke-test-registry/SKILL.md`
  (Claude: the `smoke-test-registry` skill)
  (tiny cached models, serve→swap→evict in seconds).
- For a logic-only change (a new bash function, a jq snippet) that needs no real model
  load: extract just that function into a standalone temp script, or pipe the jq snippet
  directly, and unit-test it there. Do not source or invoke `run.sh` itself for this —
  bare invocation attempts an actual serve/bind against whatever port is configured, which
  can hit the shared, possibly-in-use endpoint. `./run.sh --help` is the one safe
  full-script smoke check (read-only, no bind).

## Reasoning effort

- `reasoning_effort` is a first-class request field on `/v1/chat/completions` (top-level,
  OpenAI shape) and `/v1/responses` (`reasoning.effort`); both merge into
  `chat_template_kwargs.reasoning_effort`, which remains the low-level override. Support is
  **per model template**: the server trial-renders each model's chat template
  (`vllm_mlx/utils/effort.py`) and validates requests against the exact vocabulary —
  out-of-vocabulary values and effort on a non-supporting model get a 400 naming the
  supported set (previously: silent no-op on qwen3-thinking, Jinja 500 on qwen3.8).
  `/v1/models` lists each entry's vocabulary as `reasoning_efforts` (null = unsupported).
- Catalog reality: only `qwen3.8-27b` supports it — `low`/`medium`/`xhigh` (NOT `high`),
  template default is `xhigh`, injected as a system-prompt instruction (so changing effort
  mid-session invalidates the prefix cache). `reasoning-30b`, `fast-8b`, `tiny-*`,
  `coder-80b`: no effort dial, only binary `enable_thinking`. Harmony/GPT-OSS models are
  statically `low`/`medium`/`high`.
- An effort arriving only via `--default-chat-template-kwargs` is dropped (with a debug log)
  on non-supporting models instead of failing the request; a request-supplied one is a hard
  400 by design.

## Benchmarking

- Read `.claude/skills/benchmark-tps/SKILL.md` before measuring tok/s (Claude: the
  `benchmark-tps` skill) — it encodes the warmup protocol and
  the measurement traps (SSE chunk counting vs `usage.output_tokens`, prefix-cache cold
  start, stream-interval batching).

## Wedge detection

**The pattern behind every incident in this repo's history: this server fails by reporting
itself healthy.** A green `/v1/models`, `running=0 waiting=0 orphans=0 stalled_for_s=0`, and a
listening port are evidence of nothing until a real completion proves otherwise. Two corollaries
worth holding permanently:

- **A number too good to be physically possible is a bug, not a win.** Verify the *content* of
  completions before believing a throughput jump.
- **Check the machine before believing any diagnosis of the server** — `vm_stat` (wired vs free)
  and `sysctl vm.swapusage`. A thrashing box makes every endpoint symptom look like an endpoint bug.

Full incident catalogue, the `/v1/status` field semantics, threshold-calibration rules, and the
`--multi` / elastic-swap failure modes: `.claude/skills/diagnose-wedge/SKILL.md`
(Claude: the `diagnose-wedge` skill).

## Engine and test rules

Before changing anything under `vllm_mlx/` or `tests/`, read
`.claude/rules/engine-and-tests.md` — MLX stream ownership, the format-only-what-you-touch
rule, and the suite's order-independence requirement. Claude loads it automatically on
those paths; Codex must open it.

## Concurrent agents in this checkout

- **One agent per working tree.** On 2026-08-23 a Codex session and a Claude session edited
  `vllm_mlx/scheduler.py` simultaneously here: a `git stash` taken by one raced the other's
  writes, `git stash pop` refused with "local changes would be overwritten", and the two
  nearly landed conflicting versions of the same fix. Nothing was lost, but only because the
  stash turned out to be a byte-identical duplicate.
- If a second agent must work here at the same time, give it its own git worktree rather than
  sharing this one. Before a `git stash`/`checkout`/`reset`, check whether another agent is
  mid-edit (`ls -la` the file's mtime, `ps aux | grep -iE "codex|pytest"`).
- Do not run the model-loading integration tests while the port-8000 server is busy. A wedged
  or loaded 30B server starves them: the same suite took 429–652s with multiple spurious
  failures under load, and 4.8s with 0 failures on an idle machine.
- **Detach anything long-running**: `nohup ./run.sh ... >> log 2>&1 & disown`, from a shell
  that exits. A harness killing a tracked background task kills its whole process group — on
  2026-08-24 that took down the endpoint, Prism's API and the UI mid-map despite `nohup`.
- `pkill -f` is not enough: `prism.api.serve` ignored SIGTERM, and relaunching before it died
  left two trees racing for one port (two supervisors, two endpoint locks). After any kill,
  confirm `pgrep -f <pat>` shows exactly one tree; SIGKILL survivors.
- Verify a config change actually reached the process rather than assuming:
  `ps -wwwE -p $(pgrep -f prism.api.serve | tail -1) | tr ' ' '\n' | grep ^PRISM_`
- **Port 8000 is shared with other agent sessions.** Ask before restarting it and say when
  you're done — a restart costs whoever is mid-map their in-flight calls.
