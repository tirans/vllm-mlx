# CLAUDE.md

@.claude.local.md

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
- To smoke-test `run.sh`/registry changes fast, use the `smoke-test-registry` skill
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

## Engine gotchas

- **`serve` without `--continuous-batching` silently drops most flags.** `cli.py:252` leaves
  `scheduler_config = None` unless that flag is set, so `--max-num-seqs`, `--cache-memory-mb`,
  `--kv-cache-quantization*`, `--max-kv-size` and `--chunked-prefill-tokens` are parsed and
  discarded. They only exist on `SchedulerConfig`. (The `benchmark` subcommand always builds one.)
- SimpleEngine (the default) serializes ALL generation behind one asyncio lock. Concurrent
  clients queue, then 503 after `VLLM_MLX_SIMPLE_ENGINE_QUEUE_TIMEOUT_S` (default 120s).
- `--max-kv-size > 0` switches mlx to `RotatingKVCache`: silently drops oldest tokens AND
  disqualifies prefix caching. Leave unset to keep full context.
- **A prefix-cache HIT crosses a thread boundary, and anything lazy dies there.** `fetch()`
  runs on the asyncio event-loop thread (`Scheduler.add_request`); the arrays are first read
  inside `step()` on the engine-core worker thread. MLX >= 0.32 registers streams per thread,
  so an op graph built on one cannot be evaluated on the other -- it raises
  `RuntimeError: There is no Stream(gpu, N) in current thread`, which clients see as "the
  engine aborted this request while generating". `fetch()` and `store()` therefore `mx.eval`
  on their own thread (`_prepare_for_stepping_thread`); a sixth hit path that skips it brings
  the bug back. Guarded by `tests/test_memory_cache_thread_affinity.py`.
  - This entry used to read "first HIT after startup logs `Detected MLX stream/thread
    mismatch`, runs cache recovery, and re-prefills without its cache — warm up with two
    requests". That was wrong in a way that cost a day. It described a one-off hiccup with a
    self-heal, so a crash LOOP read as the known-benign warm-up. In fact EVERY hit crashed,
    forever: each one poisons a fresh graph, so restarting the endpoint changes nothing.
    Measured 2026-08-27 -- 10/10 crashes followed a HIT, 0/6 cold requests crashed, and a
    downstream map died 42 consecutive times. The `Detected MLX stream/thread mismatch`
    message it told you to expect appeared **0 times** in that incident's log: it comes from
    an `engine_core` fallback that `ee8af4f` made unreachable (`step()` stopped re-raising),
    and it has since been deleted. If a doc names a log line as the symptom, grep the log for
    it before believing the doc.
- **A model loaded on one thread and first *used* on another must be materialized first.**
  `mlx_lm.load` leaves some module arrays lazy, and MLX >= 0.32 will not finish an op graph
  from another thread. `mx.eval(model.parameters())` is NOT enough — the offending arrays are
  non-parameter buffers (rope frequencies, masks); only `mx.eval(model.state)` reaches them.
  `Scheduler.__init__` does this (`materialize_model_arrays`), and so does
  `SimpleEngine.prepare_for_start` — it loads on the event-loop thread but runs non-stream
  chat inside an `asyncio.to_thread` worker. Any other path handing a model across threads
  must too. This is what kept the #407 guard red from `ee8af4f` onward.
  - The rejected alternative was `bind_generation_streams()`, deleted 2026-08-28. Rebinding
    the shared `mlx_lm.generate.generation_stream` never fixed the crossing — it only moved
    the breakage onto whichever thread had not bound last.
- Prompts under ~2k tokens log `stored=False` — too small to enter the prefix cache.
- Prefill is O(n^2): 8k->2.5s, 32k->19s, 64k->74s, 131k->~5min. `/v1/status` stops
  answering entirely during a large prefill.
- `--models-config` registry entries key on `name`, not `id` (`model_registry.py:329`) — a
  common mistake to copy from a generic example. `manager.memory_budget_gb` is required
  (`:241`); a non-local `source` with no `estimated_memory_gb` fails to load (`:835`).
  `docs/guides/model-registry.md` has the correct schema.
- Registry-mode `/v1/status` now reports live `active_requests`/`num_running`/`num_waiting`/
  `generation_tps` per loaded model (`model_registry.py::list_models`) — it used to answer
  with static load state only. Poll it to tell "generating" apart from "stalled" without
  shelling in to check GPU usage.
- `./run.sh --stop --all` followed by a fresh serve leaves the port refusing connections for
  several seconds while models preload. A consumer with a short retry budget (e.g. one 2s
  connect retry) can fail outright mid-restart rather than recovering — avoid restarting
  while something else is actively calling the server.

## Testing

- **The suite goes green: 0 failed (2385 passed as of 2026-08-28), in forward and reverse file order.** It
  did not until 2026-08-28 — `main` was 6 failed / 2376 passed with every failure passing in
  isolation, because `bind_generation_streams()` poisoned the process-wide
  `mlx_lm.generate.generation_stream` and whichever engine test ran after
  `tests/test_simple_engine.py` inherited a stream it could not use. That function is gone.
  A red test is now a real signal again, so treat one as your change until proven otherwise —
  but confirm order-independence (`pytest -q $(ls -r tests/test_*.py)`) before blaming it on
  ordering, and `git stash push -- vllm_mlx tests` before concluding it is pre-existing.
- **Nothing may assign to `mlx_lm.generate.generation_stream`, and no `mx.new_stream()` may
  live in cross-thread-visible storage** (a class attribute, a module global). The stream is
  registered in the creating thread only; any other thread using it dies at its first op —
  note *op*, not context entry: `with mx.stream(s): pass` succeeds cross-thread, which is how
  `MLLMBatchGenerator`'s shared class-level `_stream` stayed latent (fixed 2026-08-28:
  per-instance, minted on the constructing thread; a second `--multi` MLLM engine could
  neither step nor even `close()`). Guarded in `tests/test_engine_core_thread_streams.py` —
  the live `generation_stream` type, a source scan of `vllm_mlx/` that catches a
  reintroduced assignment even from code the suite never executes, and a behavioral
  two-thread test that catches re-shared MLLM streams under any attribute name.
- **Format only the files you touched.** The repo is not format-clean, so `ruff format <dir>`
  reformats 50+ unrelated files straight into your diff. Use `ruff format path/to/one.py`.
- **MLX threading probes must use persistent threads** (`ThreadPoolExecutor(max_workers=1)`),
  never a bare `threading.Thread` per step — an exited thread takes its stream registry with
  it, so ephemeral threads fail for a second, unrelated reason and make a broken fix look
  tested.
- `MLLMBatchGenerator` constructs with plain fake model/processor objects — no VLM weights
  (the template/think-suffix helpers exception-guard themselves; pair with `close()` on the
  constructing thread to restore the wired limit). "Needs a live VLM" deferred a real
  stream-seam fix once; the seam fires before any forward pass.
- `test_mllm_continuous_batching.py` fabricates generators via `MLLMBatchGenerator.__new__`
  at 8 sites, hand-setting only the attributes the tested path touches — a new per-instance
  attribute breaks whichever double reaches it. Complete the double; don't run `__init__`.
- The full suite is ~32 s. Run it whole (both orders) instead of guessing affected files.

## Benchmarking

- Use the `benchmark-tps` skill before measuring tok/s — it encodes the warmup protocol and
  the measurement traps (SSE chunk counting vs `usage.output_tokens`, prefix-cache cold
  start, stream-interval batching).

## Wedge detection

**The pattern behind every incident below: this server fails by reporting itself healthy.**
Four distinct bugs found on 2026-08-24/25 were all the same shape — ghost requests holding
slots (638bd07), a step loop re-entering forever on state it cannot recover from (8dea781,
ee8af4f), a wedged engine streaming 200 + empty SSE (08991f2), and an engine-aborted request
rendered as a 0-token success (3e39673). In all four the endpoint kept listening, answered
`/v1/models` with 200, and reported `running=0 waiting=0 orphans=0 stalled_for_s=0`. Assume a
green dashboard is evidence of nothing until a real completion proves otherwise.

Two heuristics that actually found bugs here, both cheap:

- **A number too good to be physically possible is a bug, not a win.** 582 model calls in 32
  minutes against a measured 29 min/call was read as "this rung is fast" and reported as a
  success; 581 of those calls were empty. A 0.01s completion was what exposed the fourth bug.
  When throughput improves by an order of magnitude, verify the *content* before believing it.
- **Check the machine before believing any diagnosis of the server** — `vm_stat` (wired vs
  free) and `sysctl vm.swapusage`. A thrashing box makes every endpoint symptom look like an
  endpoint bug.
- **Calibrate a monitor threshold against measured healthy AND failing values**, never
  plausibility. Three set by intuition here all cried wolf: a 30s status timeout against a
  documented ~5min prefill, `Pages free` on an OS that keeps it near zero by design, and a
  wired-GB bound calibrated for one resident model then left in place for two.

- `/v1/status` reports `orphan_response_count` and `stalled_for_s` per loaded model, on both
  the registry and direct paths. `num_running > 0` with `generation_tps == 0` is ambiguous
  (a long prefill looks identical), so use these to disambiguate: a non-zero orphan count
  means sequences were decoding with no receiver, and a large `stalled_for_s` means the
  batch is occupied but not progressing. Both stay 0 on a healthy server.
- **`/v1/status` is itself unavailable during a large prefill**, which is exactly when you
  most want to read it — the endpoint is single-threaded through the prefill, so the poll
  just hangs. Give a status poll a generous timeout (180s+, not 30s) and treat a timeout as
  "busy", never as "down": a monitor at 30s reported two spurious outages on 2026-08-23
  while the server was at 168% CPU chunk-prefilling 48k tokens and answering at 27.6s. The
  log is the reliable signal while a prefill runs — `[chunked_prefill] Starting/Completed`
  lines advancing means real progress. Prefer `[stall_watchdog]` in the log over polling.
- `VLLM_MLX_STALL_WARN_S` (default 600, 0 disables) sets when the scheduler logs
  `[stall_watchdog]`. Deliberately generous: prefill is O(n^2) and emits nothing while it
  runs. The watchdog only reports — it never aborts.
- **Match the client's concurrency to `--max-num-seqs`.** Calls beyond the admission window
  queue while emitting nothing, which clients routinely misread as a dead endpoint and
  cancel. See `l3l9`'s `PRISM_LLM_MAX_CONCURRENT`, which must be exported into the serve
  process's environment — its `llm.env` is not read for that variable.
- **`--multi` with two large models and `--gpu-memory-utilization 0.90` will wire the whole
  machine.** On a 128GB Mac that limit resolves to `allocation_limit=103.9GB`, and with
  `reasoning-30b` + `qwen3.8-27b` both resident MLX took it: measured 2026-08-24,
  `wired=111.3GB`, `free=0.6GB`, swap `61.3/62.5GB`, 30.8M pageins. Stopping the endpoint
  alone restored `free=113.0GB, wired=4.8GB`. `/v1/status` showed nothing wrong —
  `running=0 waiting=0 orphans=0 stalled_for_s=0` throughout — because the wedge gauges
  answer "is the batch stuck", not "is this machine thrashing".
- **Elastic single mode swaps models, and a swap can wedge the GPU.** Observed 2026-08-25:
  a map moved from a rung using `reasoning-30b` to one using `qwen3.8-27b` under
  `max resident 1`, and the swap raised `RuntimeError: There is no Stream(gpu, 8) in
  current thread` out of `mx.eval` in the batch generator's prompt path. Every Metal
  submission after that failed with `Command buffer execution failed: Ignored (for causing
  prior/excessive GPU errors)` — 1742 of them. `scheduler.py`'s step loop deliberately
  re-raises a stream/thread error (`_is_stream_thread_error`), and a raise out of `step()`
  is a hang rather than a crash, so the endpoint stayed listening, answered `/v1/models`
  with 200, and returned 503 to every generation until it was restarted. If a workload
  alternates models per stage, prefer `--multi` so no swap ever happens — with
  `MLX_BUFFER_CACHE_LIMIT` set, two resident models cost ~77GB wired rather than the 111GB
  that made `--multi` look like the problem in the first place.
- **Check the machine before believing any endpoint diagnosis.** `vm_stat` (wired vs free)
  and `sysctl vm.swapusage` first; the scheduler's own `[Metal memory] active=` line read
  246-254GB on a 128GB box, so treat it as MLX's virtual accounting, not residency.
- **A probe-gated client stays blocked for as long as the machine is thrashing.** Prism's
  supervisor resumes only once an 8-token probe decodes inside 45s. Under the swap
  condition above, 246 probe requests produced 4 successes and a failed map waited 6h45m to
  resume; on a healthy machine the same probe returns in 0.9s and the map resumed in 39s.
  When diagnosing "why hasn't it picked back up", count successful small completions
  (`Chat completion (stream): 8 tokens`) rather than trusting `num_running`/`num_waiting`.

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

## Docs that are wrong in this checkout

(Previously listed `--mtp`/`--spec-prefill` and `--moe-top-k`; both corrected — the flags
now read `--enable-mtp`/`--specprefill` in all four READMEs, and the `moe-top-k` guides were
removed since `apply_moe_top_k_override` exists in no branch of this repo.)
