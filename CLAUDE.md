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
- `run.sh`'s own startup warmup (`start_reporter`'s two priming calls to the initially
  preloaded alias) can race a real client request in elastic mode: if a different model is
  requested while warmup is still in flight, the warmup's second call can swap the resident
  model back once it fires. Self-resolving (one extra reload), not data-affecting — wait for
  the `warm` log line before benchmarking a fresh swap.
- `./scripts/serve-qwen3-thinking.sh` is now a shim for `./run.sh reasoning-30b`.
- `./run.sh --stop --all` kills every listener on a configured port. Check what is running
  first — it does not ask.
- To smoke-test `run.sh`/registry changes fast, use the `smoke-test-registry` skill
  (tiny cached models, serve→swap→evict in seconds).

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
- First prefix-cache HIT after startup logs `Detected MLX stream/thread mismatch`, runs cache
  recovery, and re-prefills that request WITHOUT its cache (4.4s vs 0.32s on an 11k turn).
  Warm up with two requests >2k tokens before trusting any measurement.
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

## Benchmarking

- Use the `benchmark-tps` skill before measuring tok/s — it encodes the warmup protocol and
  the measurement traps (SSE chunk counting vs `usage.output_tokens`, prefix-cache cold
  start, stream-interval batching).

## Wedge detection

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

## Docs that are wrong in this checkout

(Previously listed `--mtp`/`--spec-prefill` and `--moe-top-k`; both corrected — the flags
now read `--enable-mtp`/`--specprefill` in all four READMEs, and the `moe-top-k` guides were
removed since `apply_moe_top_k_override` exists in no branch of this repo.)
