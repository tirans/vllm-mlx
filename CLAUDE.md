# CLAUDE.md

## Commands

- `PYTHONPATH=. .venv/bin/python -m vllm_mlx.cli serve ...` — the venv's `vllm-mlx` console
  script is INERT: every `.pth` in site-packages carries macOS `UF_HIDDEN`, and CPython 3.13
  skips hidden `.pth` files, so the editable install never registers.
  Permanent fix: `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`
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

- Measure tok/s from `usage.output_tokens` in the `message_delta` SSE event, NOT by counting
  `content_block_delta` chunks — with `--stream-interval N` one chunk holds N tokens, so chunk
  counting understates interval 1 and overstates higher intervals by that factor.

## Docs that are wrong in this checkout

- `README.md` advertises `--mtp` and `--spec-prefill`; the real flags are `--enable-mtp` and
  `--specprefill`. Both README spellings fail argparse.
- `--moe-top-k` and `docs/guides/moe-top-k.md` describe a feature not present in this checkout
  (no `apply_moe_top_k_override`, no `tests/test_moe_top_k.py`).
