---
paths:
  - "vllm_mlx/**"
  - "tests/**"
---

# Engine and testing rules

Loaded when working under `vllm_mlx/` or `tests/`. Always-on repo rules are in the root `CLAUDE.md`.

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

