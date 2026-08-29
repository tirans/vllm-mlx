---
name: diagnose-wedge
description: Diagnose a vllm-mlx server that looks healthy but is not serving — wedged engine, thrashing machine, ghost requests, silent prefill. Use when completions hang, return empty, stream nothing, or when throughput or latency looks impossibly good.
---

# Diagnosing a wedged vllm-mlx server


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

