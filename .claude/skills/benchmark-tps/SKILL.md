---
name: benchmark-tps
description: Benchmark generation throughput (tok/s) on the local vllm-mlx server without the known measurement traps — prefix-cache cold-start, stream-interval chunk batching, warmup races, and prefill O(n^2) stalls. Use whenever measuring, comparing, or reporting tokens/sec for any model served by run.sh.
---

# Benchmark tok/s on vllm-mlx

## Before measuring — warmup (mandatory)

1. If the server was just started by `run.sh`, wait for the `warm` log line before
   sending anything — `start_reporter`'s two priming calls can otherwise race your
   request in elastic mode and swap the resident model back mid-benchmark.
2. Send **two throwaway requests with prompts >2k tokens** to the target model:
   - Prompts under ~2k tokens log `stored=False` — they never enter the prefix cache.
   - The FIRST prefix-cache hit after startup logs `Detected MLX stream/thread
     mismatch`, runs cache recovery, and re-prefills WITHOUT the cache (4.4s vs
     0.32s on an 11k-token turn). Never trust the first hit.

## Measuring

- Compute tok/s from **`usage.output_tokens` in the `message_delta` SSE event**
  (Anthropic route) or the final `usage` chunk (OpenAI route), divided by wall time
  from first content token to done.
- **Never count `content_block_delta` chunks**: with `--stream-interval N` one chunk
  carries N tokens, so chunk-counting understates interval 1 and overstates higher
  intervals by exactly that factor.
- Separate prefill from generation: prefill is O(n^2) (8k→2.5s, 32k→19s, 64k→74s,
  131k→~5min). Time-to-first-token is a prefill measurement, not a generation one.

## During the run

- `/v1/status` stops answering during a large prefill — a non-responding status
  endpoint does not mean the server is stalled.
- In registry mode, poll `/v1/status` for `active_requests` / `generation_tps` per
  model to distinguish "generating" from "stalled" without checking GPU usage.
- SimpleEngine serializes ALL generation behind one asyncio lock — run benchmark
  clients sequentially; concurrent clients queue and 503 after
  `VLLM_MLX_SIMPLE_ENGINE_QUEUE_TIMEOUT_S` (default 120s).

## Flags that silently change results

- `serve` without `--continuous-batching` drops `--max-num-seqs`,
  `--cache-memory-mb`, `--kv-cache-quantization*`, `--max-kv-size`,
  `--chunked-prefill-tokens` (parsed, then discarded — cli.py:252).
- `--max-kv-size > 0` switches to RotatingKVCache: silently drops oldest tokens AND
  disables prefix caching. Leave unset for honest comparisons.
- Changing `reasoning_effort` mid-session on qwen3.8-27b invalidates the prefix
  cache (effort is injected as a system-prompt instruction).
