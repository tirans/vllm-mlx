# CLAUDE.md

## Commands

- `PYTHONPATH=. .venv/bin/python -m vllm_mlx.cli serve ...` — the venv's `vllm-mlx` console
  script is INERT: every `.pth` in site-packages carries macOS `UF_HIDDEN`, and CPython 3.13
  skips hidden `.pth` files, so the editable install never registers.
  Permanent fix: `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`
- `./scripts/serve-qwen3-thinking.sh` — agent-workload server (batched, 256k ctx, warmed).

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

## Benchmarking

- Measure tok/s from `usage.output_tokens` in the `message_delta` SSE event, NOT by counting
  `content_block_delta` chunks — with `--stream-interval N` one chunk holds N tokens, so chunk
  counting understates interval 1 and overstates higher intervals by that factor.

## Docs that are wrong in this checkout

- `README.md` advertises `--mtp` and `--spec-prefill`; the real flags are `--enable-mtp` and
  `--specprefill`. Both README spellings fail argparse.
- `--moe-top-k` and `docs/guides/moe-top-k.md` describe a feature not present in this checkout
  (no `apply_moe_top_k_override`, no `tests/test_moe_top_k.py`).
