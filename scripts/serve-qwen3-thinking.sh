#!/usr/bin/env bash
# Compatibility shim. The real launcher is ./run.sh, driven by ./models.json —
# this checkpoint is the `reasoning-30b` entry there.
#
#   ./run.sh --help              every model, with its size, context and port
#   ./run.sh reasoning-30b       what this script now does
#
# The tuning rationale and the measured numbers that used to live in this header
# moved to the top of ../run.sh, unchanged.
#
# ONE BEHAVIOUR CHANGE: run.sh passes --served-model-name reasoning-30b, so the
# API-visible model name is the short alias, not the full HuggingFace id. Clients
# still configured with `lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit`
# will get a 404 until you re-point them:
#
#   eval "$(./run.sh --env reasoning-30b)"
#
# Env overrides (MODEL, HOST, PORT, KV_QUANT, NO_CAFFEINATE, NO_WAIT, ...) and any
# extra `vllm-mlx serve` flags still work; per-model tuning now lives in models.json.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${MODEL:-}" ]]; then
  printf '\033[33mwarning:\033[0m MODEL=%s is ignored — models come from %s/models.json\n' \
    "$MODEL" "$REPO" >&2
fi

exec "$REPO/run.sh" reasoning-30b "$@"
