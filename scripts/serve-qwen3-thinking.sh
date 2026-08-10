#!/usr/bin/env bash
# Serve lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit on http://127.0.0.1:8000.
#
# Tuned for driving Claude Code. It replaces an earlier script that served
# Qwen/Qwen3-32B-MLX-bf16, which was the wrong shape for this workload on three counts:
#
#   1. SPEED. Qwen3-32B-MLX-bf16 is 61 GB of *dense* weights, so every token streams all
#      61 GB through memory: measured 1.4-5.4 tok/s here. This checkpoint is a 128-expert
#      MoE with 8 experts active, i.e. ~3B active params per token, at 8-bit. Same class
#      of model, a fraction of the bytes per token.
#
#   2. CONTEXT. Qwen3-32B-MLX-bf16 has max_position_embeddings=40960 and rope_scaling=None
#      — a hard 40k ceiling. This checkpoint is 262144 native. That is the whole reason
#      for the swap; no amount of server tuning lifts the 40k cap.
#
#   3. CONCURRENCY. Without --continuous-batching the server runs SimpleEngine, which
#      serializes every generation behind one asyncio lock (Metal command buffers must be
#      serialized). Claude Code issues 2-3 concurrent requests — main loop, subagents, and
#      the tool-permission check — so the extras sat in the admission queue until
#      VLLM_MLX_SIMPLE_ENGINE_QUEUE_TIMEOUT_S (120 s) fired and returned 503. That is the
#      "Anthropic stream rejected (busy)" / "temporarily unavailable, so auto mode cannot
#      determine the safety of Bash" pair in the logs.
#
#      --continuous-batching also flips on the real scheduler. This matters more than it
#      looks: in SimpleEngine mode cli.py leaves scheduler_config = None, so --max-num-seqs,
#      --cache-memory-mb, --kv-cache-quantization*, --max-kv-size, --chunked-prefill-tokens
#      and every other cache flag are parsed and then silently dropped. The previous script
#      passed --max-num-seqs 2 --cache-memory-mb 8192 into exactly that void.
#
# Measured on this machine (M5 Max, 128 GiB), 8-bit weights + 8-bit KV, on a REALISTIC
# Claude Code turn (11330 prompt tokens, 20 tools) — not a toy prompt:
#
#   decode, 1 stream        68 tok/s          (bf16 32B was 1.4-5.4 tok/s at this size)
#   decode, 3 streams       31 tok/s each, 93 tok/s aggregate, 0 rejected
#   TTFT, first turn        5.9 s
#   TTFT, later turns       0.32 s            (prefix cache; was 50-120 s of queue wait)
#   resident weights        32.6 GB           (was 61 GB), 39.7 GB peak under load
#
# Two things measured and rejected, so nobody re-litigates them:
#   --stream-interval 2/4/8   no effect (65-67 tok/s across all values); default 1 kept
#                             because it streams smoothest. NB: benchmarks that count SSE
#                             chunks instead of usage.output_tokens will "show" a 2x win
#                             here — that is the chunk batching, not real throughput.
#   KV_QUANT=0 (bf16 KV)      34.1 vs 34.0 tok/s — identical speed, 2x the memory. 8-bit
#                             KV stays on because it doubles how many conversations fit
#                             in the prefix cache.
#
# Prefill is the one place to keep expectations honest — attention is O(n^2), so each
# doubling of context roughly quadruples time-to-first-token on a COLD prompt:
#
#     8k -> 2.5 s | 16k -> 5.9 s | 32k -> 19.3 s | 64k -> 74 s | 131k -> ~5 min
#
# That is survivable only because the prefix cache makes it a once-per-session cost.
# Measured on the Claude Code pattern (same conversation, one turn appended):
#
#     turn 1 (cold, 32k)  19.6 s
#     turn 2 (cached)      0.56 s      <- 35x, 63694 tokens reused
#
# So: opening a huge file hurts once; every turn after it is nearly free. If you paste
# a 200k-token context expect to wait minutes for that first token.
#
# Usage:
#   ./scripts/serve-qwen3-thinking.sh [extra vllm-mlx serve flags...]
#
# Env overrides:
#   MODEL, HOST, PORT, MAX_NUM_SEQS, CACHE_MEMORY_MB, MAX_TOKENS, MAX_REQUEST_TOKENS,
#   KV_QUANT=0        keep the KV cache at bf16 (2x the memory, slightly better quality)
#   NO_CAFFEINATE=1   don't hold a sleep assertion for the server's lifetime
#   NO_WAIT=1         don't run the background readiness poller
#
# Examples:
#   PORT=8001 ./scripts/serve-qwen3-thinking.sh
#   KV_QUANT=0 MAX_NUM_SEQS=2 ./scripts/serve-qwen3-thinking.sh
#   ./scripts/serve-qwen3-thinking.sh --api-key "$(security find-generic-password -w -s vllm)"

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"

MODEL="${MODEL:-lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# KV geometry: 2 (K+V) x 48 layers x 4 kv-heads x 128 head-dim = 49152 values/token,
# so 96 KiB/token at bf16 and ~48 KiB/token with 8-bit KV quantization. A *full* 262144
# sequence is therefore ~12.6 GiB quantized. mlx grows the KV cache lazily, so this is a
# worst case, not a reservation — 4 slots only cost 50 GiB if all four run to full context.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# Pin the prefix cache rather than letting it auto-take ~20% of RAM (~25 GB here).
CACHE_MEMORY_MB="${CACHE_MEMORY_MB:-12288}"
# Default output cap when a client doesn't ask for one.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# Ceiling on what a client may request. The stock 32768 silently clamps large requests.
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-262144}"

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
note() { printf '\033[36m%s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }

# --- preflight ---------------------------------------------------------------

[[ -x "$PY" ]] || die "no venv interpreter at $PY — run 'cd $REPO && uv sync' first"

# Every .pth in this venv's site-packages carries the macOS UF_HIDDEN flag, and CPython
# 3.13's site.addpackage skips hidden .pth files, so the editable install never registers
# and the `vllm-mlx` console script is inert. Calling the module with PYTHONPATH works.
if ! PYTHONPATH="$REPO" "$PY" -c 'import vllm_mlx' 2>/dev/null; then
  SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
  warn "the venv cannot import vllm_mlx even with PYTHONPATH set."
  ls -lO "$SITE_PACKAGES"/*.pth 2>/dev/null >&2 || true
  cat >&2 <<EOF

If those .pth files are flagged 'hidden', CPython skips them and the editable install
never takes effect. Permanent fix:

    chflags nohidden $SITE_PACKAGES/*.pth

Then '$REPO/.venv/bin/vllm-mlx' works directly too.
EOF
  die "vllm_mlx is not importable"
fi

# -sTCP:LISTEN matters: a bare `lsof -ti tcp:$PORT` also matches CLOSED/CLOSE_WAIT
# leftovers from a suspended or half-dead server, which do not actually block a bind.
if PORT_PIDS="$(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null)" && [[ -n "$PORT_PIDS" ]]; then
  # shellcheck disable=SC2086
  ps -o pid=,command= -p $PORT_PIDS >&2 || true
  die "port $PORT is already in use by PID(s): $(echo "$PORT_PIDS" | tr '\n' ' ')
     kill it first:  kill $(echo "$PORT_PIDS" | tr '\n' ' ')"
fi

# Serve from the local HF cache when possible, so a flaky network can't stall a startup
# that is already several minutes long. Probing with the exact call the server makes
# (snapshot_download(local_files_only=True)) is the only honest test: having every weight
# shard is NOT enough — huggingface_hub raises IncompleteSnapshotError if any file from
# the revision is absent.
OFFLINE_FLAG=()
if "$PY" - "$MODEL" <<'PROBE' 2>/dev/null
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_files_only=True)
PROBE
then
  OFFLINE_FLAG=(--offline)
  note "complete snapshot in the local cache — serving --offline"
else
  warn "local snapshot is incomplete or absent; startup will contact huggingface.co."
  warn "  to make --offline work, complete it once:  $REPO/.venv/bin/hf download $MODEL"
fi

KV_FLAGS=(--kv-cache-quantization --kv-cache-quantization-bits 8)
KV_PER_TOKEN_KIB=48
if [[ "${KV_QUANT:-1}" == "0" ]]; then
  KV_FLAGS=()
  KV_PER_TOKEN_KIB=96
fi

# Advisory only — never sudo from a launcher.
RAM_GIB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
WIRED_MB="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
FULL_CTX_GIB=$(( 262144 * KV_PER_TOKEN_KIB / 1048576 ))
if [[ "$WIRED_MB" == "0" ]]; then
  note "GPU wired-memory limit: macOS default (~75% of ${RAM_GIB} GiB, i.e. ~$(( RAM_GIB * 3 / 4 )) GiB)"
  note "  ~32 GB weights + up to ${MAX_NUM_SEQS} x ${FULL_CTX_GIB} GiB KV + ${CACHE_MEMORY_MB} MB cache."
  note "  KV grows lazily, so that ceiling only lands if every slot reaches full 262144 context."
  note "  To give the worst case room, raise the limit until the next reboot:"
  note "      sudo sysctl iogpu.wired_limit_mb=$(( RAM_GIB * 7 / 8 * 1024 ))"
else
  note "GPU wired-memory limit: ${WIRED_MB} MB (explicitly set)"
fi

# --- readiness reporter ------------------------------------------------------
# cli.py loads the weights BEFORE uvicorn binds the socket, so the port stays shut until
# the model is resident. /health is unauthenticated, so this keeps working with --api-key.
if [[ -z "${NO_WAIT:-}" ]]; then
  # `exec` below replaces this shell, so an EXIT trap would never fire and a failed load
  # would orphan the poller for its full timeout. exec preserves the PID, so $$ IS the
  # server's PID — the poller watches it and leaves the moment the server dies.
  SELF=$$
  (
    start=$SECONDS
    while kill -0 "$SELF" 2>/dev/null && (( SECONDS - start < 900 )); do
      if body="$(curl -sf --max-time 5 "http://$HOST:$PORT/health" 2>/dev/null)"; then
        printf '\n\033[32mready\033[0m after %ds — %s\n' "$(( SECONDS - start ))" "$body" >&2

        # Warm up before handing the server to a human. On its FIRST prefix-cache hit the
        # batched engine logs "Detected MLX stream/thread mismatch on worker step", runs
        # cache recovery, and reschedules that request WITHOUT its cached prefix — a full
        # re-prefill. Measured cost on an 11k-token turn: 4.4 s instead of 0.32 s. Firing
        # the pair here means the engine has already settled by the first real request.
        # Two calls, because the mismatch only surfaces when the second one hits cache.
        # The filler is NOT decoration: a 13-token prompt logs `stored=False` (too small to
        # enter the prefix cache), so the second call would miss and the mismatch would
        # ambush the first real request instead. ~2k tokens is comfortably over the bar.
        # Best-effort: any failure (e.g. --api-key set) is ignored.
        # shellcheck disable=SC2183
        WARM_FILLER="$(printf 'The quick brown fox jumps over the lazy dog while refactoring a distributed cache layer. %.0s' $(seq 1 120))"
        WARM_BODY="{\"model\":\"$MODEL\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"$WARM_FILLER Reply: ok\"}]}"
        for _ in 1 2; do
          curl -sf --max-time 120 -o /dev/null \
            -H 'content-type: application/json' \
            -d "$WARM_BODY" "http://$HOST:$PORT/v1/messages" 2>/dev/null || true
        done
        printf '\033[32mwarm\033[0m — engine settled, prefix cache primed\n' >&2

        printf 'try:  curl -s http://%s:%s/v1/models\n\n' "$HOST" "$PORT" >&2
        exit 0
      fi
      sleep 5
    done
    kill -0 "$SELF" 2>/dev/null &&
      warn "server still not answering /health after 15 minutes — check the log above"
  ) &
fi

# --- run ---------------------------------------------------------------------

RUNNER=(env "PYTHONPATH=$REPO" "$PY" -m vllm_mlx.cli)
# caffeinate keeps macOS awake for the whole session; it execs the child, so signals and
# exit status pass through.
[[ -n "${NO_CAFFEINATE:-}" ]] || RUNNER=(caffeinate -dimsu "${RUNNER[@]}")

note "starting $MODEL on http://$HOST:$PORT (port opens once the weights are resident)"

# --max-kv-size is deliberately NOT passed: any value > 0 switches mlx to a RotatingKVCache,
# which silently drops the oldest tokens once the window fills AND disqualifies the prefix
# cache (snapshotting a rotating cache isn't sound). Leaving it unset keeps the full 262144.
exec "${RUNNER[@]}" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --continuous-batching \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --cache-memory-mb "$CACHE_MEMORY_MB" \
  --max-tokens "$MAX_TOKENS" \
  --max-request-tokens "$MAX_REQUEST_TOKENS" \
  ${KV_FLAGS[@]+"${KV_FLAGS[@]}"} \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  ${OFFLINE_FLAG[@]+"${OFFLINE_FLAG[@]}"} \
  "$@"
