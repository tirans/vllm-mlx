#!/usr/bin/env bash
# One launcher for every local model. Reads ./models.json; nothing is hard-coded here.
#
#   ./run.sh --help              list the models and what they are
#   ./run.sh reasoning-30b       serve one, correctly tuned, on its configured port
#   ./run.sh --multi a b         serve several on one port, loaded lazily
#   ./run.sh --env reasoning-30b print the client env vars (eval this)
#   ./run.sh --stop --all        stop every running server
#
# Anything after the first flag run.sh does not recognise is handed to
# `vllm-mlx serve` verbatim, e.g.  ./run.sh reasoning-30b --api-key hunter2
#
# ---------------------------------------------------------------------------
# WHY THE DEFAULTS IN models.json LOOK THE WAY THEY DO
#
# --continuous-batching is not optional. Without it cli.py:252 leaves
# scheduler_config = None, so --max-num-seqs, --cache-memory-mb, every
# --kv-cache-quantization* flag, --max-kv-size and --chunked-prefill-tokens are
# parsed and then silently dropped — they only exist on SchedulerConfig. It also
# swaps SimpleEngine (one asyncio lock around ALL generation, because Metal
# command buffers must be serialized) for the real scheduler. Claude Code issues
# 2-3 concurrent requests — main loop, subagents, tool-permission check — and on
# SimpleEngine the extras sit in the admission queue until
# VLLM_MLX_SIMPLE_ENGINE_QUEUE_TIMEOUT_S (120 s) fires and returns 503. That is
# the "Anthropic stream rejected (busy)" / "temporarily unavailable, so auto mode
# cannot determine the safety of Bash" pair in the logs.
#
# --max-kv-size is never passed. Any value > 0 switches mlx to a RotatingKVCache,
# which silently drops the oldest tokens once the window fills AND disqualifies
# prefix caching (snapshotting a rotating cache isn't sound).
#
# Measured for reasoning-30b on M5 Max / 128 GiB, 8-bit weights + 8-bit KV, on a
# REALISTIC Claude Code turn (11330 prompt tokens, 20 tools) — not a toy prompt:
#
#   decode, 1 stream        68 tok/s      (dense bf16 32B was 1.4-5.4 tok/s here)
#   decode, 3 streams       31 tok/s each, 93 aggregate, 0 rejected
#   TTFT, first turn        5.9 s
#   TTFT, later turns       0.32 s        (prefix cache; was 50-120 s of queue wait)
#   resident weights        32.6 GB, 39.7 GB peak under load
#
# Measured and rejected, so nobody re-litigates them:
#   --stream-interval 2/4/8   no effect (65-67 tok/s at every value); 1 streams
#                             smoothest. NB: benchmarks that count SSE chunks
#                             instead of usage.output_tokens will "show" a 2x win
#                             here — that is chunk batching, not throughput.
#   KV_QUANT=0 (bf16 KV)      34.1 vs 34.0 tok/s — same speed, 2x the memory.
#                             8-bit KV stays on: it doubles how many conversations
#                             fit in the prefix cache.
#
# Prefill is O(n^2), so each doubling of context roughly quadruples COLD TTFT:
#
#     8k -> 2.5 s | 16k -> 5.9 s | 32k -> 19.3 s | 64k -> 74 s | 131k -> ~5 min
#
# Survivable only because the prefix cache makes it a once-per-session cost:
# turn 1 (cold, 32k) 19.6 s, turn 2 (cached) 0.56 s — 35x, 63694 tokens reused.
# Paste a 200k-token context and expect minutes before the first token.
# ---------------------------------------------------------------------------
#
# Env overrides:
#   MODELS_JSON       path to the catalog (default: ./models.json)
#   HOST, PORT        override the configured host/port
#   KV_QUANT=0        keep the KV cache at bf16 (2x memory, marginally better quality)
#   NO_CAFFEINATE=1   don't hold a sleep assertion for the server's lifetime
#   NO_WAIT=1         don't run the background readiness poller / warm-up

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${MODELS_JSON:-$REPO/models.json}"
PY="$REPO/.venv/bin/python"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
note() { printf '\033[36m%s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }

command -v jq >/dev/null 2>&1 || die "jq is required (brew install jq)"
[[ -f "$CONFIG" ]] || die "no catalog at $CONFIG"
jq -e . "$CONFIG" >/dev/null 2>&1 || die "$CONFIG is not valid JSON"

# --- catalog access ----------------------------------------------------------

aliases() { jq -r '.models | keys_unsorted[]' "$CONFIG"; }

has_alias() { jq -e --arg a "$1" '.models | has($a)' "$CONFIG" >/dev/null 2>&1; }

# An API-visible model id back to a catalog alias. Servers started before this
# script existed report the full HF repo id (no --served-model-name), so match on
# `source` too — otherwise a legacy server looks like an unknown model and the
# merge path would restart it for no reason. Prints nothing when there is no match.
alias_for_id() {
  if has_alias "$1"; then printf '%s\n' "$1"; return 0; fi
  jq -r --arg s "$1" \
    'first((.models | to_entries[] | select(.value.source == $s) | .key), empty)' "$CONFIG"
}

# Per-model value, falling back to .defaults. Not `//` — that would swallow a
# deliberate `false` or `0`.
mget() {
  jq -r --arg a "$1" --arg k "$2" '
    if (.models[$a] // {}) | has($k) then .models[$a][$k]
    elif (.defaults // {}) | has($k) then .defaults[$k]
    else empty end' "$CONFIG"
}

multiget() { jq -r --arg k "$1" '(.multi // {})[$k] // empty' "$CONFIG"; }

default_alias() {
  jq -r 'first((.models | to_entries[] | select(.value.default == true) | .key), (.models | keys_unsorted[0]))' "$CONFIG"
}

# --- weight probing ----------------------------------------------------------
# One python start for the whole catalog: importing huggingface_hub costs ~1 s,
# and --help would otherwise pay it once per model.
#
# Three states, because "has the weights" and "can run --offline" are NOT the
# same question:
#
#   ok       snapshot_download(local_files_only=True) succeeds with no
#            allow_patterns — which is exactly what the server does in offline
#            mode (utils/download.py:85), so --offline is safe.
#   weights  every file mlx actually loads is cached (LLM_ALLOW_PATTERNS:
#            *.json, model*.safetensors, tokenizer.model, *.jinja, ...) but the
#            strict probe raises IncompleteSnapshotError over some file that is
#            in the revision and not on disk — typically README.md and
#            .gitattributes, which `hf download` skipped or which were never
#            fetched. The model serves fine; only --offline would break.
#   missing  no usable weights.
#
# Absolute paths are LM-Studio-style flat dirs that were never in hub layout.
# They short-circuit the whole question: utils/download.py:78 returns an existing
# local path before it ever consults the hub, so a safetensors check is enough.
probe_sources() {
  [[ $# -gt 0 ]] || return 0
  if [[ ! -x "$PY" ]]; then
    for s in "$@"; do printf '%s\tunknown\n' "$s"; done
    return 0
  fi
  PYTHONPATH="$REPO" "$PY" - "$@" 2>/dev/null <<'PROBE' || for s in "$@"; do printf '%s\tunknown\n' "$s"; done
import glob, os, sys

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from vllm_mlx.utils.download import LLM_ALLOW_PATTERNS as PATTERNS
except Exception:
    PATTERNS = ["*.json", "model*.safetensors", "*.py", "tokenizer.model",
                "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"]


def probe(src):
    if os.path.isabs(src):
        if os.path.isdir(src) and glob.glob(os.path.join(src, "*.safetensors")):
            return "ok"
        return "missing"
    if snapshot_download is None:
        return "missing"
    try:
        snapshot_download(src, local_files_only=True)
        return "ok"
    except Exception:
        pass
    try:
        snapshot_download(src, local_files_only=True, allow_patterns=PATTERNS)
        return "weights"
    except Exception:
        return "missing"


for src in sys.argv[1:]:
    print("%s\t%s" % (src, probe(src)))
PROBE
}

# Probe a single source; echoes ok|missing|unknown.
probe_one() { probe_sources "$1" | awk -F'\t' 'NR==1{print $2}'; }

# --- help / list -------------------------------------------------------------

print_help() {
  local def; def="$(default_alias)"
  local srcs=() a
  while IFS= read -r a; do srcs+=("$(mget "$a" source)"); done < <(aliases)

  local status_tsv; status_tsv="$(probe_sources ${srcs[@]+"${srcs[@]}"})"

  cat >&2 <<EOF

  vllm-mlx launcher — models come from $(basename "$CONFIG")

  usage:
    ./run.sh                             serve the default (marked *, below)
    ./run.sh <name> [serve flags...]     serve one model
    ./run.sh --multi <name> [<name>...]  serve several on one port (lazy-loaded)
    ./run.sh --multi --all               ... every downloaded model
    ./run.sh --env <name>                print client env vars
    ./run.sh --stop <name>|--all         stop a running server
    ./run.sh --list                      machine-readable catalog (TSV)

  options:
    --port N        override the model's configured port
    --no-merge      if the port is busy, fail instead of merging (see below)

  models:
EOF

  printf '  %-14s %-16s %8s %9s %6s  %s\n' NAME LABEL SIZE CONTEXT PORT STATUS >&2
  while IFS= read -r a; do
    local src label size ctx port st mark
    src="$(mget "$a" source)"
    label="$(mget "$a" label)"
    size="$(mget "$a" size_gb)"
    ctx="$(mget "$a" context)"
    port="$(mget "$a" port)"
    st="$(printf '%s\n' "$status_tsv" | awk -F'\t' -v s="$src" '$1==s{print $2; exit}')"
    case "${st:-unknown}" in
      ok)      st=$'\033[32m✓ ready (offline)\033[0m' ;;
      weights) st=$'\033[32m✓ ready\033[0m — needs network at startup' ;;
      missing) st=$'\033[33m↓ downloads on first use\033[0m' ;;
      *)       st='? (no venv — run ./scripts/setup.sh)' ;;
    esac
    mark=' '; [[ "$a" == "$def" ]] && mark='*'
    printf '%s %-14s %-16s %7sGB %9s %6s  %b\n' "$mark" "$a" "$label" "$size" "$ctx" "$port" "$st" >&2
  done < <(aliases)

  cat >&2 <<EOF

  (* = default)  descriptions:
EOF
  while IFS= read -r a; do
    printf '    %-14s %s\n' "$a" "$(mget "$a" description)" >&2
  done < <(aliases)

  cat >&2 <<EOF

  point a client at it:
    eval "\$(./run.sh --env $def)"      then run \`claude\` or any OpenAI SDK client

  serving two models on one port lets Claude Code use a cheap helper model:
    ./run.sh --multi $def tiny-1b
    export ANTHROPIC_MODEL=$def ANTHROPIC_SMALL_FAST_MODEL=tiny-1b

  env overrides: HOST PORT KV_QUANT=0 NO_CAFFEINATE=1 NO_WAIT=1 MODELS_JSON=<path>

EOF
}

print_list() {
  local srcs=() a
  while IFS= read -r a; do srcs+=("$(mget "$a" source)"); done < <(aliases)
  local status_tsv; status_tsv="$(probe_sources ${srcs[@]+"${srcs[@]}"})"
  printf 'name\tsource\tport\tsize_gb\tcontext\tstatus\n'
  while IFS= read -r a; do
    local src st
    src="$(mget "$a" source)"
    st="$(printf '%s\n' "$status_tsv" | awk -F'\t' -v s="$src" '$1==s{print $2; exit}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$a" "$src" "$(mget "$a" port)" "$(mget "$a" size_gb)" "$(mget "$a" context)" "${st:-unknown}"
  done < <(aliases)
}

# --- env block ---------------------------------------------------------------

print_env() {
  local a="$1" host port
  host="${HOST:-$(mget "$a" host)}"
  port="${PORT_OVERRIDE:-${PORT:-$(mget "$a" port)}}"
  cat <<EOF
export ANTHROPIC_BASE_URL=http://$host:$port
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_MODEL=$a
export OPENAI_BASE_URL=http://$host:$port/v1
export OPENAI_API_KEY=dummy
export OPENAI_MODEL=$a
EOF
}

# --- process control ---------------------------------------------------------

# -sTCP:LISTEN matters: a bare `lsof -ti tcp:N` also matches CLOSED/CLOSE_WAIT
# leftovers from a suspended or half-dead server, which do not block a bind.
port_pids() { lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null || true; }

served_ids() {
  curl -sf --max-time 5 "http://${2:-127.0.0.1}:$1/v1/models" 2>/dev/null \
    | jq -r '.data[].id' 2>/dev/null || true
}

stop_port() {
  local port="$1" pids ids
  pids="$(port_pids "$port")"
  [[ -n "$pids" ]] || return 1
  # Name what is about to die. A port in models.json is not proof the server on it
  # was started by this script, and "stopped the server on port 8000" is not enough
  # information to notice you just killed someone else's session.
  ids="$(served_ids "$port")"
  if [[ -n "$ids" ]]; then
    note "port $port serves: $(echo "$ids" | tr '\n' ' ')— stopping it"
  else
    # shellcheck disable=SC2086
    ps -o pid=,command= -p $pids >&2 || true
    warn "port $port is held by a process that is not answering /v1/models — stopping it anyway"
  fi
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  local i=0
  while [[ -n "$(port_pids "$port")" ]] && (( i < 30 )); do sleep 1; i=$((i+1)); done
  pids="$(port_pids "$port")"
  if [[ -n "$pids" ]]; then
    warn "port $port did not free after 30 s — SIGKILL"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
  return 0
}

do_stop() {
  local ports=() a
  if [[ "${1:-}" == "--all" ]]; then
    while IFS= read -r a; do ports+=("$(mget "$a" port)"); done < <(aliases)
    ports+=("$(multiget port)")
  else
    has_alias "$1" || die "unknown model: $1 (try ./run.sh --help)"
    ports=("${PORT_OVERRIDE:-$(mget "$1" port)}")
  fi
  local stopped=0 p seen=""
  for p in "${ports[@]}"; do
    [[ -n "$p" ]] || continue
    case " $seen " in *" $p "*) continue ;; esac
    seen="$seen $p"
    if stop_port "$p"; then note "stopped the server on port $p"; stopped=$((stopped+1)); fi
  done
  [[ $stopped -gt 0 ]] || note "nothing was listening"
}

# --- weights -----------------------------------------------------------------

ensure_weights() {   # alias -> echoes "offline" when --offline is safe
  local a="$1" src st
  src="$(mget "$a" source)"
  st="$(probe_one "$src")"

  if [[ "$st" == ok ]]; then echo offline; return 0; fi

  if [[ "$src" == /* ]]; then
    die "$a points at $src, which has no *.safetensors — fix models.json or fetch it yourself"
  fi

  if [[ "$st" == unknown ]]; then
    warn "cannot probe the cache (no venv?) — startup will contact huggingface.co"
    return 0
  fi

  if [[ "$st" == weights ]]; then
    # The weights are all here; the strict snapshot is short a README or
    # .gitattributes. Completing it is a few KB and buys --offline, so try —
    # but never let a flaky network block a server that can already run.
    note "$a has its weights but an incomplete snapshot — fetching the few missing files"
    if [[ -x "$REPO/.venv/bin/hf" ]] && "$REPO/.venv/bin/hf" download "$src" >/dev/null 2>&1 \
       && [[ "$(probe_one "$src")" == ok ]]; then
      echo offline
    else
      warn "could not complete the snapshot — serving without --offline (startup will contact huggingface.co)"
    fi
    return 0
  fi

  note "$a is not in the local cache — downloading $src (~$(mget "$a" size_gb) GB)"
  [[ -x "$REPO/.venv/bin/hf" ]] || die "no $REPO/.venv/bin/hf — run ./scripts/setup.sh first"
  "$REPO/.venv/bin/hf" download "$src" || die "download failed"
  case "$(probe_one "$src")" in
    ok)      note "download complete"; echo offline ;;
    weights) note "download complete" ;;
    *)       die "$src still has no usable weights after download" ;;
  esac
}

# --- flag assembly -----------------------------------------------------------

KV_FLAGS=()
KV_PER_TOKEN_KIB=96
init_kv_flags() {
  local a="$1"
  if [[ "${KV_QUANT:-1}" != "0" ]] && [[ "$(mget "$a" kv_cache_quantization)" == "true" ]]; then
    KV_FLAGS=(--kv-cache-quantization --kv-cache-quantization-bits "$(mget "$a" kv_cache_quantization_bits)")
    KV_PER_TOKEN_KIB=48
  else
    KV_FLAGS=()
    KV_PER_TOKEN_KIB=96
  fi
}

ram_advisory() {   # alias-or-"" , max context
  local ctx="$1" ram_gib wired full_ctx_gib
  ram_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  wired="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
  full_ctx_gib=$(( ctx * KV_PER_TOKEN_KIB / 1048576 ))
  if [[ "$wired" == "0" ]]; then
    note "GPU wired-memory limit: macOS default (~75% of ${ram_gib} GiB, i.e. ~$(( ram_gib * 3 / 4 )) GiB)"
    note "  KV worst case is ${full_ctx_gib} GiB per slot at full context; mlx grows it lazily."
    note "  To give that worst case room until the next reboot:"
    # *1024 before /8: dividing first truncates, and on a machine whose GiB count
    # is not a multiple of 8 that silently under-reports the limit by up to 1 GiB.
    note "      sudo sysctl iogpu.wired_limit_mb=$(( ram_gib * 7 * 1024 / 8 ))"
  else
    note "GPU wired-memory limit: ${wired} MB (explicitly set)"
  fi
}

preflight() {
  [[ -x "$PY" ]] || die "no venv interpreter at $PY — run ./scripts/setup.sh first"
  # Every .pth in this venv's site-packages can carry the macOS UF_HIDDEN flag, and
  # CPython 3.13's site.addpackage skips hidden .pth files, so the editable install
  # never registers and the `vllm-mlx` console script is inert. PYTHONPATH works.
  if ! PYTHONPATH="$REPO" "$PY" -c 'import vllm_mlx' 2>/dev/null; then
    local sp; sp="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
    warn "the venv cannot import vllm_mlx even with PYTHONPATH set."
    [[ -n "$sp" ]] && ls -lO "$sp"/*.pth 2>/dev/null >&2
    [[ -n "$sp" ]] && printf '\nIf those .pth files are flagged hidden, CPython skips them. Fix:\n\n    chflags nohidden %s/*.pth\n\n' "$sp" >&2
    die "vllm_mlx is not importable"
  fi
}

# --- readiness reporter ------------------------------------------------------
# cli.py loads the weights BEFORE uvicorn binds, so the port stays shut until the
# model is resident. /health is unauthenticated, so this survives --api-key.
start_reporter() {
  local host="$1" port="$2" warm_model="$3" want_warm="$4"
  [[ -z "${NO_WAIT:-}" ]] || return 0
  # `exec` below replaces this shell, so an EXIT trap would never fire and a failed
  # load would orphan the poller for its full timeout. exec preserves the PID, so
  # $$ IS the server's PID — the poller watches it and leaves when the server dies.
  local self=$$
  (
    start=$SECONDS
    while kill -0 "$self" 2>/dev/null && (( SECONDS - start < 900 )); do
      if body="$(curl -sf --max-time 5 "http://$host:$port/health" 2>/dev/null)"; then
        printf '\n\033[32mready\033[0m after %ds — %s\n' "$(( SECONDS - start ))" "$body" >&2

        if [[ "$want_warm" == "1" ]]; then
          # On its FIRST prefix-cache hit the batched engine logs "Detected MLX
          # stream/thread mismatch on worker step", runs cache recovery, and
          # reschedules that request WITHOUT its cached prefix — a full re-prefill.
          # Measured cost on an 11k-token turn: 4.4 s instead of 0.32 s. Firing the
          # pair here means the engine has settled before the first real request.
          # TWO calls, because the mismatch only surfaces when the second one HITS
          # cache. The filler is not decoration: a 13-token prompt logs stored=False
          # (too small to enter the prefix cache), so the second call would miss and
          # the mismatch would ambush the first real request instead. ~2k tokens is
          # comfortably over the bar. Best effort — failures (e.g. --api-key) ignored.
          # shellcheck disable=SC2183
          filler="$(printf 'The quick brown fox jumps over the lazy dog while refactoring a distributed cache layer. %.0s' $(seq 1 120))"
          payload="{\"model\":\"$warm_model\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"$filler Reply: ok\"}]}"
          for _ in 1 2; do
            curl -sf --max-time 180 -o /dev/null \
              -H 'content-type: application/json' \
              -d "$payload" "http://$host:$port/v1/messages" 2>/dev/null || true
          done
          printf '\033[32mwarm\033[0m — engine settled, prefix cache primed\n' >&2
        fi

        printf '\n\033[36mpoint a client at it:\033[0m\n' >&2
        printf '    export ANTHROPIC_BASE_URL=http://%s:%s\n' "$host" "$port" >&2
        printf '    export ANTHROPIC_AUTH_TOKEN=dummy\n' >&2
        printf '    export ANTHROPIC_MODEL=%s\n' "$warm_model" >&2
        printf '    export OPENAI_BASE_URL=http://%s:%s/v1\n' "$host" "$port" >&2
        printf '    export OPENAI_API_KEY=dummy\n\n' >&2
        printf '    curl -s http://%s:%s/v1/models | jq\n\n' "$host" "$port" >&2
        exit 0
      fi
      sleep 5
    done
    kill -0 "$self" 2>/dev/null &&
      warn "server still not answering /health after 15 minutes — check the log above"
  ) &
}

launch() {   # all remaining args are passed to `vllm-mlx serve`
  local runner=(env "PYTHONPATH=$REPO" "$PY" -m vllm_mlx.cli)
  # caffeinate keeps macOS awake for the session; it execs the child, so signals
  # and exit status pass through.
  [[ -n "${NO_CAFFEINATE:-}" ]] || runner=(caffeinate -dimsu "${runner[@]}")
  exec "${runner[@]}" serve "$@"
}

# --- single-model serve ------------------------------------------------------

serve_single() {
  local a="$1"; shift
  local host port src offline args
  host="${HOST:-$(mget "$a" host)}"
  port="${PORT_OVERRIDE:-${PORT:-$(mget "$a" port)}}"
  src="$(mget "$a" source)"

  preflight
  offline="$(ensure_weights "$a")"
  init_kv_flags "$a"
  ram_advisory "$(mget "$a" context)"

  # --served-model-name pins the API-visible name to the alias, so clients use the
  # same short name whether this model is served alone or merged into a registry
  # (server.py:3350; requests naming anything else are rejected at server.py:1827).
  args=(
    "$src"
    --served-model-name "$a"
    --host "$host" --port "$port"
    --max-num-seqs "$(mget "$a" max_num_seqs)"
    --cache-memory-mb "$(mget "$a" cache_memory_mb)"
    --max-tokens "$(mget "$a" max_tokens)"
    --max-request-tokens "$(mget "$a" max_request_tokens)"
    --gpu-memory-utilization "$(mget "$a" gpu_memory_utilization)"
  )
  [[ "$(mget "$a" continuous_batching)" == "true" ]] && args+=(--continuous-batching)
  args+=(${KV_FLAGS[@]+"${KV_FLAGS[@]}"})

  local rp tp
  rp="$(mget "$a" reasoning_parser)"; [[ -n "$rp" ]] && args+=(--reasoning-parser "$rp")
  tp="$(mget "$a" tool_call_parser)"
  if [[ -n "$tp" ]]; then
    args+=(--tool-call-parser "$tp")
    [[ "$(mget "$a" enable_auto_tool_choice)" == "true" ]] && args+=(--enable-auto-tool-choice)
  fi
  [[ "$offline" == "offline" ]] && args+=(--offline)

  local want_warm=0
  [[ "$(mget "$a" warmup)" == "true" ]] && want_warm=1
  start_reporter "$host" "$port" "$a" "$want_warm"
  note "starting $a ($src) on http://$host:$port"
  note "  the port opens once the weights are resident"
  launch "${args[@]}" "$@"
}

# --- multi-model serve -------------------------------------------------------

serve_multi() {
  local port="$1"; shift
  local sel=() a
  while [[ $# -gt 0 && "$1" != "--" ]]; do sel+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift

  local host; host="${HOST:-$(mget "${sel[0]}" host)}"
  preflight

  # Parsers are process-global: RegisteredModel (model_registry.py:145) has no
  # per-model reasoning_parser / tool_call_parser. Everything merged onto one port
  # shares the .multi block's pair, so say so when a model wanted something else.
  local mrp mtp
  mrp="$(multiget reasoning_parser)"
  mtp="$(multiget tool_call_parser)"
  for a in "${sel[@]}"; do
    local rp tp
    rp="$(mget "$a" reasoning_parser)"; tp="$(mget "$a" tool_call_parser)"
    [[ -n "$rp" && "$rp" != "$mrp" ]] && warn "$a wants --reasoning-parser $rp but shares '$mrp' on a merged port"
    [[ -n "$tp" && "$tp" != "$mtp" ]] && warn "$a wants --tool-call-parser $tp but shares '$mtp' on a merged port"
  done

  local all_offline=1 max_req=0 max_tok=0
  for a in "${sel[@]}"; do
    [[ "$(ensure_weights "$a")" == "offline" ]] || all_offline=0
    local mr mt
    mr="$(mget "$a" max_request_tokens)"; (( mr > max_req )) && max_req=$mr
    mt="$(mget "$a" max_tokens)";         (( mt > max_tok )) && max_tok=$mt
  done
  (( max_tok > max_req )) && max_tok=$max_req

  init_kv_flags "${sel[0]}"
  ram_advisory "$max_req"

  # --models-config is read with yaml.safe_load (model_registry.py:279) and YAML 1.2
  # is a superset of JSON, so a plain JSON document is a valid registry file. No
  # YAML emitter needed. Shape validated against load_registry_config.
  local reg="${TMPDIR:-/tmp}/vllm-mlx-registry-$port.json"
  jq -n --slurpfile cfg "$CONFIG" --args '
    $cfg[0] as $c
    | { manager: {
          memory_budget_gb: ($c.multi.memory_budget_gb // 64),
          contention_policy: ($c.multi.contention_policy // {strategy: "wait_then_fail"})
        },
        models: [ $ARGS.positional[] as $a
                  | $c.models[$a]
                  | { name: $a, source: .source, preload: false,
                      estimated_memory_gb: (.size_gb // 8) } ] }
  ' "${sel[@]}" > "$reg"
  note "registry: $reg"

  local args=(
    --models-config "$reg"
    --host "$host" --port "$port"
    --max-num-seqs "$(mget "${sel[0]}" max_num_seqs)"
    --cache-memory-mb "$(mget "${sel[0]}" cache_memory_mb)"
    --max-tokens "$max_tok"
    --max-request-tokens "$max_req"
    --gpu-memory-utilization "$(mget "${sel[0]}" gpu_memory_utilization)"
    --continuous-batching
  )
  args+=(${KV_FLAGS[@]+"${KV_FLAGS[@]}"})
  [[ -n "$mrp" ]] && args+=(--reasoning-parser "$mrp")
  [[ -n "$mtp" ]] && args+=(--tool-call-parser "$mtp" --enable-auto-tool-choice)
  (( all_offline == 1 )) && args+=(--offline)

  # Only the first model is warmed: the rest load lazily on their first request,
  # and warming them all would defeat the memory budget by making them all resident.
  start_reporter "$host" "$port" "${sel[0]}" 1
  note "starting ${#sel[@]} models on http://$host:$port — ${sel[*]}"
  note "  '${sel[0]}' loads now; the rest load on first use, budget $(multiget memory_budget_gb) GB, policy $(jq -r '.multi.contention_policy.strategy' "$CONFIG")"
  launch "${args[@]}" "$@"
}

# --- port arbitration --------------------------------------------------------

# Asked for one model on a port that is already serving something else. The running
# process cannot hot-add a model (the registry is built once at startup, cli.py:362,
# and there is no runtime load endpoint), so restart it as a registry holding both.
# From then on the memory budget decides per request whether both stay resident or
# one is evicted — "load it too if it fits, otherwise swap" — with no more restarts.
merge_or_start() {
  local a="$1"; shift
  local host port pids current keep=() id
  host="${HOST:-$(mget "$a" host)}"
  port="${PORT_OVERRIDE:-${PORT:-$(mget "$a" port)}}"

  pids="$(port_pids "$port")"
  if [[ -z "$pids" ]]; then
    serve_single "$a" "$@"
    return
  fi

  current="$(served_ids "$port" "$host")"
  if [[ -z "$current" ]]; then
    # shellcheck disable=SC2086
    ps -o pid=,command= -p $pids >&2 || true
    die "port $port is held by something that is not a vllm-mlx server (PIDs: $(echo "$pids" | tr '\n' ' '))"
  fi

  if printf '%s\n' "$current" | grep -qxF "$a"; then
    note "$a is already being served on http://$host:$port — nothing to do"
    note "  eval \"\$($0 --env $a)\""
    exit 0
  fi

  # Same weights, legacy name: a server started before this script existed, or by
  # `vllm-mlx serve` directly. Reloading 30 GB to rename it is not something to do
  # behind someone's back, so say what is there and let them decide.
  local src; src="$(mget "$a" source)"
  if printf '%s\n' "$current" | grep -qxF "$src"; then
    note "$a is already loaded on http://$host:$port, but under its full id:"
    note "    $src"
    note "  clients must use that name. To expose it as '$a' instead (reloads the weights):"
    note "      ./run.sh --stop $a && ./run.sh $a"
    exit 0
  fi

  if [[ "$NO_MERGE" == "1" ]]; then
    # shellcheck disable=SC2086
    ps -o pid=,command= -p $pids >&2 || true
    die "port $port already serves: $(echo "$current" | tr '\n' ' ')
     stop it first:  ./run.sh --stop $a --port $port"
  fi

  local mapped
  while IFS= read -r id; do
    [[ -n "$id" ]] || continue
    mapped="$(alias_for_id "$id")"
    if [[ -n "$mapped" ]]; then
      keep+=("$mapped")
    else
      warn "dropping '$id' from the merge — nothing in $(basename "$CONFIG") matches it"
    fi
  done <<< "$current"

  keep+=("$a")
  warn "port $port already serves: $(echo "$current" | tr '\n' ' ')"
  warn "restarting it as a multi-model server for: ${keep[*]}"
  warn "  this reloads the weights that are up now (minutes for a 30B) — ^C within 5 s to abort"
  sleep 5
  stop_port "$port" || true
  serve_multi "$port" "${keep[@]}" -- "$@"
}

# --- argument parsing --------------------------------------------------------

MODE=serve
PORT_OVERRIDE=""
NO_MERGE=0
WANT_ALL=0
SEL=()
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)   MODE=help; shift ;;
    --list)      MODE=list; shift ;;
    --env)       MODE="env"; shift ;;   # quoted: bare `env` reads as the command
    --stop)      MODE=stop; shift ;;
    --multi)     MODE=multi; shift ;;
    --all)       WANT_ALL=1; shift ;;
    --no-merge)  NO_MERGE=1; shift ;;
    --port)      [[ -n "${2:-}" ]] || die "--port needs a value"; PORT_OVERRIDE="$2"; shift 2 ;;
    --)          shift; EXTRA+=("$@"); break ;;
    # First flag run.sh doesn't know ends its own parsing; the rest is the
    # server's, so `--api-key foo` keeps its value.
    -*)          EXTRA+=("$@"); break ;;
    *)           SEL+=("$1"); shift ;;
  esac
done

case "$MODE" in
  help) print_help; exit 0 ;;
  list) print_list; exit 0 ;;
esac

if [[ "$MODE" == "stop" ]]; then
  if (( WANT_ALL == 1 )); then do_stop --all; else
    [[ ${#SEL[@]} -eq 1 ]] || die "--stop takes one model name, or --all"
    do_stop "${SEL[0]}"
  fi
  exit 0
fi

if (( WANT_ALL == 1 )) && [[ "$MODE" == "multi" ]]; then
  # --all means "every model already on disk". Say which ones that left out:
  # silently serving a subset of what was asked for reads as full coverage.
  SKIPPED=()
  while IFS= read -r a; do
    case "$(probe_one "$(mget "$a" source)")" in
      ok|weights) SEL+=("$a") ;;
      *)          SKIPPED+=("$a") ;;
    esac
  done < <(aliases)
  [[ ${#SKIPPED[@]} -eq 0 ]] ||
    warn "not downloaded, so not served: ${SKIPPED[*]} — name one explicitly to fetch it"
  [[ ${#SEL[@]} -gt 0 ]] || die "no downloaded models to serve"
fi

if [[ ${#SEL[@]} -eq 0 ]]; then
  if [[ "$MODE" == "serve" ]]; then
    # Bare `./run.sh`: serve the catalog default rather than just printing help —
    # "default": true in models.json should actually mean something.
    SEL=("$(default_alias)")
    note "no model given — serving the default: ${SEL[0]}"
  else
    print_help
    exit 0
  fi
fi
for a in "${SEL[@]}"; do has_alias "$a" || die "unknown model: $a (try ./run.sh --help)"; done

case "$MODE" in
  env)
    [[ ${#SEL[@]} -eq 1 ]] || die "--env takes one model name"
    print_env "${SEL[0]}"
    ;;
  multi)
    [[ ${#SEL[@]} -ge 1 ]] || die "--multi needs at least one model name"
    MP="${PORT_OVERRIDE:-${PORT:-$(multiget port)}}"
    if [[ -n "$(port_pids "$MP")" ]]; then
      die "port $MP is already in use — ./run.sh --stop --all first"
    fi
    serve_multi "$MP" "${SEL[@]}" -- ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  serve)
    [[ ${#SEL[@]} -eq 1 ]] || die "serve takes one model name — did you mean --multi ${SEL[*]}?"
    merge_or_start "${SEL[0]}" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
esac
