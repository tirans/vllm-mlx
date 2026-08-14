#!/usr/bin/env bash
# One-command build for the vllm-mlx server: sync the venv, fetch the model
# weights, verify the snapshot, and optionally hand straight over to the server.
#
#   ./scripts/setup.sh                  sync deps + download the default model
#   ./scripts/setup.sh --serve          ... then exec serve-qwen3-thinking.sh
#   ./scripts/setup.sh --serve --repro  ... serving with --disable-prefix-cache
#   MODEL=<repo/id> ./scripts/setup.sh  a different checkpoint
#
# Why a separate script from serve-qwen3-thinking.sh: that one refuses to start
# on an incomplete snapshot and tells you to run `hf download`. This is that
# step, made explicit, so a fresh machine is two commands instead of a puzzle.
#
# --repro exists because the prefix cache is the measured cause of the model's
# non-reproducibility: for one byte-identical prompt at temperature 0 the server
# returns different text depending on the state of its prefix cache, and with
# caching off the same prompt is byte-identical across process restarts. It costs
# a Claude Code session its ~35x cached-prefill win, and costs a workload of
# all-distinct prompts (e.g. a Prism map) roughly nothing, because such a
# workload never gets the repeat hit the cache exists for.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
MODEL="${MODEL:-lmstudio-community/Qwen3-30B-A3B-Thinking-2507-MLX-8bit}"

SERVE=0
REPRO=0
for arg in "$@"; do
  case "$arg" in
    --serve) SERVE=1 ;;
    --repro) REPRO=1 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
note() { printf '\033[36m%s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }

# --- 1. dependencies ---------------------------------------------------------

command -v uv >/dev/null 2>&1 || die "uv is not installed — https://docs.astral.sh/uv/"

note "syncing the venv ..."
(cd "$REPO" && uv sync)
[[ -x "$PY" ]] || die "uv sync finished but there is no interpreter at $PY"

# Every .pth in this venv carries the macOS UF_HIDDEN flag on some checkouts, and
# CPython 3.13's site.addpackage skips hidden .pth files, so the editable install
# never registers and the `vllm-mlx` console script is inert. Clearing the flag is
# the permanent fix; PYTHONPATH is the workaround the serve script already uses.
SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -n "$SITE_PACKAGES" ]] && ls -lO "$SITE_PACKAGES"/*.pth 2>/dev/null | grep -q hidden; then
  note "clearing UF_HIDDEN on $SITE_PACKAGES/*.pth ..."
  chflags nohidden "$SITE_PACKAGES"/*.pth || warn "chflags failed; PYTHONPATH will still work"
fi

PYTHONPATH="$REPO" "$PY" -c 'import vllm_mlx' 2>/dev/null \
  || die "vllm_mlx is still not importable after uv sync"

# --- 2. weights --------------------------------------------------------------

snapshot_complete() {
  # The only honest test is the call the server itself makes: having every weight
  # shard is NOT enough, because huggingface_hub raises IncompleteSnapshotError if
  # any file from the revision is missing.
  "$PY" - "$MODEL" <<'PROBE' 2>/dev/null
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_files_only=True)
PROBE
}

if snapshot_complete; then
  note "model already complete in the local cache: $MODEL"
else
  note "downloading $MODEL (tens of GB on a first run) ..."
  "$REPO/.venv/bin/hf" download "$MODEL" || die "download failed"
  snapshot_complete || die "snapshot still incomplete after download — rerun, or clear ~/.cache/huggingface"
  note "download complete"
fi

# --- 3. run ------------------------------------------------------------------

if [[ $SERVE -eq 0 ]]; then
  note "done. start the server with:"
  note "    ./scripts/serve-qwen3-thinking.sh                        (fast, prefix cache on)"
  note "    ./scripts/serve-qwen3-thinking.sh --disable-prefix-cache (reproducible)"
  exit 0
fi

FLAGS=()
[[ $REPRO -eq 1 ]] && FLAGS+=(--disable-prefix-cache)
exec "$REPO/scripts/serve-qwen3-thinking.sh" ${FLAGS[@]+"${FLAGS[@]}"}
