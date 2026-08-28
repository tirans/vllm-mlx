---
name: smoke-test-registry
description: Fast end-to-end smoke test of run.sh / model-registry changes using small cached models (serve → swap → evict → stop in seconds instead of waiting on the 27B default). Use after editing run.sh, model_registry.py, models.json, or server model-routing code.
---

# Smoke-test run.sh / registry changes

## Pick a small model

```bash
./run.sh --list        # status column: ok = cached, missing = would download
```

Use `tiny-0.6b` or `tiny-1b` — a full serve→swap→evict cycle takes seconds.

## Cycle

```bash
# 1. Serve one tiny model in elastic single mode (registry-backed, max_resident_models: 1)
./run.sh tiny-0.6b

# 2. Wait for the `warm` log line before testing swaps — the startup warmup's second
#    priming call can otherwise swap the resident model back after your request.

# 3. Exercise the elastic swap: request a DIFFERENT cached catalog alias by name on
#    /v1/chat/completions — it should unload tiny-0.6b, load the requested one, and
#    answer (not 404). Requests must use catalog ALIASES, not HF repo ids
#    (--served-model-name; server.py rejects other names with 404).

# 4. Verify /v1/models lists EVERY catalog entry with a per-entry `loaded` bool,
#    and /v1/status shows live active_requests/generation_tps for the loaded model.

# 5. Tear down
./run.sh --stop --all   # kills every listener on configured ports, no confirmation
```

## Cautions

- `./run.sh --stop --all` does not ask — check nothing important is running first,
  and avoid restarting while a consumer (e.g. Prism on port 8000) is actively
  calling: the port refuses connections for several seconds during preload.
- `--pin <name>` is the separate hard-pinned path (no registry, 404 on other names,
  only path where `--auto-unload-idle-seconds` applies) — test it separately if the
  change touches it.
- Parsers (`--reasoning-parser`/`--tool-call-parser`) are process-global, fixed at
  startup from the launched alias — a swapped-in model keeps the original parsers.
  Don't misread that as a routing bug.
- `timeout` is unavailable in this shell — use `nohup <cmd> > log 2>&1 &` plus
  `sleep`/polling.
