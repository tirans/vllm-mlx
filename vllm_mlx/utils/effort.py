# SPDX-License-Identifier: Apache-2.0
"""Per-model reasoning-effort introspection.

There is no cross-model vocabulary for reasoning effort: harmony/GPT-OSS models
accept ``low/medium/high`` via the openai-harmony renderer, Qwen3.8's chat
template accepts ``low/medium/xhigh`` (and raises ``raise_exception`` on
anything else), and most templates ignore the variable entirely. This module
answers, per model, "which values does this template actually accept?" so the
server can validate a request's ``reasoning_effort`` instead of silently
ignoring it (qwen3-thinking) or letting a Jinja exception surface as a 500
(qwen3.8 with an out-of-vocabulary value).

Detection is by trial render, not by parsing the Jinja: render the template
once per candidate value and keep the values that render. That exercises the
exact ``raise_exception`` guard a real request would hit, so the result is
correct by construction for any template. A template that never references
``reasoning_effort`` is classified unsupported without rendering at all.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: What the openai-harmony renderer accepts (`ReasoningEffort.{LOW,MEDIUM,HIGH}`);
#: harmony models bypass the Jinja template entirely, so this set is static.
HARMONY_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: The union of effort vocabularies seen in the wild: OpenAI's request enum
#: (none/minimal/low/medium/high/xhigh), Anthropic's (low..xhigh/max), and the
#: template-specific sets are all subsets of this. Trial-rendering each one is
#: cheap (pure-CPU Jinja), so over-inclusion costs nothing and future-proofs
#: against a template inventing a new level we already probe for.
CANDIDATE_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

_PROBE_MESSAGES = [{"role": "user", "content": "probe"}]


def _build_jinja_env():
    """A sandboxed Jinja environment matching what HF tokenizers use.

    ``raise_exception`` and ``strftime_now`` are the two globals HF injects
    that real chat templates call; ``tojson`` ships with jinja2 but HF binds
    its own json.dumps-based version, matched here for fidelity.
    """
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, **kw)
    return env


def efforts_for_template(template: str | None) -> tuple[str, ...] | None:
    """Which ``reasoning_effort`` values this chat template accepts.

    Returns ``None`` when the template does not support the variable at all
    (no template, or it never references ``reasoning_effort``). Returns a
    possibly-empty tuple when it does: the values that rendered. An empty
    tuple means "referenced but undeterminable" (every probe render failed,
    e.g. a template needing inputs the probe does not supply) — callers must
    treat that as unknown and skip validation rather than reject everything.
    """
    if not template or "reasoning_effort" not in template:
        return None
    try:
        compiled = _build_jinja_env().from_string(template)
    except Exception:
        logger.debug("chat template failed to compile during effort probing")
        return ()
    supported = []
    for candidate in CANDIDATE_EFFORTS:
        try:
            compiled.render(
                messages=_PROBE_MESSAGES,
                add_generation_prompt=True,
                reasoning_effort=candidate,
            )
        except Exception:
            continue
        supported.append(candidate)
    return tuple(supported)


def template_from_tokenizer(tokenizer) -> str | None:
    """The chat template string a loaded tokenizer would render with."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template:
        return template
    # mlx-lm's TokenizerWrapper proxies attributes; fall back to the wrapped
    # HF tokenizer when the proxy does not expose chat_template directly.
    inner = getattr(tokenizer, "_tokenizer", None)
    template = getattr(inner, "chat_template", None)
    return template if isinstance(template, str) and template else None


def template_from_source(source: str | None) -> str | None:
    """Read a model's chat template off disk without loading the model.

    ``source`` is an HF repo id or a local path. Looks for
    ``chat_template.jinja`` first (newer exports), then the ``chat_template``
    key of ``tokenizer_config.json``. Returns None when the model is not
    cached locally — callers report "unknown" rather than guessing.
    """
    if not source:
        return None
    root = Path(source).expanduser()
    if not root.is_dir():
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache / f"models--{source.replace('/', '--')}"
        snapshots = (
            sorted((repo_dir / "snapshots").glob("*")) if repo_dir.is_dir() else []
        )
        if not snapshots:
            return None
        root = snapshots[-1]
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        try:
            return jinja.read_text(encoding="utf-8")
        except OSError:
            return None
    config = root / "tokenizer_config.json"
    if config.is_file():
        try:
            template = json.loads(config.read_text(encoding="utf-8")).get(
                "chat_template"
            )
        except (OSError, ValueError):
            return None
        if isinstance(template, str) and template:
            return template
    return None


def supported_efforts(
    *,
    tokenizer=None,
    source: str | None = None,
    use_harmony: bool = False,
) -> tuple[str, ...] | None:
    """Resolve a model's supported reasoning-effort values.

    Harmony models are static (the renderer, not the template, consumes the
    value). Otherwise the loaded tokenizer's template wins; the on-disk
    template is the fallback for models that are not loaded.
    """
    if use_harmony:
        return HARMONY_EFFORTS
    template = template_from_tokenizer(tokenizer) if tokenizer is not None else None
    if template is None:
        template = template_from_source(source)
    return efforts_for_template(template)
