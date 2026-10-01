# SPDX-License-Identifier: Apache-2.0
"""Pure reasoning-effort precedence and validation for chat templates."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EffortResolution:
    template_kwargs: Mapping[str, object] | None
    replace_kwargs: bool = False
    dropped_default: bool = False
    error_detail: str | None = None


def resolve_reasoning_effort(
    *,
    resolved_template_kwargs: Mapping[str, object] | None,
    request_effort: str | None,
    request_template_kwargs: Mapping[str, object] | None,
    supported_efforts: tuple[str, ...] | None,
    model_name: str,
) -> EffortResolution:
    """Resolve the existing request/default policy without changing its inputs."""
    ctk = resolved_template_kwargs
    explicit = request_effort or (request_template_kwargs or {}).get("reasoning_effort")
    effort = explicit or (ctk or {}).get("reasoning_effort")
    if effort is None:
        return EffortResolution(ctk)

    replace = request_effort is not None
    if replace:
        ctk = dict(ctk or {})
        ctk["reasoning_effort"] = request_effort
        effort = request_effort

    if supported_efforts is None:
        if explicit is None:
            filtered = {
                key: value
                for key, value in (ctk or {}).items()
                if key != "reasoning_effort"
            }
            return EffortResolution(filtered, True, True)
        detail = (
            f"model {model_name!r} does not support reasoning_effort "
            "(its chat template ignores the parameter); remove it or use a "
            "model whose /v1/models entry lists reasoning_efforts"
        )
        return EffortResolution(ctk, replace, error_detail=detail)

    if supported_efforts and effort not in supported_efforts:
        detail = (
            f"reasoning_effort {effort!r} is not supported by model "
            f"{model_name!r}; supported values: {', '.join(supported_efforts)}"
        )
        return EffortResolution(ctk, replace, error_detail=detail)

    return EffortResolution(ctk, replace)
