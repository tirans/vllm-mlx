# SPDX-License-Identifier: Apache-2.0
"""Pure contract tests for the fork's reasoning-effort resolution policy."""

from dataclasses import FrozenInstanceError

import pytest

from vllm_mlx.effort_policy import EffortResolution, resolve_reasoning_effort


def resolve(**overrides):
    arguments = {
        "resolved_template_kwargs": None,
        "request_effort": None,
        "request_template_kwargs": None,
        "supported_efforts": ("low", "medium", "xhigh"),
        "model_name": "guarded",
    }
    arguments.update(overrides)
    return resolve_reasoning_effort(**arguments)


@pytest.mark.parametrize("kwargs", [None, {}, {"enable_thinking": False}])
def test_no_effort_preserves_resolved_shape_and_identity(kwargs):
    result = resolve(resolved_template_kwargs=kwargs)

    assert result.template_kwargs is kwargs
    assert not result.replace_kwargs
    assert not result.dropped_default
    assert result.error_detail is None


def test_top_level_wins_and_does_not_mutate_inputs():
    resolved = {"reasoning_effort": "medium", "enable_thinking": False}
    request_kwargs = {"reasoning_effort": "medium", "request_only": 1}

    result = resolve(
        resolved_template_kwargs=resolved,
        request_effort="xhigh",
        request_template_kwargs=request_kwargs,
    )

    assert result.template_kwargs == {
        "reasoning_effort": "xhigh",
        "enable_thinking": False,
    }
    assert result.template_kwargs is not resolved
    assert result.replace_kwargs
    assert not result.dropped_default
    assert result.error_detail is None
    assert resolved == {"reasoning_effort": "medium", "enable_thinking": False}
    assert request_kwargs == {"reasoning_effort": "medium", "request_only": 1}


def test_request_template_value_is_validated_without_replacement():
    resolved = {"reasoning_effort": "medium", "other": 1}
    result = resolve(
        resolved_template_kwargs=resolved,
        request_template_kwargs={"reasoning_effort": "medium"},
    )

    assert result.template_kwargs is resolved
    assert not result.replace_kwargs
    assert result.error_detail is None


def test_unsupported_default_is_removed_without_mutating_input():
    original = {"reasoning_effort": "low", "enable_thinking": True}
    result = resolve(
        resolved_template_kwargs=original,
        supported_efforts=None,
        model_name="plain",
    )

    assert result.template_kwargs == {"enable_thinking": True}
    assert result.replace_kwargs and result.dropped_default
    assert result.error_detail is None
    assert original == {"reasoning_effort": "low", "enable_thinking": True}


@pytest.mark.parametrize("surface", ["top", "template"])
def test_unsupported_request_effort_returns_exact_error(surface):
    resolved = {"reasoning_effort": "low", "other": 1}
    arguments = (
        {"request_effort": "low"}
        if surface == "top"
        else {"request_template_kwargs": {"reasoning_effort": "low"}}
    )
    result = resolve(
        resolved_template_kwargs=resolved,
        supported_efforts=None,
        model_name="plain",
        **arguments,
    )

    assert result.error_detail == (
        "model 'plain' does not support reasoning_effort "
        "(its chat template ignores the parameter); remove it or use a "
        "model whose /v1/models entry lists reasoning_efforts"
    )
    assert result.template_kwargs == resolved
    assert result.replace_kwargs is (surface == "top")
    assert not result.dropped_default


@pytest.mark.parametrize("surface", ["top", "default"])
def test_invalid_value_returns_vocabulary_and_preserves_override(surface):
    original = {"reasoning_effort": "medium", "other": 1}
    arguments = {"request_effort": "high"} if surface == "top" else {}
    if surface == "default":
        original["reasoning_effort"] = "high"
    result = resolve(resolved_template_kwargs=original, **arguments)

    assert result.error_detail == (
        "reasoning_effort 'high' is not supported by model 'guarded'; "
        "supported values: low, medium, xhigh"
    )
    assert result.template_kwargs == {"reasoning_effort": "high", "other": 1}
    assert result.replace_kwargs is (surface == "top")
    assert not result.dropped_default
    if surface == "top":
        assert original["reasoning_effort"] == "medium"


def test_unknown_vocabulary_allows_effort_without_dropping_kwargs():
    resolved = {"reasoning_effort": "custom", "other": False}
    result = resolve(resolved_template_kwargs=resolved, supported_efforts=())

    assert result.template_kwargs is resolved
    assert not result.replace_kwargs
    assert not result.dropped_default
    assert result.error_detail is None


def test_falsey_request_template_value_is_explicit_on_unsupported_model():
    resolved = {"reasoning_effort": "", "other": 1}
    request_kwargs = {"reasoning_effort": ""}
    result = resolve(
        resolved_template_kwargs=resolved,
        request_template_kwargs=request_kwargs,
        supported_efforts=None,
        model_name="plain",
    )

    assert result.template_kwargs is resolved
    assert not result.replace_kwargs and not result.dropped_default
    assert result.error_detail == (
        "model 'plain' does not support reasoning_effort "
        "(its chat template ignores the parameter); remove it or use a "
        "model whose /v1/models entry lists reasoning_efforts"
    )
    assert resolved["reasoning_effort"] == ""
    assert request_kwargs["reasoning_effort"] == ""


def test_falsey_top_level_value_still_replaces_and_is_validated():
    result = resolve(
        resolved_template_kwargs={"reasoning_effort": "medium"},
        request_effort="",
    )

    assert result.template_kwargs == {"reasoning_effort": ""}
    assert result.replace_kwargs
    assert result.error_detail == (
        "reasoning_effort '' is not supported by model 'guarded'; "
        "supported values: low, medium, xhigh"
    )


def test_resolution_record_is_frozen():
    result = EffortResolution(None)
    with pytest.raises(FrozenInstanceError):
        result.replace_kwargs = True
