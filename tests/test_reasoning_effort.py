# SPDX-License-Identifier: Apache-2.0
"""Reasoning-effort introspection and request validation.

The synthetic templates mirror the two real-world shapes: Qwen3.8's guarded
vocabulary (renders low/medium/xhigh, raise_exception on anything else) and
the qwen3-thinking family's templates that never reference reasoning_effort.
"""

import pytest
from fastapi import HTTPException

from vllm_mlx.api.models import ChatCompletionRequest, Message
from vllm_mlx.utils.effort import (
    CANDIDATE_EFFORTS,
    HARMONY_EFFORTS,
    efforts_for_template,
    supported_efforts,
)

# The guard pattern lifted from mlx-community/Qwen3.8-27B-8bit's template.
GUARDED_TEMPLATE = """
{%- set resolved = reasoning_effort|default('xhigh') %}
{%- if resolved not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort) }}
{%- endif %}
{%- for message in messages %}
{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
"""

PLAIN_TEMPLATE = """
{%- for message in messages %}
{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
"""


class TestEffortsForTemplate:
    def test_guarded_template_yields_exact_vocabulary(self):
        assert efforts_for_template(GUARDED_TEMPLATE) == ("low", "medium", "xhigh")

    def test_template_without_reference_is_unsupported(self):
        assert efforts_for_template(PLAIN_TEMPLATE) is None

    def test_missing_template_is_unsupported(self):
        assert efforts_for_template(None) is None
        assert efforts_for_template("") is None

    def test_unguarded_reference_accepts_every_candidate(self):
        template = "{{ reasoning_effort|default('medium') }}: {{ messages[0].content }}"
        assert efforts_for_template(template) == CANDIDATE_EFFORTS

    def test_broken_template_reports_unknown_not_unsupported(self):
        # References the variable but cannot compile: callers must skip
        # validation (empty tuple), never reject every value (None).
        assert efforts_for_template("{% if reasoning_effort %}{% endif") == ()


class TestSupportedEfforts:
    def test_harmony_is_static(self):
        assert supported_efforts(use_harmony=True) == HARMONY_EFFORTS

    def test_tokenizer_template_wins(self):
        class FakeTokenizer:
            chat_template = GUARDED_TEMPLATE

        assert supported_efforts(tokenizer=FakeTokenizer()) == (
            "low",
            "medium",
            "xhigh",
        )

    def test_wrapped_tokenizer_falls_through(self):
        class Inner:
            chat_template = GUARDED_TEMPLATE

        class Wrapper:
            _tokenizer = Inner()

        assert supported_efforts(tokenizer=Wrapper()) == ("low", "medium", "xhigh")

    def test_no_template_anywhere_is_unsupported(self):
        class Bare:
            pass

        assert supported_efforts(tokenizer=Bare(), source=None) is None


class _GuardedEngine:
    """Engine double exposing a tokenizer with the guarded template."""

    use_harmony_rendering = False

    class tokenizer:
        chat_template = GUARDED_TEMPLATE


class _PlainEngine:
    use_harmony_rendering = False

    class tokenizer:
        chat_template = PLAIN_TEMPLATE


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=kwargs.pop("model", "test-model"),
        messages=[Message(role="user", content="hi")],
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _fresh_cache():
    from vllm_mlx import server

    server._effort_support_cache.clear()
    yield
    server._effort_support_cache.clear()


class TestApplyReasoningEffort:
    def test_absent_effort_keeps_kwargs_shape_without_probing(self, monkeypatch):
        from vllm_mlx import server

        def fail_probe(*args):
            raise AssertionError("no-effort request probed model template")

        monkeypatch.setattr(server, "_supported_reasoning_efforts", fail_probe)
        chat_kwargs = {"temperature": 0.2}

        server._apply_reasoning_effort(chat_kwargs, _PlainEngine(), _request())

        assert chat_kwargs == {"temperature": 0.2}

    def test_top_level_effort_merges_into_chat_template_kwargs(self):
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs: dict = {}
        request = _request(reasoning_effort="low")
        _apply_reasoning_effort(chat_kwargs, _GuardedEngine(), request)
        assert chat_kwargs["chat_template_kwargs"]["reasoning_effort"] == "low"

    def test_top_level_wins_over_chat_template_kwargs(self):
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs: dict = {"chat_template_kwargs": {"reasoning_effort": "medium"}}
        request = _request(
            reasoning_effort="xhigh",
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        _apply_reasoning_effort(chat_kwargs, _GuardedEngine(), request)
        assert chat_kwargs["chat_template_kwargs"]["reasoning_effort"] == "xhigh"

    def test_out_of_vocabulary_value_is_400_with_vocabulary(self):
        from vllm_mlx.server import _apply_reasoning_effort

        request = _request(reasoning_effort="high")
        with pytest.raises(HTTPException) as exc:
            _apply_reasoning_effort({}, _GuardedEngine(), request)
        assert exc.value.status_code == 400
        assert "low, medium, xhigh" in exc.value.detail

    def test_top_level_override_is_written_before_vocabulary_error(self):
        from vllm_mlx.server import _apply_reasoning_effort

        original = {"reasoning_effort": "medium", "enable_thinking": False}
        chat_kwargs = {"chat_template_kwargs": original}
        with pytest.raises(HTTPException) as exc:
            _apply_reasoning_effort(
                chat_kwargs, _GuardedEngine(), _request(reasoning_effort="high")
            )

        assert exc.value.status_code == 400
        assert chat_kwargs["chat_template_kwargs"] == {
            "reasoning_effort": "high",
            "enable_thinking": False,
        }
        assert chat_kwargs["chat_template_kwargs"] is not original
        assert original["reasoning_effort"] == "medium"

    def test_invalid_server_default_is_rejected_on_guarded_template(self):
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs = {"chat_template_kwargs": {"reasoning_effort": "high"}}
        with pytest.raises(HTTPException) as exc:
            _apply_reasoning_effort(chat_kwargs, _GuardedEngine(), _request())

        assert exc.value.status_code == 400
        assert "supported values: low, medium, xhigh" in exc.value.detail

    def test_unknown_vocabulary_passes_effort_through(self, monkeypatch):
        from vllm_mlx import server

        monkeypatch.setattr(server, "_supported_reasoning_efforts", lambda *a: ())
        chat_kwargs = {
            "chat_template_kwargs": {"reasoning_effort": "custom", "other": 1}
        }

        server._apply_reasoning_effort(chat_kwargs, _GuardedEngine(), _request())

        assert chat_kwargs["chat_template_kwargs"] == {
            "reasoning_effort": "custom",
            "other": 1,
        }

    def test_effort_on_unsupporting_model_is_400(self):
        from vllm_mlx.server import _apply_reasoning_effort

        request = _request(reasoning_effort="low")
        with pytest.raises(HTTPException) as exc:
            _apply_reasoning_effort({}, _PlainEngine(), request)
        assert exc.value.status_code == 400
        assert "does not support reasoning_effort" in exc.value.detail

    def test_chat_template_kwargs_effort_is_validated_too(self):
        # The historical pass-through location must no longer silently no-op.
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs: dict = {"chat_template_kwargs": {"reasoning_effort": "low"}}
        request = _request(chat_template_kwargs={"reasoning_effort": "low"})
        with pytest.raises(HTTPException) as exc:
            _apply_reasoning_effort(chat_kwargs, _PlainEngine(), request)
        assert exc.value.status_code == 400

    def test_server_default_effort_is_dropped_not_rejected(self):
        # An effort arriving only via --default-chat-template-kwargs must not
        # fail requests against a model that ignores it.
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs: dict = {"chat_template_kwargs": {"reasoning_effort": "low"}}
        request = _request()  # request itself carries no effort
        _apply_reasoning_effort(chat_kwargs, _PlainEngine(), request)
        assert "reasoning_effort" not in chat_kwargs["chat_template_kwargs"]

    def test_falsey_request_template_effort_is_rejected_as_explicit(self):
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs = {
            "chat_template_kwargs": {
                "reasoning_effort": "",
                "enable_thinking": False,
            }
        }
        request = _request(chat_template_kwargs={"reasoning_effort": ""})

        with pytest.raises(HTTPException) as exc:
            _apply_reasoning_effort(chat_kwargs, _PlainEngine(), request)

        assert exc.value.status_code == 400
        assert "does not support reasoning_effort" in exc.value.detail
        assert chat_kwargs["chat_template_kwargs"] == {
            "reasoning_effort": "",
            "enable_thinking": False,
        }

    def test_no_effort_anywhere_is_untouched(self):
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs: dict = {}
        _apply_reasoning_effort(chat_kwargs, _PlainEngine(), _request())
        assert chat_kwargs == {}

    def test_valid_value_passes(self):
        from vllm_mlx.server import _apply_reasoning_effort

        chat_kwargs: dict = {}
        request = _request(reasoning_effort="xhigh")
        _apply_reasoning_effort(chat_kwargs, _GuardedEngine(), request)
        assert chat_kwargs["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
