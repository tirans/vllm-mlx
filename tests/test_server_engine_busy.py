# SPDX-License-Identifier: Apache-2.0
"""HTTP-level behaviour when the serialized engine refuses a request.

An admission failure must always reach the client as a retryable 503 (or, for
streams whose status line is already committed, an in-band error frame) --
never as a raw 500 or a silently truncated response.
"""

import json
import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Server tests require Apple Silicon",
)

BUSY_MESSAGE = "SimpleEngine generation queue is full; retry after 2s"


def _busy_error():
    from vllm_mlx.engine.base import EngineBusy

    return EngineBusy(BUSY_MESSAGE, reason="queue_full", retry_after=2)


class BusyEngine:
    """Stands in for a SimpleEngine whose serialized route is saturated."""

    model_name = "test-model"
    is_mllm = False
    preserve_native_tool_format = False

    async def chat(self, messages, **kwargs):
        raise _busy_error()

    async def generate(self, *args, **kwargs):
        raise _busy_error()

    async def stream_chat(self, *args, **kwargs):
        raise _busy_error()
        yield  # pragma: no cover - makes this an async generator

    async def stream_generate(self, *args, **kwargs):
        raise _busy_error()
        yield  # pragma: no cover - makes this an async generator


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import vllm_mlx.server as server

    monkeypatch.setattr(server, "_engine", BusyEngine())
    monkeypatch.setattr(server, "_model_name", "test-model")
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    monkeypatch.setattr(server, "_default_max_tokens", 128)
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(
        server,
        "_rate_limiter",
        server.RateLimiter(requests_per_minute=60, enabled=False),
    )
    return TestClient(server.app)


def _sse_events(text: str) -> list[dict]:
    """Parse the ``data:`` payloads out of an SSE body."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :].strip()
            if payload and payload != "[DONE]":
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return events


class TestAnthropicMessagesBusy:
    """The reported bug: /v1/messages 500'd instead of returning 503."""

    def test_returns_503_not_500(self, client):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "2"

        # The Anthropic SDK only recognises the bare envelope, so it must not
        # be nested under FastAPI's "detail" key.
        envelope = response.json()
        assert "detail" not in envelope
        assert envelope["type"] == "error"
        assert envelope["error"]["type"] == "overloaded_error"
        assert BUSY_MESSAGE in envelope["error"]["message"]

    def test_stream_emits_error_event_and_one_terminal(self, client):
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200
        body = response.text

        assert "event: error" in body, "busy stream must report the cause in-band"
        assert body.count("event: message_stop") == 1

        errors = [e for e in _sse_events(body) if e.get("type") == "error"]
        assert errors, "expected an error payload in the stream"
        assert errors[0]["error"]["type"] == "overloaded_error"


class TestOpenAIEndpointsBusy:
    def test_chat_completions_returns_503(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "2"
        envelope = response.json()
        assert "detail" not in envelope
        assert envelope["error"]["code"] == "text_generation_busy"

    def test_chat_completions_stream_emits_error_frame(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200
        body = response.text
        assert body.count("data: [DONE]") == 1

        errors = [e for e in _sse_events(body) if "error" in e]
        assert errors, "expected an error frame in the stream"
        assert errors[0]["error"]["code"] == "text_generation_busy"

    def test_completions_returns_503(self, client):
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi"},
        )

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "2"


class TestResponsesEndpointBusy:
    def test_returns_503(self, client):
        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "hi"},
        )

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "2"

    def test_stream_emits_error_event(self, client):
        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "hi", "stream": True},
        )

        assert response.status_code == 200
        body = response.text
        assert "event: error" in body
        # The Responses stream has no [DONE] terminal; don't invent one.
        assert "data: [DONE]" not in body

        errors = [e for e in _sse_events(body) if e.get("type") == "error"]
        assert errors
        assert errors[0]["code"] == "text_generation_busy"


class TestEngineBusyBackstop:
    """Nothing may reach the client as a 500, even on an unpatched path."""

    def test_handler_renders_both_dialects(self):
        import asyncio
        from types import SimpleNamespace

        import vllm_mlx.server as server

        def render(path):
            request = SimpleNamespace(url=SimpleNamespace(path=path))
            response = asyncio.run(
                server._engine_busy_exception_handler(request, _busy_error())
            )
            return response, json.loads(bytes(response.body))

        anthropic_response, anthropic_body = render("/v1/messages")
        assert anthropic_response.status_code == 503
        assert anthropic_response.headers["retry-after"] == "2"
        assert anthropic_body["error"]["type"] == "overloaded_error"

        openai_response, openai_body = render("/v1/chat/completions")
        assert openai_response.status_code == 503
        assert openai_body["error"]["code"] == "text_generation_busy"


class TestEnsureSSETerminal:
    """The wrapper must surface causes and survive client disconnects."""

    def test_emits_error_frame_then_single_terminal(self):
        import asyncio

        import vllm_mlx.server as server

        async def boom():
            yield "chunk\n\n"
            raise _busy_error()

        async def collect():
            return [
                chunk
                async for chunk in server._ensure_sse_terminal(
                    boom(),
                    "TERMINAL",
                    server._stream_error_frame(server._openai_error_frame),
                )
            ]

        chunks = asyncio.run(collect())
        assert chunks[0] == "chunk\n\n"
        assert "text_generation_busy" in chunks[1]
        assert chunks.count("TERMINAL") == 1

    def test_engine_fault_also_gets_an_error_frame(self):
        """The 2026-08-25 incident: a wedged engine must not look successful.

        A streaming response commits its 200 before the engine is called, so a
        mid-stream engine fault cannot become a 5xx. With no in-band frame the
        client sees a well-formed, EMPTY, successful stream -- 581 of those were
        cached as artifacts by a downstream consumer in 10.5 seconds. This used
        to return None for anything that was not EngineBusy.
        """
        import asyncio

        import vllm_mlx.server as server

        async def wedged():
            raise RuntimeError(
                "[METAL] Command buffer execution failed: Ignored (for causing "
                "prior/excessive GPU errors)"
            )
            yield  # pragma: no cover - unreachable, makes this an async generator

        async def collect():
            return [
                chunk
                async for chunk in server._ensure_sse_terminal(
                    wedged(),
                    "TERMINAL",
                    server._stream_error_frame(server._openai_error_frame),
                )
            ]

        chunks = asyncio.run(collect())
        # An error frame BEFORE the terminal, so the stream is not silently empty.
        assert len(chunks) == 2, chunks
        assert "METAL" in chunks[0]
        assert "engine_error" in chunks[0]
        assert chunks[1] == "TERMINAL"

    def test_engine_fault_is_distinguishable_from_admission_failure(self):
        """A client must tell "retry shortly" from "this engine is broken"."""
        import json

        import vllm_mlx.server as server

        busy = json.loads(
            server._anthropic_error_frame(_busy_error()).split("data: ", 1)[1]
        )
        fault = json.loads(
            server._anthropic_error_frame(RuntimeError("engine died")).split(
                "data: ", 1
            )[1]
        )
        assert busy["error"]["type"] == "overloaded_error"
        assert fault["error"]["type"] == "api_error"

        busy_oai = json.loads(
            server._openai_error_frame(_busy_error()).split("data: ", 1)[1]
        )
        fault_oai = json.loads(
            server._openai_error_frame(RuntimeError("engine died")).split("data: ", 1)[
                1
            ]
        )
        assert busy_oai["error"]["code"] == "text_generation_busy"
        assert fault_oai["error"]["code"] == "engine_error"

    def test_error_message_is_bounded(self):
        """An MLX driver string must not be streamed in full."""
        import json

        import vllm_mlx.server as server

        frame = server._openai_error_frame(RuntimeError("x" * 5000))
        msg = json.loads(frame.split("data: ", 1)[1])["error"]["message"]
        assert len(msg) <= server._STREAM_ERROR_MESSAGE_LIMIT
        assert msg.endswith("...")

    def test_aclose_does_not_raise(self):
        """Yielding from the finally block used to break on client disconnect."""
        import asyncio

        import vllm_mlx.server as server

        async def endless():
            while True:
                yield "chunk\n\n"

        async def drive():
            gen = server._ensure_sse_terminal(endless(), "TERMINAL")
            assert await gen.__anext__() == "chunk\n\n"
            await gen.aclose()  # RuntimeError before the fix

        asyncio.run(drive())
