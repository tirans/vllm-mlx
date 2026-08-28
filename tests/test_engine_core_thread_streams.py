# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for MLX stream/thread ownership in engine loops."""

import asyncio
import threading
from types import SimpleNamespace

import pytest


class _SchedulerOutput:
    outputs = []
    finished_request_ids = []


@pytest.mark.anyio
async def test_engine_core_runs_all_scheduler_steps_on_one_worker_thread():
    """Continuous batching must not bounce MLX steps across threads."""
    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = object.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=0, stream_interval=1)
    engine._running = True
    engine._steps_executed = 0
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}

    main_thread = threading.get_ident()

    class FakeScheduler:
        batch_generator = SimpleNamespace(_partial=None)

        def __init__(self):
            self.calls = 0
            self.step_threads: list[int] = []
            self.close_threads: list[int] = []

        def has_requests(self):
            return self.calls < 3

        def step(self):
            self.step_threads.append(threading.get_ident())
            self.calls += 1
            if self.calls == 3:
                engine._running = False
            return _SchedulerOutput()

        def _close_batch_generator(self):
            self.close_threads.append(threading.get_ident())

    scheduler = FakeScheduler()
    engine.scheduler = scheduler

    await asyncio.wait_for(engine._engine_loop(), timeout=2)

    assert scheduler.step_threads
    assert len(set(scheduler.step_threads)) == 1
    assert scheduler.step_threads[0] != main_thread
    assert scheduler.close_threads == [scheduler.step_threads[0]]


@pytest.mark.anyio
async def test_mllm_scheduler_runs_steps_on_model_load_thread():
    """MLLM keeps generation on the event-loop thread that loaded the model."""
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    scheduler = object.__new__(MLLMScheduler)
    scheduler._running = True

    main_thread = threading.get_ident()
    step_threads: list[int] = []
    close_threads: list[int] = []

    class FakeBatchGenerator:
        _partial = None

        def close(self):
            close_threads.append(threading.get_ident())

    scheduler.batch_generator = FakeBatchGenerator()

    def has_requests():
        return len(step_threads) < 3

    def step():
        step_threads.append(threading.get_ident())
        if len(step_threads) == 3:
            scheduler._running = False

    scheduler.has_requests = has_requests
    scheduler.step = step

    await asyncio.wait_for(scheduler._process_loop(), timeout=2)

    assert step_threads
    assert len(set(step_threads)) == 1
    assert step_threads[0] == main_thread
    assert close_threads == []


def test_mlx_lm_generation_stream_is_thread_local_and_stays_that_way():
    """No vllm_mlx module may publish a thread-affine stream into a shared global.

    This is the invariant that replaced ``bind_generation_streams``. mlx-lm's
    module-level ``generation_stream`` is an ``mx.ThreadLocalStream`` -- correct from
    every thread, including engine workers. Overwriting it with an ``mx.new_stream()``
    binds it to whichever thread wrote last, and every other thread in the process then
    dies on it with ``There is no Stream(gpu, N) in current thread``.

    That was not a theoretical hazard: it is what made the suite order-dependent (six
    failures on ``main``, all of them green in isolation, because ``test_simple_engine``
    ran ``engine/simple.py`` and poisoned the global for every engine test after it),
    and under ``--multi`` it is one engine reaching into another's in-flight generation.

    Asserted two ways on purpose. The runtime half proves the global is still
    thread-local *right now*, but it can only see the modules this test happened to
    import. The source half is the one that catches a reintroduction, because the defect
    is a write from anywhere in the package -- it does not need this test's imports to
    have run.
    """
    import importlib
    import pathlib
    import re

    import mlx.core as mx

    module = importlib.import_module("mlx_lm.generate")
    assert isinstance(module.generation_stream, mx.ThreadLocalStream), (
        "mlx_lm.generate.generation_stream is no longer a ThreadLocalStream "
        f"({type(module.generation_stream)!r}) -- something in this process rebound it"
    )

    package = pathlib.Path(__file__).resolve().parent.parent / "vllm_mlx"
    assign = re.compile(
        r"""(setattr\(\s*\w+\s*,\s*["']generation_stream["']|"""
        r"""\.generation_stream\s*=(?!=))"""
    )
    offenders = [
        f"{path.relative_to(package.parent)}:{n}: {line.strip()}"
        for path in sorted(package.rglob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if assign.search(line)
    ]
    assert not offenders, (
        "a module assigns to a shared `generation_stream`; use mlx-lm's "
        "ThreadLocalStream default and materialize arrays at the thread seam "
        "instead (see vllm_mlx/mlx_streams.py):\n" + "\n".join(offenders)
    )
