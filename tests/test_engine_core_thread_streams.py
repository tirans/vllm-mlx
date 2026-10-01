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
            self.ssd_close_threads: list[int] = []

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

        def close_ssd_tier(self):
            self.ssd_close_threads.append(threading.get_ident())

    scheduler = FakeScheduler()
    engine.scheduler = scheduler

    await asyncio.wait_for(engine._engine_loop(), timeout=2)

    assert scheduler.step_threads
    assert len(set(scheduler.step_threads)) == 1
    assert scheduler.step_threads[0] != main_thread
    assert scheduler.close_threads == [scheduler.step_threads[0]]
    assert scheduler.ssd_close_threads
    assert scheduler.ssd_close_threads[0] != main_thread


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


def test_a_stream_created_on_one_thread_faults_only_at_the_first_op_elsewhere():
    """The MLX mechanism behind every incident in this file, pinned directly.

    ``mx.new_stream`` registers the stream in the calling thread only, and using
    it from any other thread raises ``There is no Stream(gpu, N) in current
    thread``. The half worth pinning is *when*: entering ``with mx.stream(s)``
    on a foreign thread does NOT raise -- the fault fires at the first op inside
    the block. That latency is what let a shared stream hide in
    ``MLLMBatchGenerator`` -- an idle ``next()`` crosses the seam, runs no op,
    and returns clean, so nothing detonated until a real request arrived.
    """
    from concurrent.futures import ThreadPoolExecutor

    import mlx.core as mx
    import pytest

    if not hasattr(mx, "clear_streams"):  # pragma: no cover
        pytest.skip("requires MLX >= 0.32, where streams are registered per thread")

    owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owner")
    other = ThreadPoolExecutor(max_workers=1, thread_name_prefix="other")
    try:
        stream = owner.submit(lambda: mx.new_stream(mx.default_device())).result()

        def enter_only():
            with mx.stream(stream):
                pass
            return "ok"

        def enter_and_op():
            with mx.stream(stream):
                mx.eval(mx.array([1.0, 2.0]) * 2)
            return "ok"

        assert other.submit(enter_only).result() == "ok"
        with pytest.raises(RuntimeError, match=r"no Stream\("):
            other.submit(enter_and_op).result()
        assert owner.submit(enter_and_op).result() == "ok"
    finally:
        owner.shutdown(wait=True)
        other.shutdown(wait=True)


def test_mllm_batch_generators_own_their_streams_per_instance():
    """Two MLLM engines on two threads must not share one generation stream.

    ``MLLMBatchGenerator._stream`` used to be a class attribute: the first
    instance's constructing thread minted an ``mx.new_stream()`` and every later
    instance in the process reused it. Reproduced before the fix, with plain
    fake models and no VLM weights: the second generator's thread could neither
    step (``with mx.stream(...)`` around an op -- the shape of all four
    production sites) nor ``close()`` (``mx.synchronize`` on the foreign stream
    raises, and the wired-limit restore on the line after it never runs). Both
    died with the production string ``There is no Stream(gpu, N) in current
    thread``. Same defect shape as the deleted ``bind_generation_streams``: a
    thread-affine stream in cross-thread-visible storage.

    The invariant now: no ``_stream`` on the class; each instance mints its own
    on its constructing thread -- which is its stepping thread, because
    ``MLLMScheduler._ensure_batch_generator`` runs inside ``step()``.
    """
    import contextlib
    import logging
    from concurrent.futures import ThreadPoolExecutor

    import mlx.core as mx
    import pytest

    if not hasattr(mx, "clear_streams"):  # pragma: no cover
        pytest.skip("requires MLX >= 0.32, where streams are registered per thread")

    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

    assert not hasattr(MLLMBatchGenerator, "_stream"), (
        "MLLMBatchGenerator grew a class-level _stream again; the stream must "
        "be per-instance, minted on the constructing thread"
    )

    class FakeLanguageModel:
        pass

    class FakeModel:
        language_model = FakeLanguageModel()

    class FakeProcessor:
        pass

    def build():
        return MLLMBatchGenerator(model=FakeModel(), processor=FakeProcessor())

    def step_own_stream(generator):
        # The shape of every production site: an op inside the generator's
        # stream context, on the generator's own thread.
        with mx.stream(generator._stream):
            mx.eval(mx.array([1.0, 2.0]) * 2)
        return "ok"

    logging.getLogger("vllm_mlx.mllm_batch_generator").setLevel(logging.ERROR)
    first = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-a")
    second = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-b")
    gen_a = gen_b = None
    try:
        gen_a = first.submit(build).result()
        gen_b = second.submit(build).result()

        assert gen_a._stream is not gen_b._stream, (
            "two generators share one stream; the second engine's thread "
            "cannot use it and dies on its first real request"
        )
        assert first.submit(step_own_stream, gen_a).result() == "ok"
        assert second.submit(step_own_stream, gen_b).result() == "ok"

        # close() synchronizes the stream before restoring the wired limit;
        # it must succeed from each generator's own thread.
        second.submit(gen_b.close).result()
        gen_b = None
        first.submit(gen_a.close).result()
        gen_a = None
    finally:
        # Best-effort cleanup on the red path so a failure above does not also
        # leak the raised wired limit.
        for executor, generator in ((second, gen_b), (first, gen_a)):
            if generator is not None:
                with contextlib.suppress(Exception):
                    executor.submit(generator.close).result()
        first.shutdown(wait=True)
        second.shutdown(wait=True)
