# SPDX-License-Identifier: Apache-2.0
"""End-to-end guards for MLX stream ownership across the engine's threads.

Two incidents live here, and the second one corrected the first one's story.

**#407.** ``mx.eval`` over KV cache state raised ``RuntimeError: There is no
Stream(gpu, N) in current thread`` for Llama 3.x, because mlx-lm's module-level
``generation_stream`` is thread-local. The docstring here used to claim the fix was
"keep step on the event-loop thread". That is no longer true, and describing the
code as it was is worse than saying nothing: ``EngineCore`` steps on a dedicated
worker thread today, deliberately.

**2026-08-27.** Every prefix-cache HIT crashed the request it hit on, with the same
error, and one endpoint restart did not clear it. The cache is fetched on the
asyncio event-loop thread (``Scheduler.add_request``) and first read on the
engine-core worker thread, and MLX >= 0.32 will not evaluate an op graph from a
thread other than the one that built it. So the real invariant is not "which thread
steps" but: **nothing lazy may cross between them.** Two changes enforce it -- the
cache evaluates at the seam, and the scheduler owns a stream that is not registered
per-thread.

``test_prefix_cache_hit_survives_engine_threads`` covers the second incident: the
unit tests in ``test_memory_cache_thread_affinity.py`` pin the mechanism, this one
proves a real model on a real engine actually serves a cache hit.
"""

import asyncio
import logging

import pytest

# Llama 3.x reliably surfaces the cross-thread stream mismatch. Qwen3 does not.
TEST_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    try:
        from mlx_lm import load

        return load(TEST_MODEL)
    except Exception as e:
        pytest.skip(f"Could not load model {TEST_MODEL}: {e}")


@pytest.mark.anyio
async def test_engine_core_no_cross_thread_stream_error(model_and_tokenizer, caplog):
    """EngineCore must run prefill + decode without raising
    ``There is no Stream(gpu, N) in current thread``.

    A regression that moves ``scheduler.step`` off the event-loop thread
    (e.g. re-introducing a ThreadPoolExecutor) reintroduces issue #407.
    """
    from vllm_mlx import AsyncEngineCore, SamplingParams

    model, tokenizer = model_and_tokenizer
    params = SamplingParams(max_tokens=5, temperature=0.0)

    caplog.set_level(logging.ERROR, logger="vllm_mlx.scheduler")

    engine = AsyncEngineCore(model, tokenizer)
    await engine.__aenter__()
    await asyncio.sleep(0.05)

    rid = await engine.add_request("Hello", params)
    tokens = 0
    async for out in engine.stream_outputs(rid, timeout=30):
        tokens += 1
        if out.finished:
            break

    bg = engine.engine.scheduler.batch_generator
    await engine.__aexit__(None, None, None)

    stream_errors = [
        r.message
        for r in caplog.records
        if "Stream(gpu" in r.message or "no Stream" in r.message
    ]
    assert (
        not stream_errors
    ), f"scheduler logged cross-thread stream errors: {stream_errors}"
    assert tokens > 0, "no tokens streamed"
    assert bg is not None, (
        "batch generator was None after generation, meaning the scheduler's "
        "error-recovery path fired. See issue #407."
    )


@pytest.mark.anyio
async def test_prefix_cache_hit_survives_engine_threads(model_and_tokenizer, caplog):
    """A prefix-cache HIT must generate, not abort the request it hit on.

    This is the 2026-08-27 crash end to end. The setup matters, and each knob is
    here to stop the test passing vacuously:

    * ``kv_cache_quantization`` on -- the crash needs the ``mx.dequantize`` ops that
      only the quantized fetch path creates.
    * ``kv_cache_min_quantize_tokens`` lowered -- at the 256 default a short test
      prompt is stored unquantized and nothing is exercised.
    * the same prompt twice -- the first request MISSES and stores; only the second
      can hit. Asserting on one request would test nothing.

    Before the fix the second request raised ``no Stream(gpu, N)`` inside step, and
    the scheduler aborted it with ``finish_reason="error"`` after logging
    ``Error in batch generation step``.
    """
    from vllm_mlx import AsyncEngineCore, SamplingParams
    from vllm_mlx.engine_core import EngineConfig
    from vllm_mlx.scheduler import SchedulerConfig

    model, tokenizer = model_and_tokenizer
    config = EngineConfig(
        scheduler_config=SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            kv_cache_quantization=True,
            kv_cache_min_quantize_tokens=1,
        )
    )

    caplog.set_level(logging.ERROR, logger="vllm_mlx.scheduler")

    # Long enough to clear MemoryCacheConfig.min_prefix_tokens (128), or the entry
    # is declined on store and the second request is another miss.
    prompt = "Count carefully and explain each step. " * 40
    params = SamplingParams(max_tokens=8, temperature=0.0)

    engine = AsyncEngineCore(model, tokenizer, config)
    await engine.__aenter__()
    try:
        await asyncio.sleep(0.05)

        streamed = []
        for _ in range(2):
            rid = await engine.add_request(prompt, params)
            tokens = 0
            finish_reason = None
            async for out in engine.stream_outputs(rid, timeout=60):
                tokens += 1
                finish_reason = out.finish_reason
                if out.finished:
                    break
            streamed.append((tokens, finish_reason))

        cache = engine.engine.scheduler.memory_aware_cache
        hits = cache.get_stats().get("hits", 0) if cache is not None else 0
    finally:
        await engine.__aexit__(None, None, None)

    assert hits > 0, (
        f"no cache hit was recorded ({streamed}), so this test would pass even "
        "with the bug present -- check min_prefix_tokens and the prompt length"
    )

    stream_errors = [
        r.message
        for r in caplog.records
        if "no Stream(" in r.message or "Stream(gpu" in r.message
    ]
    assert (
        not stream_errors
    ), f"a prefix-cache hit raised a cross-thread stream error: {stream_errors}"
    assert streamed[1][0] > 0, f"the cache-hit request streamed no tokens: {streamed}"
    assert (
        streamed[1][1] != "error"
    ), f"the cache-hit request was aborted by the scheduler: {streamed}"


def test_a_model_first_used_off_its_loading_thread_needs_materializing():
    """The invariant `Scheduler.__init__` upholds, pinned directly.

    `mlx_lm.load` leaves some of a module's arrays unevaluated, and MLX >= 0.32 will
    not finish an op graph from a thread other than the one that built it. So a model
    loaded on thread A and *first used* on thread B dies during that first forward --
    which is exactly the engine's shape, since callers load wherever they happen to be
    and `EngineCore` steps on its own worker.

    Both halves are asserted, because the interesting half is the negative one: this
    is why `mx.eval(model.parameters())` is not the fix. The arrays at fault are not
    parameters -- they are the module's other buffers (rope frequencies, masks) built
    in `__init__` -- and only `Module.state` reaches them.

    Loads its own copy rather than using the module fixture: that one has already been
    used by the tests above, so it is materialized and would pass either way.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import mlx.core as mx

    try:
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except Exception as e:  # pragma: no cover
        pytest.skip(f"mlx_lm unavailable: {e}")

    if not hasattr(mx, "clear_streams"):  # pragma: no cover
        pytest.skip("requires MLX >= 0.32, where streams are registered per thread")

    from vllm_mlx.mlx_streams import materialize_model_arrays

    def first_forward(model, tokenizer):
        cache = make_prompt_cache(model)
        tokens = mx.array([tokenizer.encode("Hello there, how are you today?")])
        model(tokens, cache=cache)
        mx.eval([c.state for c in cache])
        return "ok"

    def load_here():
        assert threading.get_ident() is not None
        return load(TEST_MODEL)

    # --- negative: parameters() is not enough --------------------------------
    loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="loader")
    stepper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stepper")
    try:
        model, tokenizer = loader.submit(load_here).result()
        loader.submit(lambda: mx.eval(model.parameters())).result()
        with pytest.raises(RuntimeError, match=r"no Stream\("):
            stepper.submit(first_forward, model, tokenizer).result()
    finally:
        loader.shutdown(wait=True)
        stepper.shutdown(wait=True)

    # --- positive: Module.state is ------------------------------------------
    loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="loader")
    stepper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stepper")
    try:
        model, tokenizer = loader.submit(load_here).result()
        assert loader.submit(materialize_model_arrays, model).result() is True
        assert stepper.submit(first_forward, model, tokenizer).result() == "ok"
    finally:
        loader.shutdown(wait=True)
        stepper.shutdown(wait=True)
