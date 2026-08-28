# SPDX-License-Identifier: Apache-2.0
"""Thread-affinity regression guard for the prefix-cache crash loop.

Incident (2026-08-27): every prefix-cache HIT killed the request it hit on, with
``RuntimeError: There is no Stream(gpu, N) in current thread`` surfacing to clients
as "the engine aborted this request while generating". 10/10 crashes followed a hit;
0/6 cold requests crashed. A downstream map died 42 consecutive times.

The mechanism is a thread split, not a cache-logic bug:

* ``MemoryAwarePrefixCache.fetch`` is called from ``Scheduler.add_request``, which
  ``AsyncEngineCore.add_request`` runs on the **asyncio event-loop thread**.
* With ``kv_quantize`` on, every hit path returns ``_dequantize_cache(...)``, which
  builds ``mx.dequantize`` + slice ops. Those are **lazy** — no ``mx.eval``.
* The arrays are first evaluated inside ``scheduler.step``, which the engine loop
  runs on its **engine-core worker thread**.

MLX >= 0.32 registers streams per thread, so an op graph built against a stream
belonging to thread A cannot be evaluated from thread B. The graph is built on the
loop thread and evaluated on the worker thread, so every hit is poisoned fresh —
which is why one endpoint restart did not clear the loop.

These tests deliberately use **persistent** worker threads
(``ThreadPoolExecutor(max_workers=1)``), never bare ``threading.Thread``. A thread
that has exited takes its stream registry with it, so ephemeral threads fail for a
second, unrelated reason and would make a broken fix look tested.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is not installed (non-Apple platform)")
pytest.importorskip("mlx_lm", reason="mlx_lm is not installed")

# Per-thread stream registration — and therefore this whole failure mode — arrived in
# MLX 0.32. On older MLX the ops are not thread-affine and there is nothing to guard.
pytestmark = pytest.mark.skipif(
    not hasattr(mx, "clear_streams"),
    reason="requires MLX >= 0.32, where streams are registered per thread",
)

STREAM_ERROR = "no Stream("


@pytest.fixture
def worker():
    """A persistent stand-in for the engine-core worker thread."""
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-core")
    try:
        yield lambda fn, *a: pool.submit(fn, *a).result()
    finally:
        pool.shutdown(wait=True)


@pytest.fixture
def loop_thread():
    """A persistent stand-in for the asyncio event-loop thread that calls fetch()."""
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="loop")
    try:
        yield lambda fn, *a: pool.submit(fn, *a).result()
    finally:
        pool.shutdown(wait=True)


def _make_layers(n_tokens: int, n_layers: int = 2):
    """Real ``KVCache`` layers, sized so ``mx.quantize`` accepts them.

    ``mx.quantize`` requires the last dimension to be divisible by the group size
    (64), so head_dim is 128 rather than a token number.
    """
    from mlx_lm.models.cache import KVCache

    layers = []
    for _ in range(n_layers):
        kv = KVCache()
        kv.keys = mx.random.normal((1, 2, n_tokens, 128)).astype(mx.float16)
        kv.values = mx.random.normal((1, 2, n_tokens, 128)).astype(mx.float16)
        kv.offset = n_tokens
        mx.eval(kv.keys, kv.values)
        layers.append(kv)
    return layers


def _consume(layers) -> float:
    """Do to the fetched cache what the stepping thread does: actually read it.

    ``mx.eval`` alone is the weaker assertion; summing forces the arrays through a
    real kernel on the consuming thread, which is what attention does.
    """
    total = 0.0
    for layer in layers:
        for arr in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            # mx.quantize returns a list of (w, scales, biases) on MLX 0.32, a tuple
            # on older builds; the wrapper stores whatever it was handed.
            if isinstance(arr, (tuple, list)):
                mx.eval(*arr)
                total += float(mx.sum(arr[0].astype(mx.float32)).item())
            elif arr is not None:
                mx.eval(arr)
                total += float(mx.sum(arr.astype(mx.float32)).item())
    return total


def _new_cache(**overrides):
    from unittest.mock import MagicMock

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    config = MemoryCacheConfig(
        kv_quantize=True,
        kv_bits=8,
        kv_group_size=64,
        # Both floors default high enough (256 / 128) that a small unit-test cache
        # would be silently declined and the test would pass vacuously.
        kv_min_quantize_tokens=1,
        min_prefix_tokens=1,
        **overrides,
    )
    return MemoryAwarePrefixCache(MagicMock(), config)


class TestFetchAcrossEngineThreads:
    """The crash itself: store on the worker, fetch on the loop, consume on the worker."""

    @pytest.mark.parametrize(
        ("stored_tokens", "fetched_tokens", "expected_match"),
        [
            (list(range(64)), list(range(64)), "exact"),
            # A supersequence hit takes a different return path (it trims first), and
            # the incident hit both. Testing only the exact path would leave the
            # trim-then-dequantize path unguarded.
            (list(range(64)), list(range(48)), "supersequence"),
        ],
        ids=["exact-hit", "supersequence-hit"],
    )
    def test_fetch_result_consumable_from_stepping_thread(
        self, worker, loop_thread, stored_tokens, fetched_tokens, expected_match
    ):
        cache = _new_cache()

        # store() runs on the worker thread, as it does in production.
        assert worker(cache.store, stored_tokens, _make_layers(len(stored_tokens)))

        # fetch() runs on the event-loop thread, as it does in production.
        fetched, remaining = loop_thread(cache.fetch, fetched_tokens)
        assert fetched is not None, "expected a cache hit — the test is set up wrong"
        assert cache._last_match_type == expected_match
        assert remaining == []

        # And the stepping thread has to be able to USE it. This is the assertion the
        # incident violated.
        try:
            worker(_consume, fetched)
        except RuntimeError as error:  # pragma: no cover - the red state
            if STREAM_ERROR in str(error):
                pytest.fail(
                    f"cache fetch returned thread-affine lazy ops: {error}. "
                    "fetch() must evaluate its result on the fetching thread."
                )
            raise

    def test_stored_entry_is_not_left_lazy(self, worker, loop_thread):
        """``store`` must evaluate what it quantizes, on the thread that quantized it.

        ``_quantize_cache`` builds lazy ``mx.quantize`` ops. If they are stored
        unevaluated, the entry itself is affine to the storing thread and a later
        ``fetch`` on any other thread is poisoned before it dequantizes anything —
        so evaluating only at fetch time would fix half the bug.
        """
        cache = _new_cache()
        tokens = list(range(64))
        assert worker(cache.store, tokens, _make_layers(len(tokens)))

        entry = cache._entries[tuple(tokens)]
        try:
            loop_thread(_consume, entry.cache)
        except RuntimeError as error:  # pragma: no cover - the red state
            if STREAM_ERROR in str(error):
                pytest.fail(
                    f"store() left the quantized entry lazy: {error}. "
                    "The entry is bound to the storing thread."
                )
            raise


class TestThreadAffinityMechanism:
    """Why the fix is 'evaluate on the producing thread' and not a bigger refactor.

    This is the hypothesis test that sized the fix. If the second case ever goes red,
    concrete buffers have become thread-affine too and evaluating at the seam is no
    longer sufficient — the fetch would have to be restructured so the ops are
    *created* on the stepping thread.
    """

    def test_lazy_ops_do_not_survive_a_thread_hop(self, worker, loop_thread):
        from vllm_mlx.memory_cache import _dequantize_cache, _quantize_cache

        quantized = worker(lambda: _quantize_cache(_make_layers(64)))
        worker(_consume, quantized)

        lazy = loop_thread(_dequantize_cache, quantized)
        with pytest.raises(RuntimeError, match=r"no Stream\("):
            worker(_consume, lazy)

    def test_evaluated_arrays_do_survive_a_thread_hop(self, worker, loop_thread):
        from vllm_mlx.memory_cache import _dequantize_cache, _quantize_cache

        quantized = worker(lambda: _quantize_cache(_make_layers(64)))
        worker(_consume, quantized)

        def dequantize_and_evaluate():
            out = _dequantize_cache(quantized)
            _consume(out)
            return out

        evaluated = loop_thread(dequantize_and_evaluate)
        # No raise, and the values are real: this is what makes evaluating at the
        # seam a complete fix rather than a narrowing of the race.
        assert worker(_consume, evaluated) == pytest.approx(
            loop_thread(_consume, evaluated)
        )
