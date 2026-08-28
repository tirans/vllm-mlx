# SPDX-License-Identifier: Apache-2.0
"""Helpers for binding MLX generation streams to worker threads."""

import importlib
import logging
import threading
from collections.abc import Iterable

import mlx.core as mx

logger = logging.getLogger(__name__)

# Serialize stream rebinding so module-level generation_stream references are
# updated atomically across concurrent engine threads.
_STREAM_REBIND_LOCK = threading.Lock()


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> object:
    """Bind mlx-lm/mlx-vlm generation streams to the current thread.

    MLX streams are thread-local. If a model is loaded on one thread and
    generation runs on another, module-level generation streams created during
    import can point at a stream that does not exist in the worker thread.

    This intentionally creates a fresh stream for the current worker call and
    replaces module-level generation_stream handles under a process-local lock.
    It is an admission/ownership fix, not a batching optimization; callers
    should invoke it at worker-entry boundaries rather than inside token loops.
    """
    with _STREAM_REBIND_LOCK:
        default_stream = mx.new_stream(mx.default_device())
        mx.set_default_stream(default_stream)
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generation_stream"):
                setattr(module, "generation_stream", default_stream)
        return default_stream


def materialize_model_arrays(model: object) -> bool:
    """Force every array a model holds concrete, on the thread calling this.

    Returns True if anything was materialized.

    MLX >= 0.32 registers streams per thread and will not evaluate an op graph from
    a thread other than the one that built it. A model loaded on thread A therefore
    cannot be *first used* on thread B: `mlx_lm.load` leaves some of the module's
    arrays unevaluated, so the first forward pass on B tries to finish a graph that
    belongs to A's stream and dies with
    `RuntimeError: There is no Stream(gpu, N) in current thread`.

    That is precisely the shape of `EngineCore`: callers load a model on whatever
    thread they happen to be on, hand it to the engine, and the engine steps on its
    own worker thread. Measured on Llama-3.2-1B (load on main, forward on a worker):

        no eval                -> RuntimeError: There is no Stream(gpu, 1) ...
        mx.eval(parameters())  -> RuntimeError: There is no Stream(gpu, 1) ...
        mx.eval(model.state)   -> OK

    `parameters()` is not enough because the offending arrays are NOT parameters --
    they are the module's other buffers (rotary-embedding frequencies, masks) built
    during `__init__`. `Module.state` covers every attribute the module holds, which
    is why it is the one that works.

    This does not move work, it only moves *when* it happens: those arrays are
    evaluated on the first forward pass regardless. It just has to happen on the
    thread that created them.
    """
    state = getattr(model, "state", None)
    if state is None:
        return False
    try:
        mx.eval(state)
    except Exception:  # pragma: no cover - an exotic model must not break startup
        logger.warning(
            "could not materialize the model's arrays on this thread; a first "
            "forward pass from another thread may fail with a stream error",
            exc_info=True,
        )
        return False
    return True
