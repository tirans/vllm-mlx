# SPDX-License-Identifier: Apache-2.0
"""Helpers for keeping MLX arrays usable across the threads an engine runs on.

There used to be a `bind_generation_streams()` here, called at every worker-entry
boundary. It is gone, and the reason is worth keeping: it created a fresh
`mx.new_stream()` -- which MLX >= 0.32 registers in the *calling* thread -- and then
published it into the module-level `mlx_lm.generate.generation_stream` that every
engine, scheduler and test in the process shares. Whichever thread bound last owned
that global, and every other thread then died on it with
`RuntimeError: There is no Stream(gpu, N) in current thread`.

Measured: mlx-lm's own default for that attribute is an `mx.ThreadLocalStream`, which
is already correct from every thread, so overwriting it was strictly a downgrade. It
never bought anything either -- `mx.set_default_stream` is itself per-thread in MLX
0.32 (a worker that never binds still gets its own default stream), and
`mlx_vlm.generate` has no `generation_stream` attribute at all, so that half of the
call was always a no-op.

What actually makes a worker safe is `materialize_model_arrays` below: nothing lazy
may cross a thread boundary.
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


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
