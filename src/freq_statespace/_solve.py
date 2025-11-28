"""General-purpose wrapper around Optimistix solvers."""
from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optimistix as optx
from jaxtyping import PyTree
from optimistix._least_squares import _ToMinimiseFn
from optimistix._misc import OutAsArray

from ._config import DeviceLike


@dataclass(frozen=True)
class SolveResult:
    """Container for optimization results."""
    
    theta: PyTree
    aux: Any  # problem-dependent auxiliary output
    loss_history: np.ndarray
    iter_count: int
    iter_times: np.ndarray
    converged: bool
    wall_time: float


def tree_device_put(tree: Any, device: DeviceLike) -> Any:
    """Move only inexact-array leaves of a PyTree to the given device."""
    if device is None:
        return tree

    def _to_device(x):
        if eqx.is_inexact_array(x):
            return jax.device_put(x, device=device)
        return x

    return jax.tree_util.tree_map(_to_device, tree)


def _available_platforms() -> set[str]:
    plats: set[str] = set()
    for p in ["cpu", "gpu", "tpu"]:
        try:
            if jax.devices(p):
                plats.add(p)
        except RuntimeError:
            pass
    return plats


def _get_device(device: DeviceLike):
    if device is None:
        return None

    if isinstance(device, jax.Device):
        return device

    platform = device.lower()
    try:
        # NOTE: Calling jax.devices with a string like below is currently an
        # experimental JAX feature; the API may change.
        candidates = jax.devices(platform)
    except RuntimeError as e:
        raise RuntimeError(
            f"Requested device '{platform}', but no such device is available. "
            f"Available platforms: {sorted(_available_platforms())}"
        ) from e

    return candidates[0]


@contextlib.contextmanager
def _maybe_default_device(device):
    """Set JAX's default device inside the context if provided."""
    if device is None:
        yield
    else:
        with jax.default_device(device):
            yield


def solve(
    theta_init: PyTree,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser,
    args: tuple | eqx.Module,
    loss_fn: Callable[[PyTree, tuple | eqx.Module], tuple],
    max_iter: int,
    print_every: int,
    device: DeviceLike = None
) -> SolveResult:
    """Solve an optimization problem with an Optimistix solver."""
    jax_device = _get_device(device)
    
    with _maybe_default_device(jax_device):
        # Parameters live on the chosen device
        theta_init = tree_device_put(theta_init, jax_device)
        args = tree_device_put(args, jax_device)
        
        loss_fn = OutAsArray(loss_fn)
        
        if isinstance(solver, optx.AbstractMinimiser):
            loss_fn = _ToMinimiseFn(loss_fn)

        loss_fn = eqx.filter_closure_convert(loss_fn, theta_init, args)

        tags = frozenset()
        f_struct, aux_struct = loss_fn.out_struct
        options: dict[str, Any] = {}

        step = eqx.filter_jit(
            eqx.Partial(
                solver.step, fn=loss_fn, args=args, options=options, tags=tags
            )
        )
        terminate = eqx.filter_jit(
            eqx.Partial(
                solver.terminate, fn=loss_fn, args=args, options=options, tags=tags
            )
        )

        state = solver.init(
            loss_fn, theta_init, args, options, f_struct, aux_struct, tags
        )
        converged = terminate(y=theta_init, state=state)[0]

        iter_count = 0
        theta = theta_init
        loss_history = np.zeros((max_iter,))
        iter_times = np.zeros((max_iter,))

        _ = step(y=theta, state=state)  # warm-up JIT

        start_time = time.time()

        while not converged and iter_count < max_iter:
            iter_start = time.perf_counter()
            theta, state, aux = step(y=theta, state=state)
            iter_end = time.perf_counter()

            scalar_loss = aux[0]
            if print_every > 0 and (iter_count % print_every == 0):
                jax.debug.print(
                    "    Iter {0} | loss = {1:.4e}", iter_count + 1, scalar_loss
                )

            loss_history[iter_count] = scalar_loss
            iter_times[iter_count] = iter_end - iter_start

            converged = terminate(y=theta, state=state)[0]
            iter_count += 1

        wall_time = time.time() - start_time

    return SolveResult(
        theta=theta,
        aux=aux,
        loss_history=loss_history[:iter_count],
        iter_count=iter_count,
        iter_times=iter_times[:iter_count],
        converged=converged,
        wall_time=wall_time,
    )
