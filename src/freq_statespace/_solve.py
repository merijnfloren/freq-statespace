"""General-purpose wrapper around Optimistix solvers."""
from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable
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
    aux: Any
    loss_history: np.ndarray
    iter_count: int
    iter_times: np.ndarray
    converged: bool
    wall_time: float
    

class BatchedArgs(eqx.Module):
    """Wrapper that enables batching of arguments over the realization axis."""
    
    args: eqx.Module
    num_realizations: int
    slice_fn: Callable[[eqx.Module, slice], eqx.Module]
    
    def iter_batches(self, batch_size: int) -> Iterable[BatchedArgs]:
        """Yield batched objects over realizations."""
        full_batches = self.num_realizations // batch_size
        for k in range(full_batches):
            start = k * batch_size
            end = start + batch_size
            batch_args = self.slice_fn(self.args, slice(start, end))
            yield BatchedArgs(
                args=batch_args,
                num_realizations=batch_size,
                slice_fn=self.slice_fn
            )


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


def _prepare_solve(jax_device, batch_size, theta_init, loss_fn, args, solver):
    
    # # Parameters live on the chosen device
    # theta_init = tree_device_put(theta_init, jax_device)
    # args = tree_device_put(args, jax_device)
    
    # batched = (
    #     isinstance(args, BatchedArgs)
    #     and batch_size is not None
    #     and int(batch_size) != int(args.num_realizations)
    # )

    # if batched:
    #     args = next(args.iter_batches(batch_size))
    #     args = args.args

    # loss_fn = OutAsArray(loss_fn)
    # if isinstance(solver, optx.AbstractMinimiser):
    #     loss_fn = _ToMinimiseFn(loss_fn)
    #     loss_fn = eqx.filter_closure_convert(
    #         loss_fn, theta_init, args
    #     )
    # elif isinstance(solver, optx.AbstractLeastSquaresSolver):
    #     loss_fn = eqx.filter_closure_convert(
    #         loss_fn, theta_init, args
    #     )
    # else:
    #     raise ValueError("Unknown solver type.")

    # tags = frozenset()
    # f_struct, aux_struct = loss_fn.out_struct
    # options: dict[str, Any] = {}

    # step = eqx.filter_jit(
    #     eqx.Partial(
    #         solver.step, fn=loss_fn, args=args, options=options, tags=tags
    #     )
    # )
    # terminate = eqx.filter_jit(
    #     eqx.Partial(
    #         solver.terminate, fn=loss_fn, args=args, options=options, tags=tags
    #     )
    # )

    # state = solver.init(
    #     loss_fn, theta_init, args, options, f_struct, aux_struct, tags
    # )
    # converged = terminate(y=theta_init, state=state)[0]

    # _ = step(y=theta_init, state=state)  # warm-up JIT

    # return step, terminate, state, converged, batched
    # Put parameters and args on device
    theta_init = tree_device_put(theta_init, jax_device)
    args = tree_device_put(args, jax_device)

    batched = False
    if isinstance(args, BatchedArgs):
        if batch_size is not None and int(batch_size) != int(args.num_realizations):
            sample_args = next(args.iter_batches(batch_size)).args  # TODO: CHECK IF THIS IS REALLY THE FIRST BATCH
            batched = True
        else:
            sample_args = args.args
    else:
        sample_args = args

    # # For compilation we just need a representative args with the right shape
    # if batched:
    #     first_batch = next(args.iter_batches(batch_size))
    #     sample_args = first_batch.args
    # else:
    #     sample_args = args

    # Ensure loss_fn returns arrays in a consistent way
    loss_fn = OutAsArray(loss_fn)
    if isinstance(solver, optx.AbstractMinimiser):
        loss_fn = _ToMinimiseFn(loss_fn)

    # Closure-convert loss_fn so its PyTree state is explicit
    loss_fn = eqx.filter_closure_convert(loss_fn, theta_init, sample_args)

    tags = frozenset()
    f_struct, aux_struct = loss_fn.out_struct
    options: dict[str, Any] = {}

    def _step(theta, state, args_):
        return solver.step(
            fn=loss_fn,
            y=theta,
            state=state,
            args=args_,
            options=options,
            tags=tags,
        )

    def _terminate(theta, state, args_):
        return solver.terminate(
            fn=loss_fn,
            y=theta,
            state=state,
            args=args_,
            options=options,
            tags=tags,
        )

    # JIT just the shape; args_ is a normal (dynamic) argument
    step = eqx.filter_jit(_step)
    terminate = eqx.filter_jit(_terminate)

    # Initialise solver state using the sample args (for shape)
    state = solver.init(
        loss_fn, theta_init, sample_args, options, f_struct, aux_struct, tags
    )
    converged = terminate(theta_init, state, sample_args)[0]

    # Warm-up JIT
    _ = step(theta_init, state, sample_args)

    return step, terminate, state, converged, batched


def solve(
    theta_init: PyTree,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser,
    args: tuple | BatchedArgs,
    loss_fn: Callable[[PyTree, tuple | eqx.Module], tuple],
    max_iter: int,
    print_every: int,
    device: DeviceLike = None,
    batch_size: int | None = None,
) -> SolveResult:
    """Solve an optimization problem with an Optimistix solver."""
    jax_device = _get_device(device)
    
    with _maybe_default_device(jax_device):
  
        step, terminate, state, converged, batched = _prepare_solve(jax_device, batch_size, theta_init, loss_fn, args, solver)
        
        if isinstance(args, BatchedArgs):
            if batch_size is not None and int(batch_size) != int(args.num_realizations):
                pass
            else:
                args = args.args
 
                

        iter_count = 0
        theta = theta_init
        loss_history = np.zeros((max_iter,))
        iter_times = np.zeros((max_iter,))

        start_time = time.time()

        while not converged and iter_count < max_iter:
            iter_start = time.perf_counter()
            
            iter_ = 0
            if isinstance(args, BatchedArgs) and batched:
                for batch_args in args.iter_batches(batch_size):
                    # jax.debug.print(
                    #     "  Batch {0}/{1}", iter_ + 1, args.num_realizations // batch_size
                    # )
                    # jax.debug.print(
                    #     "    Y shape: {0}", batch_args.args.f_data[3][55]
                    # )
                    theta, state, aux = step(theta, state, batch_args.args)
                    iter_ += 1
            else:  
                theta, state, aux = step(theta, state, args)
            
            
            iter_end = time.perf_counter()

            scalar_loss = aux[0]
            if print_every > 0 and (iter_count % print_every == 0):
                jax.debug.print(
                    "    Iter {0} | loss = {1:.4e}", iter_count + 1, scalar_loss
                )

            loss_history[iter_count] = scalar_loss
            iter_times[iter_count] = iter_end - iter_start

            converged = terminate(theta, state, args)[0]
            iter_count += 1

        wall_time = time.time() - start_time

    return SolveResult(
        theta=theta,
        aux=aux, # TODO: CHECK IF THIS IS CORRECT WHEN BATCHING, IT SEEMS ONLY THE LAST BATCH AUX IS RETURNED
        loss_history=loss_history[:iter_count],
        iter_count=iter_count,
        iter_times=iter_times[:iter_count],
        converged=converged,
        wall_time=wall_time,
    )


# def solve(
#     theta_init: Y,
#     solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser,
#     args: tuple | BatchedArgs,
#     loss_fn: Fn,
#     max_iter: int,
#     print_every: int,
#     device: DeviceLike = None,
#     batch_size: int | None = None,
# ) -> SolveResult:
#     """Solve an optimization problem with an Optimistix solver."""
#     jax_device = _get_device(device)
    
#     with _maybe_default_device(jax_device):
#         # Parameters live on the chosen device
#         theta_init = tree_device_put(theta_init, jax_device)
#         args = tree_device_put(args, jax_device)
        
#         if isinstance(args, BatchedArgs):
#             # showcase batching by printing the shape of Y per batcch
#             batch_size = 4 if batch_size is None else batch_size
#             # for batch_args in args.iter_batches(batch_size):
#             #     jax.debug.print(
#             #         "  Batch with Y shape: {0}", batch_args.args.f_data[3].shape
#             #     )
#             args = args.args

#         loss_fn = OutAsArray(loss_fn)

#         if isinstance(solver, optx.AbstractMinimiser):
#             loss_fn = _ToMinimiseFn(loss_fn)
#             loss_fn = eqx.filter_closure_convert(
#                 loss_fn, theta_init, args
#             )
#             loss_fn = cast(Fn[Y, Scalar, Aux], loss_fn)
#         elif isinstance(solver, optx.AbstractLeastSquaresSolver):
#             loss_fn = eqx.filter_closure_convert(
#                 loss_fn, theta_init, args
#             )
#             loss_fn = cast(Fn[Y, Out, Aux], loss_fn)
#         else:
#             raise ValueError("Unknown solver type.")

#         tags = frozenset()
#         f_struct, aux_struct = loss_fn.out_struct
#         options: dict[str, Any] = {}

#         step = eqx.filter_jit(
#             eqx.Partial(
#                 solver.step, fn=loss_fn, args=args, options=options, tags=tags
#             )
#         )
#         terminate = eqx.filter_jit(
#             eqx.Partial(
#                 solver.terminate, fn=loss_fn, args=args, options=options, tags=tags
#             )
#         )

#         state = solver.init(
#             loss_fn, theta_init, args, options, f_struct, aux_struct, tags
#         )
#         converged = terminate(y=theta_init, state=state)[0]

#         iter_count = 0
#         theta = theta_init
#         loss_history = np.zeros((max_iter,))
#         iter_times = np.zeros((max_iter,))

#         _ = step(y=theta, state=state)  # warm-up JIT

#         start_time = time.time()

#         while not converged and iter_count < max_iter:
#             iter_start = time.perf_counter()
#             theta, state, aux = step(y=theta, state=state)
#             iter_end = time.perf_counter()

#             scalar_loss = aux[0]
#             if print_every > 0 and (iter_count % print_every == 0):
#                 jax.debug.print(
#                     "    Iter {0} | loss = {1:.4e}", iter_count + 1, scalar_loss
#                 )

#             loss_history[iter_count] = scalar_loss
#             iter_times[iter_count] = iter_end - iter_start

#             converged = terminate(y=theta, state=state)[0]
#             iter_count += 1

#         wall_time = time.time() - start_time

#     return SolveResult(
#         theta=theta,
#         aux=aux,
#         loss_history=loss_history[:iter_count],
#         iter_count=iter_count,
#         iter_times=iter_times[:iter_count],
#         converged=converged,
#         wall_time=wall_time,
#     )
