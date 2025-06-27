from dataclasses import dataclass
import time
from typing import Any, cast, Union

import equinox as eqx
import jax
from jaxtyping import PyTree, Scalar
import numpy as np

import optimistix as optx
from optimistix._custom_types import Aux, Fn, Out, Y
from optimistix._least_squares import _ToMinimiseFn
from optimistix._misc import OutAsArray


@dataclass(frozen=True)
class SolveResult:
    """Result of the optimization process."""
    theta: Y
    aux: Any
    loss_history: np.ndarray
    iter_count: int
    iter_times: np.ndarray
    converged: bool
    wall_time: float


def solve(
    theta_init: Y,
    solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractMinimiser],
    args: PyTree[Any],
    loss_fn: Fn,
    max_iter: int
) -> SolveResult:

    loss_fn = OutAsArray(loss_fn)
    if isinstance(solver, optx.AbstractMinimiser):
        loss_fn = _ToMinimiseFn(loss_fn)
        loss_fn = eqx.filter_closure_convert(loss_fn, theta_init, args)
        loss_fn = cast(Fn[Y, Scalar, Aux], loss_fn)
    elif isinstance(solver, optx.AbstractLeastSquaresSolver):
        loss_fn = eqx.filter_closure_convert(loss_fn, theta_init, args)
        loss_fn = cast(Fn[Y, Out, Aux], loss_fn)
    else:
        raise ValueError('Unknown solver type.')

    tags = frozenset()
    f_struct, aux_struct = loss_fn.out_struct
    options = {}

    # JIT compile step and terminate
    step = eqx.filter_jit(
        eqx.Partial(solver.step, fn=loss_fn, args=args, options=options, tags=tags)  # noqa: E501
    )
    terminate = eqx.filter_jit(
        eqx.Partial(solver.terminate, fn=loss_fn, args=args, options=options, tags=tags)  # noqa: E501
    )

    # Initial state
    state = solver.init(loss_fn, theta_init, args, options, f_struct, aux_struct, tags)  # noqa: E501
    converged = terminate(y=theta_init, state=state)[0]

    iter_count = 0
    theta = theta_init
    loss_history = np.zeros((max_iter,))
    iter_times = np.zeros((max_iter,))

    # Warm up JIT compilation
    _ = step(y=theta, state=state)

    start_time = time.time()

    while not converged and iter_count < max_iter:
        iter_start = time.perf_counter()
        theta, state, aux = step(y=theta, state=state)
        iter_end = time.perf_counter()

        loss = aux[0]
        jax.debug.print(
            "   Iteration {iter_count}, Loss: {loss:.4e}",
            iter_count=iter_count,
            loss=loss
        )

        loss_history[iter_count] = loss
        iter_times[iter_count] = iter_end - iter_start

        converged = terminate(y=theta, state=state)[0]
        iter_count += 1

    wall_time = time.time() - start_time

    return SolveResult(theta, aux, loss_history[:iter_count], iter_count,
                       iter_times[:iter_count], converged, wall_time)
