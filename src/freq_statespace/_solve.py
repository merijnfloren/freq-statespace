"""General-purpose wrapper/logger around Optimistix solvers."""
import time
from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
import numpy as np
import optimistix as optx
from jaxtyping import PyTree, Scalar
from optimistix._custom_types import Aux, Fn, Out, Y
from optimistix._least_squares import _ToMinimiseFn
from optimistix._misc import OutAsArray


@dataclass(frozen=True)
class SolveResult:
    """Result of the optimization process.

    Attributes
    ----------
    theta : Any
        Final solution vector or PyTree of parameters.
    aux : tuple
        Auxiliary output from the final loss evaluation. The first element is
        always the scalar loss value.
    loss_history : np.ndarray
        Sequence of loss values at each iteration.
    iter_count : int
        Number of iterations executed before termination.
    iter_times : np.ndarray
        Wall-clock duration (in seconds) for each iteration.
    converged : bool
        Whether the termination criterion was met.
    wall_time : float
        Total wall-clock time (in seconds) spent in the optimization loop.

    """

    theta: Y
    aux: Any
    loss_history: np.ndarray
    iter_count: int
    iter_times: np.ndarray
    converged: bool
    wall_time: float


def solve(
    theta_init: Y,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser,
    args: PyTree[Any],
    loss_fn: Fn,
    max_iter: int
) -> SolveResult:
    """Solve an optimization problem using a JAX-compatible Optimistix solver.

    Supports both least-squares and general minimization solvers. Tracks
    iteration timing and loss history. Applies JIT compilation to improve
    performance.

    Parameters
    ----------
    theta_init : Any
        Initial guess for the parameters to optimize (PyTree).
    solver : `AbstractLeastSquaresSolver` or `AbstractMinimiser`
        Solver instance from the Optimistix library.
    args : PyTree[Any]
        Additional static/dynamic arguments passed to the loss function.
    loss_fn : Callable
        Objective function to be minimized. Must return a tuple: (loss, aux).
    max_iter : int
        Maximum number of iterations allowed before termination.

    Returns
    -------
    SolveResult
        Structured result containing final parameters, loss trace, timing,
        and convergence information.

    """
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

        scalar_loss = aux[0]
        jax.debug.print("    Iter {0} | loss = {1:.4e}", iter_count + 1, scalar_loss)  # noqa: E501

        loss_history[iter_count] = scalar_loss
        iter_times[iter_count] = iter_end - iter_start

        converged = terminate(y=theta, state=state)[0]
        iter_count += 1

    wall_time = time.time() - start_time

    return SolveResult(theta, aux, loss_history[:iter_count], iter_count,
                       iter_times[:iter_count], converged, wall_time)
