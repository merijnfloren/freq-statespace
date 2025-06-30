"""
Optimization of fully initialized NL-LFR models using forward simulations.
"""
from typing import NamedTuple, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from src.data_manager import InputOutputData
from src._model_structures import ModelNonlinearLFR
from src._solve import solve


class _OptiArgs(NamedTuple):
    """
    Static arguments for time-domain simulation and frequency-domain loss.

    Attributes
    ----------
    theta_static : ModelNonlinearLFR
        Static portion of the model (non-optimized parameters).
    u : jnp.ndarray
        Time-domain input signal, averaged over periods, shape (N, nu, R).
    Y : jnp.ndarray
        Frequency-domain output signal, averaged over periods,
        shape (F, ny, R).
    Lambda : jnp.ndarray
        Inverse noise variance matrices for weighting, shape (F, ny, ny).
    handicap : int
        Number of initial samples to discard from simulation to mitigate
        transients.
    """
    theta_static: ModelNonlinearLFR
    u: jnp.ndarray
    Y: jnp.ndarray
    Lambda: jnp.ndarray
    handicap: int


def run(
   io_data: InputOutputData,
   model_init: ModelNonlinearLFR,
   solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractMinimiser],
   handicap: Optional[int] = 1000,
   max_iter: int = 1000,
) -> ModelNonlinearLFR:
    """
    Optimize a nonlinear model by forward simulations.

    Parameters
    ----------
    io_data : InputOutputData
        Measured input-output data in time and frequency domains.
    model_init : ModelNonlinearLFR
        Initial model to optimize.
    solver : optx.AbstractLeastSquaresSolver or optx.AbstractMinimiser
        Optimizer used to minimize the loss.
    handicap : Optional[int]
        Number of initial samples to discard in simulation
        (default: 25% of data if None).
    max_iter : int
        Maximum number of optimization iterations.

    Returns
    -------
    ModelNonlinearLFR
        Optimized NL-LFR model.
    """

    if handicap is None:  # we start 25% 'ahead of time'
        handicap = int(np.ceil(0.25 * io_data.time.u.shape[0]))

    theta0, args = _prepare_problem(
        io_data, model_init, handicap
    )

    # Optimize the model parameters
    print('Starting iterative optimization...')
    solve_result = solve(theta0, solver, args, _loss_fn, max_iter)
    print('\n')

    return eqx.combine(solve_result.theta, args.theta_static)


def _loss_fn(theta: ModelNonlinearLFR, args: _OptiArgs) -> tuple:
    """
    Loss function comparing simulated model output to measured data in the
    frequency domain.

    Parameters
    ----------
    theta : ModelNonlinearLFR
        Model parameters being optimized (dynamic portion).
    args : _OptiArgs
        Static arguments for simulation and loss computation.

    Returns
    -------
    tuple
        - A tuple of real and imaginary parts of the residual.
        - A tuple containing the scalar mean squared error loss.
    """

    N = args.u.shape[0]
    R = args.u.shape[2]

    theta = eqx.combine(theta, args.theta_static)

    y_hat = theta.simulate(args.u, handicap=args.handicap)[0]
    Y_hat = jnp.fft.rfft(y_hat, axis=0)

    # --- Loss computation ---
    Y_loss = jnp.sqrt(args.Lambda / (R * N)) @ (args.Y - Y_hat)

    loss = (Y_loss.real, Y_loss.imag)

    MSE_loss = jnp.sum(jnp.abs(Y_loss)**2)
    return loss, (MSE_loss,)


def _prepare_problem(
    io_data: InputOutputData,
    model_init: ModelNonlinearLFR,
    handicap: int
) -> tuple[ModelNonlinearLFR, dict]:
    """
    Prepare the loss function initial value and arguments.

    Parameters
    ----------
    io_data : InputOutputData
        Measured data to use for training.
    model_init : ModelNonlinearLFR
        Initial model guess for parameter optimization.
    handicap : int
        Number of samples to discard from the beginning of simulation.

    Returns
    -------
    tuple
        - theta0 : ModelNonlinearLFR
            Partitioned dynamic parameters to be optimized.
        - args : _OptiArgs
            Static arguments for loss computation.
    """

    F, ny, R, P = io_data.freq.U.shape

    theta0, theta_static = eqx.partition(model_init, eqx.is_inexact_array)

    u = io_data.time.u
    Y = io_data.freq.Y

    u_bar = jnp.mean(u, axis=-1)
    Y_bar = jnp.mean(Y, axis=-1)

    # Compute weighting matrix Lambda
    Lambda = np.zeros((F, ny, ny))

    Y = io_data.freq.Y
    Y_P = Y.mean(axis=3)
    if P > 1:
        var_noise = ((np.abs(Y - Y_P[..., None])**2).sum(axis=(2, 3))
                     / R / (P - 1))
        for k in range(F):
            np.fill_diagonal(Lambda[k], 1 / var_noise[k])
    else:
        var_noise = None
        for k in range(F):
            np.fill_diagonal(Lambda[k], np.eye(ny))

    return theta0, _OptiArgs(
        theta_static=theta_static,
        u=u_bar,
        Y=Y_bar,
        Lambda=Lambda,
        handicap=handicap
    )
