"""Nonparametric BLA, parametric subspace identification, and optimizer."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from freq_statespace import _misc
from freq_statespace._config import SOLVER
from freq_statespace._data_manager import (
    FrequencyData,
    InputOutputData,
    NonparametricBLA,
)
from freq_statespace._frequency_response import compute_frequency_response
from freq_statespace._model_structures import ModelBLA
from freq_statespace._solve import solve
from freq_statespace.dep import fsid


MAX_ITER = 1000


def _compute_weighted_residual(theta_dyn: ModelBLA, args: tuple) -> tuple:
    """Compute the weighted residuals between the nonparametric and parametric BLA."""
    theta_static, G_nonpar, f_data, W = args

    theta = eqx.combine(theta_dyn, theta_static)

    G_par = theta._frequency_response(f_data)
    loss = jnp.sqrt(W / G_nonpar.size) * (G_par - G_nonpar)
    loss_real = (loss.real, loss.imag)
    scalar_loss = jnp.sum(jnp.abs(loss) ** 2)

    return loss_real, (scalar_loss,)


def _normalize_states(model: ModelBLA, freq: FrequencyData) -> ModelBLA:
    """Normalize BLA model states to have unit variance."""
    nx, nu = model.B_u.shape

    f_data = freq.f

    G_xu = ModelBLA(  # note that this is not really the BLA
        A=model.A,
        B_u=model.B_u,  # usual state dynamics
        C_y=np.eye(nx),
        D_yu=np.zeros((nx, nu)),  # full-state output
        ts=model.ts,
        norm=model.norm,
    )._frequency_response(f_data)

    X = G_xu @ freq.U
    x = np.fft.irfft(X, axis=0)
    x_std = np.std(x, axis=(0, 2))

    Tx = np.diag(x_std)
    Tx_inv = np.diag(1 / x_std)

    # Apply similarity transformation: x_norm = Tx_inv * x
    return ModelBLA(
        A=Tx_inv @ model.A @ Tx,
        B_u=Tx_inv @ model.B_u,
        C_y=model.C_y @ Tx,
        D_yu=model.D_yu,
        ts=model.ts,
        norm=model.norm,
    )


def compute_nonparametric(U: np.ndarray, Y: np.ndarray) -> NonparametricBLA:
    """Compute nonparametric BLA and variance estimates from input-output data.

    Parameters
    ----------
    U : np.ndarray, shape (F, nu, R, P)
        DFT input spectrum at the excited frequencies across realizations and
        periods.
    Y : np.ndarray, shape (F, ny, R, P)
        DFT output spectrum at the excited frequencies across realizations and
        periods.

    Returns
    -------
    `NonparametricBLA`
        Nonparametric BLA estimate with frequency response and variance
        estimates.

    """
    G = compute_frequency_response(U, Y)

    M, P = G.shape[3:5]

    # Compute noise variance
    G_P = G.mean(axis=4)  # shape (F, ny, nu, M)
    if P > 1:
        sqr_error = np.abs(G - G_P[..., None]) ** 2  # shape (F, ny, nu, M, P)
        tot_sqr_error = sqr_error.sum(axis=(3, 4))  # shape (F, ny, nu)
        var_noise = tot_sqr_error / (M * (P - 1))  # shape (F, ny, nu)
        var_noise = jnp.asarray(var_noise)
    else:
        var_noise = None

    # Compute total variance
    G_bla = G_P.mean(axis=3)  # shape (F, ny, nu)
    if M > 1:
        sqr_error = np.abs(G_P - G_bla[..., None]) ** 2  # shape (F, ny, nu, M)
        tot_sqr_error = sqr_error.sum(axis=3)  # shape (F, ny, nu)
        var_tot = tot_sqr_error / (M - 1)  # shape (F, ny, nu)
        var_tot = jnp.asarray(var_tot)
    else:
        var_tot = None

    G_bla = jnp.asarray(G_bla)

    return NonparametricBLA(G_bla, var_noise, var_tot)


def subspace_id(data: InputOutputData, nx: int, q: int) -> ModelBLA:
    """Parametrize a state-space model using the frequency-domain subspace method.

    Parameters
    ----------
    data : `InputOutputData`
        Estimation data.
    nx : int
        State dimension of the system to be identified.
    q : int
        Subspace dimensioning parameter, must be greater than `nx`.

    Returns
    -------
    `ModelBLA`
        Estimated state-space model in BLA form.

    Raises
    ------
    ValueError
        If `q `is not greater than `nx`.

    """
    header = " Frequency-domain subspace identification "
    print(f"{header:=^72}")
    if q <= nx:
        raise ValueError(
            f"Subspace dimension q={q} must be greater than state dimension nx={nx}."
        )

    freq = data.freq
    f_data = freq.f[freq.f_idx]
    fs = freq.fs
    ts = 1 / fs
    z = 2 * np.pi * f_data / fs

    G_bla = freq.G_bla
    F, ny, nu = G_bla.G.shape

    # Convert BLA to input-output form for FSID algorithm compatibility
    Y = np.transpose(G_bla.G, (0, 2, 1)).reshape(nu * F, ny)
    U = np.tile(np.eye(nu), (F, 1))
    zj = np.repeat(np.exp(z * 1j), nu)

    # Create weighting matrix (inverse of total variance)
    if G_bla.var_tot is not None:
        W_temp = 1 / G_bla.var_tot

        # The steps below are to make it compatible with fsid.gfdsid
        W_temp = np.transpose(np.sqrt(W_temp), (0, 2, 1)).reshape(nu * F, ny)
        W = np.zeros((nu * F, ny, ny))
        for k in range(nu * F):
            np.fill_diagonal(W[k], W_temp[k])
    else:
        W = np.empty(0)

    # Ensure that zj, Y, and U are numpy arrays
    zj = np.asarray(zj, dtype=np.complex128)
    Y = np.asarray(Y, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)

    # Perform frequency subspace identification
    fddata = (zj, Y, U)
    A, B_u, C_y, D_yu = fsid.gfdsid(fddata=fddata, n=nx, q=q, estTrans=False, w=W)[:4]

    model = ModelBLA(A, B_u, C_y, D_yu, ts, data.norm)

    _misc.evaluate_model_performance(model, data)

    return model


def optimize(
    model: ModelBLA,
    data: InputOutputData,
    *,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser = SOLVER,
    max_iter: int = MAX_ITER,
) -> ModelBLA:
    """Refine the parameters of the BLA using frequency-response computations.

    Parameters
    ----------
    model : `ModelBLA`
        Initial BLA model to be optimized.
    data : `InputOutputData`
        Estimation data.
    solver : `optx.AbstractLeastSquaresSolver` or `optx.AbstractMinimiser`
        Any least-squares solver or general minimization solver from the
        Optimistix or Optax libraries.
    max_iter : int
        Maximum number of optimization iterations.

    Returns
    -------
    `ModelBLA`
        BLA model with optimized parameters.

    """
    header = " BLA optimization  "
    print(f"{header:=^72}")

    freq = data.freq
    G_bla = freq.G_bla
    f_data = freq.f[freq.f_idx]

    # Create weighting matrix (inverse of total variance)
    if G_bla.var_tot is not None:
        W = 1 / G_bla.var_tot
    else:
        W = jnp.ones_like(G_bla.G)

    model = _normalize_states(model, freq)
    theta0_dyn, theta_static = eqx.partition(model, eqx.is_inexact_array)

    args = (theta_static, jnp.asarray(G_bla.G), f_data, W)

    # Optimize the model parameters
    print("Starting iterative optimization...")
    solve_result = solve(theta0_dyn, solver, args, _compute_weighted_residual, max_iter)

    model = eqx.combine(solve_result.theta, theta_static)
    model = _normalize_states(model, freq)

    _misc.evaluate_model_performance(model, data, solve_result=solve_result)

    return model
