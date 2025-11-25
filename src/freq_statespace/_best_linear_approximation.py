"""Nonparametric BLA, parametric subspace identification, and optimizer."""
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from . import _misc
from ._config import PRINT_EVERY, SOLVER, DeviceLike
from ._data_manager import (
    FrequencyData,
    InputOutputData,
    NonparametricBLA,
)
from ._frequency_response import compute_frequency_response
from ._model_structures import ModelBLA
from ._solve import solve
from .dep import fsid


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

    G_xu = ModelBLA(  # parametric u->x frequency response; not the true BLA
        A=model.A, B_u=model.B_u, C_y=np.eye(nx), D_yu=np.zeros((nx, nu)), 
        ts=model.ts, norm=model.norm,
    )._frequency_response(f_data)

    X = G_xu @ freq.U
    x = np.fft.irfft(X, axis=0)
    x_std = np.std(x, axis=(0, 2))

    Tx = np.diag(x_std)
    Tx_inv = np.diag(1 / x_std)

    # Apply similarity transformation: x_norm = Tx_inv * x
    return ModelBLA(
        A=Tx_inv @ model.A @ Tx, B_u=Tx_inv @ model.B_u,
        C_y=model.C_y @ Tx, D_yu=model.D_yu,
        ts=model.ts, norm=model.norm,
    )
    
    
def _validate_weighting(
    freq_weighting: bool,
    var_tot: jnp.ndarray | None,
    print_warning: bool
) -> bool:
    """Check if weighting can be applied based on BLA total variance availability."""
    if freq_weighting and var_tot is None:
        if print_warning:
            print(
                "Warning: Frequency weighting based on BLA total variance requested, "
                "but such estimate is not available. Proceeding without weighting."
            )
        freq_weighting = False
    return freq_weighting


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


def subspace_id(
    data: InputOutputData,
    nx: int,
    nq: int | None = None,
    freq_weighting: bool = True,
    logging_enabled: bool = True
) -> ModelBLA:
    """Parametrize a state-space model using the frequency-domain subspace method.

    Parameters
    ----------
    data : `InputOutputData`
        Estimation data.
    nx : int
        State dimension of the system to be identified.
    nq : int | None, optional
        Subspace dimensioning parameter, must be greater than `nx`. Defaults to
        `nx + 1` if not provided.
    freq_weighting : bool
        Whether to use frequency weighting based on the inverse of the total variance
        on the nonparametric BLA. Defaults to `True`.
    logging_enabled : bool
        Whether to print a summary of the identification results. Defaults to `True`.

    Returns
    -------
    `ModelBLA`
        Estimated state-space model in BLA form.

    Raises
    ------
    ValueError
        If `nq` is not greater than `nx`.

    """
    if logging_enabled:
        header = " Frequency-domain subspace identification "
        print(f"{header:=^72}")
    
    nq = nx + 1 if nq is None else nq
    if nq <= nx:
        raise ValueError(
            f"Subspace dimension nq={nq} must be greater than state dimension nx={nx}."
        )

    freq = data.freq
    f_data = freq.f[freq.f_idx]
    fs = freq.fs
    z = 2 * np.pi * f_data / fs

    G_bla = freq.G_bla
    F, ny, nu = G_bla.G.shape
    
    freq_weighting = _validate_weighting(
        freq_weighting, G_bla.var_tot, logging_enabled
    )

    # Convert BLA to input-output form for FSID algorithm compatibility
    Y = np.transpose(G_bla.G, (0, 2, 1)).reshape(nu * F, ny)
    U = np.tile(np.eye(nu), (F, 1))
    zj = np.repeat(np.exp(z * 1j), nu)

    # Create weighting matrix (inverse of total variance)
    if freq_weighting:
        W_temp = 1 / G_bla.var_tot

        # The four lines below are to ensure compatibility with fsid.gfdsid
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

    # Perform frequency-domain subspace identification
    fddata = (zj, Y, U)
    A, B_u, C_y, D_yu = fsid.gfdsid(fddata=fddata, n=nx, q=nq, estTrans=False, w=W)[:4]

    model = ModelBLA(A, B_u, C_y, D_yu, 1 / fs, data.norm)

    if logging_enabled:
        _misc.evaluate_model_performance(model, data)

    return model


def optimize(
    model: ModelBLA,
    data: InputOutputData,
    *,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser = SOLVER,
    freq_weighting: bool = True,
    max_iter: int = MAX_ITER,
    print_every: int = PRINT_EVERY,
    device: DeviceLike = None,
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
        Optimistix or Optax libraries. Defaults to `SOLVER`.
    freq_weighting : bool
        Whether to use frequency weighting based on the inverse of the total variance
        on the nonparametric BLA. Defaults to `True`.
    max_iter : int
        Maximum number of optimization iterations. Defaults to `MAX_ITER`.
    print_every : int
        Frequency of printing iteration information. If set to `0`, only a
        summary is printed. If set to `-1`, no printing is done. Defaults to
        `PRINT_EVERY`.
    device : `DeviceLike`, optional
        Device on which to perform the computations. Can be either a device
        name (`"cpu"`, `"gpu"`, or `"tpu"`) or a specific JAX device. If not
        provided, the default JAX device is used.
        
    Returns
    -------
    `ModelBLA`
        BLA model with optimized parameters.

    """
    logging_enabled = print_every != -1
    
    if logging_enabled:
        header = " BLA optimization  "
        print(f"{header:=^72}")

    freq = data.freq
    G_bla = freq.G_bla
    f_data = freq.f[freq.f_idx]
    
    print_warning = print_every != -1
    freq_weighting = _validate_weighting(
        freq_weighting, G_bla.var_tot, print_warning
    )

    # Create weighting matrix (inverse of total variance)
    if freq_weighting:
        W = 1 / G_bla.var_tot
    else:
        W = jnp.ones_like(G_bla.G)

    model = _normalize_states(model, freq)
    theta0_dyn, theta_static = eqx.partition(model, eqx.is_inexact_array)

    args =(theta_static, jnp.asarray(G_bla.G), f_data, W)

    # Optimize the model parameters
    if logging_enabled:
        print("Starting iterative optimization...")
    solve_result = solve(
        theta0_dyn, solver, args, _compute_weighted_residual,
        max_iter, print_every, device
    )

    model = eqx.combine(solve_result.theta, theta_static)
    model = _normalize_states(model, freq)

    if logging_enabled:
        _misc.evaluate_model_performance(model, data, solve_result=solve_result)

    return model
