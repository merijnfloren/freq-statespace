"""
Frequency-domain system identification using nonparametric BLA and parametric
subspace and optimization-based approaches.

This module provides:
- Estimation of nonparametric Best Linear Approximations (BLA) with variance
  metrics.
- Subspace-based frequency-domain identification.
- Iterative refinement via nonlinear least-squares optimization.
- Visualization tools for BLA magnitude and distortion levels.
"""

from dataclasses import dataclass
from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx

from dep import fsid
from src.frequency_response import compute_frequency_response, FrequencyResponse  # noqa: E501
from src.data_manager import FrequencyData, InputOutputData
from src._model_structures import ModelBLA
from src._solve import solve


@dataclass(frozen=True)
class NonparametricBLA:
    """
    Nonparametric Best Linear Approximation (BLA) and associated metadata.

    Attributes
    ----------
    G : np.ndarray
        Nonparametric BLA estimate, shape (F, ny, nu).
    freq_resp : FrequencyResponse
        Raw frequency response matrix and metadata, shape (F, ny, nu, M, P).
    var_noise : Optional[np.ndarray]
        Estimated noise variance across realizations, shape (F, ny, nu).
    var_tot : Optional[np.ndarray]
        Estimated total variance across periods, shape (F, ny, nu).
    """
    G: np.ndarray
    freq_resp: FrequencyResponse
    var_noise: Optional[np.ndarray]
    var_tot: Optional[np.ndarray]

    def plot(self) -> None:
        """
        Plot the BLA magnitude and distortion levels (if available).

        Returns
        -------
        None
        """
        return _plot(self)


def compute_nonparametric(data: InputOutputData) -> NonparametricBLA:
    """
    Compute nonparametric BLA and variance estimates from input-output data.

    Parameters
    ----------
    data : InputOutputData
        Combined time- and frequency-domain data.

    Returns
    -------
    NonparametricBLA
        BLA result with noise and total variance estimates.
    """

    freq_resp = compute_frequency_response(data)
    G = freq_resp.G

    M, P = G.shape[3:5]

    # Average over periods and compute noise variance
    G_P = G.mean(axis=4)
    if P > 1:
        var_noise = (np.abs(G - G_P[..., None])**2).sum(axis=4) / P / (P - 1)
    else:
        var_noise = None

    # Average over realizations and compute total variance
    G_bla = G_P.mean(axis=3)
    if M > 1:
        var_tot = (np.abs(G_P - G_bla[..., None])**2).sum(axis=3) / M / (M - 1)
        if var_noise is not None:
            var_noise = var_noise.mean(axis=3) / M
    else:
        var_tot = None

    return NonparametricBLA(G_bla, freq_resp, var_noise, var_tot)


def freq_subspace_id(G_nonpar: NonparametricBLA, nx: int, q: int) -> ModelBLA:
    """
    Estimate a parametric state-space model using frequency-domain subspace ID.

    Parameters
    ----------
    G_nonpar : NonparametricBLA
        Nonparametric BLA with associated frequency and variance data.
    nx : int
        Desired model order (number of states).
    q : int
        Subspace dimension parameter. Must be greater than nx.

    Returns
    -------
    ModelBLA
        Parametrized state-space model identified from BLA data.
    """

    freq = G_nonpar.freq_resp.freq
    f_data = freq.f[freq.f_idx]
    fs = freq.fs
    ts = 1 / fs
    z = 2 * np.pi * f_data / fs

    F, ny, nu = G_nonpar.G.shape

    # Convert BLA to input-output form for FSID algorithm compatibility
    Y = np.transpose(G_nonpar.G, (0, 2, 1)).reshape(nu * F, ny)
    U = np.tile(np.eye(nu), (F, 1))
    zj = np.repeat(np.exp(z * 1j), nu)

    # Create weighting matrix (inverse of total variance)
    if G_nonpar.var_tot is not None:
        W_temp = 1 / G_nonpar.var_tot

        # The steps below are to make it compatible with fsid.gfdsid
        W_temp = np.transpose(np.sqrt(W_temp), (0, 2, 1)).reshape(nu * F, ny)
        W = np.zeros((nu * F, ny, ny))
        for k in range(nu * F):
            np.fill_diagonal(W[k], W_temp[k])
    else:
        W = np.empty(0)

    # Perform frequency subspace identification
    A, B_u, C_y, D_yu = fsid.gfdsid(fddata=(zj, Y, U), n=nx,
                                    q=q, estTrans=False, w=W)[:4]
    return ModelBLA(
        A=A,
        B_u=B_u,
        C_y=C_y,
        D_yu=D_yu,
        ts=ts
    )


def freq_iterative_optimization(
    G_nonpar: NonparametricBLA,
    G_par_init: ModelBLA,
    solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractMinimiser],
    max_iter: int
) -> ModelBLA:
    """
    Refine a parametric model by minimizing the discrepancy with the BLA.

    Parameters
    ----------
    G_nonpar : NonparametricBLA
        Nonparametric BLA to fit.
    G_par_init : ModelBLA
        Initial parametric model to refine.
    solver : Union[optx.AbstractLeastSquaresSolver, optx.AbstractMinimiser]
        Optimistix solver: either a least-squares one or a more
        general-purpose minimizer.
    max_iter : int
        Maximum number of optimization iterations.

    Returns
    -------
    ModelBLA
        Refined state-space model after optimization.
    """
    # Create weighting matrix (inverse of total variance)
    if G_nonpar.var_tot is not None:
        W = 1 / G_nonpar.var_tot
    else:
        W = jnp.ones_like(G_nonpar.G)

    freq = G_nonpar.freq_resp.freq
    f_data = freq.f[freq.f_idx]

    G_par_init = _normalize_states(G_par_init, freq)
    theta0_dyn, theta_static = eqx.partition(G_par_init, eqx.is_inexact_array)

    args = (theta_static, G_nonpar.G, f_data, W)

    # Optimize the model parameters
    print('Starting iterative optimization...')
    solve_result = solve(theta0_dyn, solver, args, _loss_fn, max_iter)
    print('\n')

    theta_opti = solve_result.theta
    G_par_opti = eqx.combine(theta_opti, theta_static)
    return _normalize_states(G_par_opti, freq)


def _loss_fn(theta_dyn: ModelBLA, args: tuple) -> tuple:
    """
    Compute the loss and residual for frequency-domain model optimization.

    Parameters
    ----------
    theta_dyn : ModelBLA
        Dynamic parameters to optimize.
    args : tuple
        Tuple containing (static parameters, BLA data,
                          frequency vector, weights).

    Returns
    -------
    tuple
        - A tuple of real and imaginary parts of the residual.
        - A tuple containing the scalar mean squared error loss.
    """
    theta_static, G_nonpar, f_data, W = args

    theta = eqx.combine(theta_dyn, theta_static)

    G_par = theta.frequency_response(f_data)
    G_loss = jnp.sqrt(W / G_nonpar.size) * (G_par - G_nonpar)
    loss = (G_loss.real, G_loss.imag)

    MSE_loss = jnp.sum(jnp.abs(G_loss)**2)
    return loss, (MSE_loss,)


def _normalize_states(model: ModelBLA, freq: FrequencyData) -> ModelBLA:
    """
    Normalize model states to have unit variance for better numerical
    conditioning.

    Parameters
    ----------
    model : ModelBLA
        Parametric model whose state basis should be normalized.
    freq : FrequencyData
        Frequency-domain input data for estimating state response.

    Returns
    -------
    ModelBLA
        Transformed model with normalized state coordinates.
    """
    nx, nu = model.B_u.shape

    f_data = freq.f

    G_xu = ModelBLA(
        A=model.A, B_u=model.B_u,  # usual state dynamics
        C_y=np.eye(nx), D_yu=np.zeros((nx, nu)),  # full-state output
        ts=model.ts
    ).frequency_response(f_data)

    U_mean = freq.U.mean(axis=-1)
    X = G_xu @ U_mean
    x = np.fft.irfft(X, axis=0)
    x_std = np.std(x, axis=(0, 2))

    Tx = np.diag(x_std)
    Tx_inv = np.diag(1 / x_std)

    # Apply similarity transformation: x_norm = Tx_inv * x
    return ModelBLA(
        A=Tx_inv @ model.A @ Tx, B_u=Tx_inv @ model.B_u,
        C_y=model.C_y @ Tx, D_yu=model.D_yu, ts=model.ts
    )


def _plot(bla: NonparametricBLA) -> None:
    """
    Plot the magnitude of the BLA along with estimated distortion levels.

    Parameters
    ----------
    bla : NonparametricBLA
        BLA object containing magnitude and variance estimates.

    Returns
    -------
    None
        This function only produces a matplotlib plot.
    """
    ny, nu = bla.G.shape[1:3]
    freq = bla.freq_resp.freq

    scale_matrix = (bla.freq_resp.norm.y_std[:, None]
                    / bla.freq_resp.norm.u_std[None, :])
    G = scale_matrix * bla.G

    # Calculate optimal figure size based on subplot grid
    fig_width = min(5 * nu, 15)  # max 15 inches wide
    fig_height = min(4 * ny, 12)  # max 12 inches tall
    _, axes = plt.subplots(ny, nu, figsize=(fig_width, fig_height))

    # Ensure axes is 2D for consistent indexing
    if ny == 1 and nu == 1:
        axes = np.array([[axes]])
    elif ny == 1 or nu == 1:
        axes = axes.reshape(ny, nu)

    f_data = freq.f[freq.f_idx]

    for i in range(ny):
        for j in range(nu):
            ax = axes[i, j]

            # Plot BLA magnitude
            mag_db = 20 * np.log10(np.abs(G[:, i, j]))
            ax.plot(f_data, mag_db, label='BLA magnitude')

            # Plot distortion levels if available
            if bla.var_noise is not None:
                std_noise = scale_matrix[i, j] * np.sqrt(bla.var_noise[:, i, j])  # noqa: E501
                noise_db = 20 * np.log10(std_noise)
                ax.plot(f_data, noise_db, label='Noise std')

            if bla.var_tot is not None:
                std_tot = scale_matrix[i, j] * np.sqrt(bla.var_tot[:, i, j])
                tot_db = 20 * np.log10(std_tot)
                ax.plot(f_data, tot_db, label='Total std')

            # Format subplot
            ax.set_title(f'G[{i+1},{j+1}]')
            ax.legend()
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Magnitude [dB]')

    plt.tight_layout()
    plt.show()
