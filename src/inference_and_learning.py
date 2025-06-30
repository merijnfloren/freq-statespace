"""
NL-LFR initialization using inference and learning.
"""

from typing import NamedTuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from src.data_manager import InputOutputData
from src.basis_functions import AbstractBasisFunction
from src.nonlinear_functions import create_custom_basis_function_model
from src._model_structures import ModelBLA, ModelNonlinearLFR
from src._solve import solve


class _ThetaWZ(eqx.Module):
    """
    Decision variables for inference and learning.

    Attributes
    ----------
    B_w_star : jnp.ndarray, shape (nx, nw)
    C_z_star : jnp.ndarray, shape (nz, nx)
    D_yw_star : jnp.ndarray, shape (ny, nw).
    D_zu_star : jnp.ndarray, shape (nz, nu).
    """
    B_w_star: jnp.ndarray = eqx.field(converter=jnp.asarray)
    C_z_star: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_yw_star: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_zu_star: jnp.ndarray = eqx.field(converter=jnp.asarray)


class _OptiArgs(NamedTuple):
    """
    Static arguments used for optimization.

    Attributes
    ----------
    theta_uy : tuple
        Tuple of linear model matrices (A, B_u, C_y).
    phi : AbstractBasisFunction
        Nonlinear basis function model.
    lambda_w : jnp.ndarray
        Regularization weight that controls latent signal variance.
    fixed_point_iters : int
        Number of fixed-point iterations to run.
    f_data : tuple
        Tuple containing (frequencies, sampling frequency, U, Y, G_yu).
    Lambda : jnp.ndarray
        Inverse noise variance weighting matrices, shape (F, ny, ny).
    Tz_inv : jnp.ndarray
        Inverse normalization scaling for latent signal z, shape (nz, nz).
    epsilon : jnp.ndarray
        Regularization parameter to ensure numerical stability.
    N : int
        Number of time samples per realization.
    """
    theta_uy: tuple
    phi: AbstractBasisFunction
    lambda_w: jnp.ndarray
    fixed_point_iters: int
    f_data: tuple
    Lambda: jnp.ndarray
    Tz_inv: jnp.ndarray
    epsilon: jnp.ndarray
    N: int


def run(
   io_data: InputOutputData,
   bla: ModelBLA,
   phi: AbstractBasisFunction,
   nw: int,
   lambda_w: float,
   fixed_point_iters: int,
   solver: Union[optx.AbstractLeastSquaresSolver, optx.AbstractMinimiser],
   max_iter: int = 1000,
   seed: int = 42,
   epsilon: float = 1e-8
) -> ModelNonlinearLFR:
    """
    Run inference and learning.

    Parameters
    ----------
    io_data : InputOutputData
        Measured input-output data in time and frequency domains.
    bla : ModelBLA
        Initial linear model used to define A, B_u, C_y, D_yu matrices.
    phi : AbstractBasisFunction
        Basis function model for static nonlinearity.
    nw : int
        Dimension of the latent disturbance signal w.
    lambda_w : float
        Regularization weight that controls latent signal variance.
    fixed_point_iters : int
        Number of fixed-point iterations for z-w consistency.
    solver : optx.AbstractLeastSquaresSolver or optx.AbstractMinimiser
        Optimizer to minimize the loss function.
    max_iter : int
        Maximum number of optimization iterations.
    seed : int
        PRNG seed for parameter initialization.
    epsilon : float
        Numerical regularization constant for matrix inversion.

    Returns
    -------
    ModelNonlinearLFR
        Fully initialized NL-LFR model.
    """
    theta0, args = _prepare_problem(
        io_data, bla, phi, nw, lambda_w, fixed_point_iters, seed, epsilon
    )

    # Optimize the model parameters
    print('Starting iterative optimization...')
    solve_result = solve(theta0, solver, args, _loss_fn, max_iter)
    print('\n')

    theta_opt = solve_result.theta
    aux = solve_result.aux

    beta = aux[-1]

    return ModelNonlinearLFR(
        A=args.theta_uy[0],
        B_u=args.theta_uy[1],
        C_y=args.theta_uy[2],
        D_yu=bla.D_yu,
        B_w=theta_opt.B_w_star,
        C_z=args.Tz_inv @ theta_opt.C_z_star,
        D_yw=theta_opt.D_yw_star,
        D_zu=args.Tz_inv @ theta_opt.D_zu_star,
        f_static=create_custom_basis_function_model(
            nw, phi, beta
        ),
        ts=io_data.time.ts
    )


def _loss_fn(theta: _ThetaWZ, args: _OptiArgs) -> tuple:
    """
    Loss function for inference + learning in nonlinear latent models.

    Parameters
    ----------
    theta : _ThetaWZ
        Decision variables.
    args : _OptiArgs
        Static arguments for inference and learning.

    Returns
    -------
    tuple
        - A tuple of real and imaginary parts of the residual.
        - A tuple (MSE loss, beta_hat), where beta_hat are the learned
          nonlinear parameters.
    """
    f_full, fs, U, Y, G_yu = args.f_data

    A = args.theta_uy[0]
    B_u = args.theta_uy[1]
    C_y = args.theta_uy[2].astype(complex)

    B_w = theta.B_w_star.astype(complex)
    C_z = (args.Tz_inv @ theta.C_z_star).astype(complex)
    D_yw = theta.D_yw_star
    D_zu = args.Tz_inv @ theta.D_zu_star

    nw = D_yw.shape[1]
    nz, nu = D_zu.shape
    F = U.shape[0]
    R = U.shape[2]

    Theta = jnp.vstack((B_w, D_yw)).T @ jnp.vstack((B_w, D_yw))
    Theta += args.epsilon / args.lambda_w * jnp.eye(nw)

    z = 2 * jnp.pi * f_full / fs
    zj = jnp.exp(z * 1j)

    I_nx = jnp.eye(A.shape[0])

    def _compute_parametric_Gs(k):
        G_x = jnp.linalg.solve(zj[k] * I_nx - A, jnp.hstack((B_u, B_w)))
        return (
            C_y @ G_x[:, nu:] + D_yw,  # G_yw
            C_z @ G_x[:, :nu] + D_zu,  # G_zu
            C_z @ G_x[:, nu:]  # G_zw
        )

    G_yw, G_zu, G_zw = jax.vmap(_compute_parametric_Gs)(jnp.arange(F))

    # --- Nonparametric inference ---
    def _infer_nonparametric_signals(k):
        Psi = G_yw[k, ...].T @ args.Lambda[k, ...]
        Phi = Psi @ G_yw[k, ...] + args.lambda_w * Theta
        W_hat = jnp.linalg.solve(
            Phi,
            Psi @ (Y[k, ...] - G_yu[k, ...] @ U[k, ...])
        )
        Z_hat = G_zu[k, ...] @ U[k, ...] + G_zw[k, ...] @ W_hat
        Y_hat = G_yu[k, ...] @ U[k, ...] + G_yw[k, ...] @ W_hat
        return W_hat, Z_hat, Y_hat

    W_star, Z_star, Y_hat = jax.vmap(_infer_nonparametric_signals)(jnp.arange(F))  # noqa: E501

    # --- Parametric learning ---
    w_star = jnp.fft.irfft(W_star, n=args.N, axis=0)
    z_star = jnp.fft.irfft(Z_star, n=args.N, axis=0)

    w_star_stacked = jnp.transpose(w_star, (2, 0, 1)).reshape(args.N * R, nw)
    z_star_stacked = jnp.transpose(z_star, (2, 0, 1)).reshape(args.N * R, nz)

    phi_z_star = args.phi.compute_features(z_star_stacked)
    beta_hat = jnp.linalg.solve(
        phi_z_star.T @ phi_z_star,
        phi_z_star.T @ w_star_stacked
    )

    # --- Fixed-point iterations ---
    def _fixed_point_iteration(_, phi_z):
        w_stacked = phi_z @ beta_hat
        w = jnp.transpose(w_stacked.reshape(R, args.N, nw), (1, 2, 0))
        W = jnp.fft.rfft(w, axis=0)
        Z = G_zu @ U + G_zw @ W
        z = jnp.fft.irfft(Z, n=args.N, axis=0)
        z_stacked = jnp.transpose(z, (2, 0, 1)).reshape(args.N * R, nz)
        return args.phi.compute_features(z_stacked)

    phi_z = jax.lax.fori_loop(
        0, args.fixed_point_iters, _fixed_point_iteration, phi_z_star, unroll=True  # noqa: E501
    )

    w_hat_stacked = phi_z @ beta_hat
    w_hat = jnp.transpose(w_hat_stacked.reshape(R, args.N, nw), (1, 2, 0))
    W_beta = jnp.fft.rfft(w_hat, axis=0)

    # --- Loss computation ---
    Y_hat = G_yu @ U + G_yw @ W_beta
    Y_loss = jnp.sqrt(args.Lambda / (R * args.N)) @ (Y - Y_hat)

    loss = (Y_loss.real, Y_loss.imag)

    MSE_loss = jnp.sum(jnp.abs(Y_loss)**2)
    return loss, (MSE_loss, beta_hat)


def _prepare_problem(
    io_data: InputOutputData,
    bla: ModelBLA,
    phi: AbstractBasisFunction,
    nw: int,
    lambda_w: float,
    fixed_point_iters: int,
    seed: int,
    epsilon: float
) -> tuple[ModelNonlinearLFR, dict]:
    """
    Prepare the problem data structures for inference and learning.

    Parameters
    ----------
    io_data : InputOutputData
        Combined time- and frequency-domain input-output data.
    bla : ModelBLA
        Initial linear model used for structure and response generation.
    phi : AbstractBasisFunction
        Basis function structure for nonlinear modeling.
    nw : int
        Dimension of disturbance signal.
    lambda_w : float
        Regularization on latent signal variance.
    fixed_point_iters : int
        Number of fixed-point iterations for z-w consistency.
    seed : int
        PRNG seed for reproducible parameter initialization.
    epsilon : float
        Regularization constant for numerical inversion.

    Returns
    -------
    tuple
        - _ThetaWZ: Initialized parameter module.
        - _OptiArgs: Static arguments used throughout optimization.
    """
    nz = phi.nz
    ny, nx = bla.C_y.shape
    N, nu, R, P = io_data.time.u.shape
    F = io_data.freq.U.shape[0]

    u = io_data.time.u.mean(axis=-1)

    # Initialize theta_wz
    key = jax.random.PRNGKey(seed)
    key_B_w, key_C_z, key_D_yw, key_D_zu = jax.random.split(key, 4)

    B_w_star = jax.random.normal(key_B_w, (nx, nw))
    C_z_star = jax.random.normal(key_C_z, (nz, nx))
    D_zu_star = jax.random.normal(key_D_zu, (nz, nu))
    D_yw_star = jax.random.normal(key_D_yw, (ny, nw))

    theta_wz = _ThetaWZ(B_w_star, C_z_star, D_yw_star, D_zu_star)
    theta_uy = (jnp.asarray(bla.A), jnp.asarray(bla.B_u), jnp.asarray(bla.C_y))

    # Compute z_star normalization
    beta_dummy = np.zeros((phi.num_features(), nw))
    f_static_dummy = create_custom_basis_function_model(
        nw, phi, beta_dummy
    )
    nonlin_lfr_dummy = ModelNonlinearLFR(
        A=bla.A,
        B_u=bla.B_u,
        C_y=bla.C_y,
        D_yu=bla.D_yu,
        B_w=np.zeros_like(B_w_star),
        C_z=C_z_star,
        D_yw=np.zeros_like(D_yw_star),
        D_zu=D_zu_star,
        f_static=f_static_dummy,
        ts=io_data.time.ts
    )
    handicap = int(np.ceil(0.25 * N))
    z_star = nonlin_lfr_dummy.simulate(u, handicap=handicap)[-1]
    z_star_min, z_star_max = z_star.min(axis=(0, 2)), z_star.max(axis=(0, 2))
    T_z_inv = jnp.diag(2 / (z_star_max - z_star_min))

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

    U_bar = jnp.asarray(io_data.freq.U.mean(axis=3))
    Y_bar = jnp.asarray(io_data.freq.Y.mean(axis=3))
    f_full = io_data.freq.f
    G_yu = bla.frequency_response(f_full)
    f_data = (f_full, 1 / io_data.time.ts, U_bar, Y_bar, G_yu)

    return theta_wz, _OptiArgs(
        theta_uy=theta_uy,
        phi=phi,
        lambda_w=jnp.asarray(lambda_w),
        fixed_point_iters=fixed_point_iters,
        f_data=f_data,
        Lambda=jnp.asarray(Lambda),
        Tz_inv=T_z_inv,
        epsilon=jnp.asarray(epsilon),
        N=N
    )
