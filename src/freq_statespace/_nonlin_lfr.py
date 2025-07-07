"""NL-LFR model inference and learning, and nonlinear optimization."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from freq_statespace import _misc
from freq_statespace._config import SEED, SOLVER
from freq_statespace._data_manager import FrequencyData, InputOutputData
from freq_statespace._model_structures import ModelBLA, ModelNonlinearLFR
from freq_statespace._solve import solve
from freq_statespace.f_static._feature_maps import AbstractFeatureMap
from freq_statespace.f_static._nonlin_funcs import (
    AbstractNonlinearFunction,
    BasisFunctionModel,
)


MAX_ITER_INFERENCE_AND_LEARNING = 1000
MAX_ITER_OPTIMIZATION = 100
EPSILON = 1e-10


class _ThetaWZ(eqx.Module):
    """Decision variables for inference and learning."""

    B_w_star: jnp.ndarray = eqx.field(converter=jnp.asarray)
    C_z_star: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_yw_star: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_zu_star: jnp.ndarray = eqx.field(converter=jnp.asarray)


class _ArgsInferenceLearning(NamedTuple):
    """Static arguments for inference and learning."""

    theta_uy: tuple  # (A, B_u, C_y)
    phi: AbstractFeatureMap
    lambda_w: float
    fixed_point_iters: int
    f_data: tuple  # (frequencies, sampling frequency, U, Y, G_yu)
    Lambda: jnp.ndarray  # shape (F, ny, ny)
    Tz_inv: jnp.ndarray  # shape (nz, nz)
    epsilon: float
    N: int


class _ArgsNonlinOptimization(NamedTuple):
    """Static arguments for nonlinear optimization."""

    theta_static: ModelNonlinearLFR
    u: jnp.ndarray  # shape (N, nu, R)
    Y: jnp.ndarray  # shape (N, ny, R)
    Lambda: jnp.ndarray  # shape (F, ny, ny)
    x0: jnp.ndarray  # shape (nx, R)
    offset: int


def _compute_weighting_matrix(freq: FrequencyData) -> jnp.ndarray:
    """Compute weighting matrix for the loss function."""
    F, ny = freq.Y.shape[:2]
    var_noise = freq.Y_var_noise

    # Each (ny x ny) matrix is diagonal with elements equal to the inverse of the noise
    # variance for each output at each frequency. If only one is available (`P==1`),
    # identity matrices are returned instead.
    Lambda = np.zeros((F, ny, ny))
    for k in range(F):
        diag = np.eye(ny) if var_noise is None else np.diag(1 / var_noise[k])
        Lambda[k, ...] = diag

    return jnp.asarray(Lambda)


def _compute_bla_loss(bla: ModelBLA, data: InputOutputData) -> float:
    """Compute the BLA performance on the loss function that is used in this module."""
    Lambda = _compute_weighting_matrix(data.freq)
    Y = data.freq.Y
    U = data.freq.U

    Y_hat = bla._frequency_response(data.freq.f) @ U
    loss = _compute_weighted_residual(Y, Y_hat, Lambda)
    loss_scalar = jnp.sum(jnp.abs(loss) ** 2)

    return loss_scalar


def _create_basis_function_model_given_beta(
    nw: int, phi: AbstractFeatureMap, beta: jnp.ndarray
) -> BasisFunctionModel:
    """Create a `BasisFunctionModel` instance given `beta`.

    When a user does not want to perform inference and learning, and instead
    wants to perform nonlinear optimization directly, a `BasisFunctionModel` is
    initialized randomly with a seed (specifying a custom `beta` is not
    supported). This (private!) function bypasses that behavior by replacing
    the random `beta` with the provided matrix obtained from inference and
    learning. A dummy seed is still required for initialization, but it does
    not affect the final result.
    """
    dummy_seed = 0

    return eqx.tree_at(
        where=lambda tree: tree.beta,
        pytree=BasisFunctionModel(nw, phi, dummy_seed),
        replace=beta,
    )


def _prepare_inference_and_learning(
    bla: ModelBLA,
    data: InputOutputData,
    phi: AbstractFeatureMap,
    nw: int,
    lambda_w: float,
    fixed_point_iters: int,
    seed: int,
    epsilon: float,
) -> tuple[ModelNonlinearLFR, _ArgsInferenceLearning]:
    """Prepare initial guess and function arguments for inference and learning."""
    nz = phi.nz
    ny, nx = bla.C_y.shape
    N, nu = data.time.u.shape[:2]

    f_full = data.freq.f
    U = data.freq.U
    Y = data.freq.Y

    # Initialize theta_wz
    key = jax.random.key(seed)
    key_B_w, key_C_z, key_D_yw, key_D_zu = jax.random.split(key, 4)

    B_w_star = jax.random.normal(key_B_w, (nx, nw))
    C_z_star = jax.random.normal(key_C_z, (nz, nx))
    D_zu_star = jax.random.normal(key_D_zu, (nz, nu))
    D_yw_star = jax.random.normal(key_D_yw, (ny, nw))

    theta_wz = _ThetaWZ(B_w_star, C_z_star, D_yw_star, D_zu_star)
    theta_uy = (bla.A, bla.B_u, bla.C_y)

    # Compute z_star normalization
    G_zu = ModelBLA(  # note that this is not really the BLA
        A=bla.A, B_u=bla.B_u, C_y=C_z_star, D_yu=D_zu_star, ts=bla.ts, norm=bla.norm
    )._frequency_response(f_full)

    Z_star = G_zu @ U
    z_star = np.fft.irfft(Z_star, axis=0)
    z_star_min, z_star_max = z_star.min(axis=(0, 2)), z_star.max(axis=(0, 2))
    Tz_inv = jnp.diag(2 / (z_star_max - z_star_min))

    G_yu = bla._frequency_response(f_full)
    f_data = (f_full, 1 / data.time.ts, U, Y, G_yu)

    args = _ArgsInferenceLearning(
        theta_uy=theta_uy,
        phi=phi,
        lambda_w=lambda_w,
        fixed_point_iters=fixed_point_iters,
        f_data=f_data,
        Lambda=_compute_weighting_matrix(data.freq),
        Tz_inv=Tz_inv,
        epsilon=epsilon,
        N=N,
    )

    return theta_wz, args


def _prepare_nonlin_optimization(
    data: InputOutputData, model_init: ModelNonlinearLFR, offset: int
) -> tuple[ModelNonlinearLFR, _ArgsNonlinOptimization]:
    """Prepare initial guess and function arguments for nonlinear optimization."""
    theta0, theta_static = eqx.partition(model_init, eqx.is_inexact_array)

    bla = super(ModelNonlinearLFR, model_init)

    # Select the initial state from steady-state BLA simulations
    x_bla = bla._simulate(data.time.u, offset=data.time.u.shape[0])[1]
    x0 = x_bla[-offset, :, :]

    args = _ArgsNonlinOptimization(
        theta_static=theta_static,
        u=data.time.u,
        Y=data.freq.Y,
        Lambda=_compute_weighting_matrix(data.freq),
        x0=x0,
        offset=offset,
    )

    return theta0, args


def _compute_weighted_residual(
    Y: jnp.ndarray, Y_hat: jnp.ndarray, Lambda: jnp.ndarray
) -> jnp.ndarray:
    """Compute the weighted residuals between the measured and predicted outputs.

    Parameters
    ----------
    Y : jnp.ndarray, shape (N, ny, R)
        Measured output spectrum, averaged over periods.
    Y_hat : jnp.ndarray, shape (N, ny, R)
        Simulated output spectrum.
    Lambda : jnp.ndarray, shape (F, ny, ny)
        Weight matrix.

    """
    N, _, R = Y.shape
    return jnp.sqrt(Lambda / (R * N)) @ (Y - Y_hat)


def _loss_inference_and_learning(
    theta: _ThetaWZ, args: _ArgsInferenceLearning
) -> tuple:
    """Implement the inference and learning method and compute the loss."""
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
            C_z @ G_x[:, nu:],  # G_zw
        )

    G_yw, G_zu, G_zw = jax.vmap(_compute_parametric_Gs)(jnp.arange(F))

    # 1) Nonparametric inference
    def _infer_nonparametric_signals(k):
        Psi = G_yw[k, ...].T @ args.Lambda[k, ...]
        Phi = Psi @ G_yw[k, ...] + args.lambda_w * Theta
        W_hat = jnp.linalg.solve(Phi, Psi @ (Y[k, ...] - G_yu[k, ...] @ U[k, ...]))
        Z_hat = G_zu[k, ...] @ U[k, ...] + G_zw[k, ...] @ W_hat
        Y_hat = G_yu[k, ...] @ U[k, ...] + G_yw[k, ...] @ W_hat
        return W_hat, Z_hat, Y_hat

    W_star, Z_star, Y_hat = jax.vmap(_infer_nonparametric_signals)(jnp.arange(F))  # noqa: E501

    # 2) Parametric learning
    w_star = jnp.fft.irfft(W_star, n=args.N, axis=0)
    z_star = jnp.fft.irfft(Z_star, n=args.N, axis=0)

    w_star_stacked = jnp.transpose(w_star, (2, 0, 1)).reshape(args.N * R, nw)
    z_star_stacked = jnp.transpose(z_star, (2, 0, 1)).reshape(args.N * R, nz)

    # 2a) Compute beta
    phi_z_star = args.phi._compute_features(z_star_stacked)
    beta_hat = jnp.linalg.solve(
        phi_z_star.T @ phi_z_star, phi_z_star.T @ w_star_stacked
    )

    # 2b) Perform fixed-point iterations
    def _fixed_point_iteration(_, phi_z):
        w_stacked = phi_z @ beta_hat
        w = jnp.transpose(w_stacked.reshape(R, args.N, nw), (1, 2, 0))
        W = jnp.fft.rfft(w, axis=0)
        Z = G_zu @ U + G_zw @ W
        z = jnp.fft.irfft(Z, n=args.N, axis=0)
        z_stacked = jnp.transpose(z, (2, 0, 1)).reshape(args.N * R, nz)
        return args.phi._compute_features(z_stacked)

    phi_z = jax.lax.fori_loop(
        0,
        args.fixed_point_iters,
        _fixed_point_iteration,
        phi_z_star,
        unroll=True,  # noqa: E501
    )

    w_hat_stacked = phi_z @ beta_hat
    w_hat = jnp.transpose(w_hat_stacked.reshape(R, args.N, nw), (1, 2, 0))
    W_beta = jnp.fft.rfft(w_hat, axis=0)

    # 2c) Compute loss
    Y_hat = G_yu @ U + G_yw @ W_beta
    loss = _compute_weighted_residual(Y, Y_hat, args.Lambda)
    loss_real = (loss.real, loss.imag)
    scalar_loss = jnp.sum(jnp.abs(loss) ** 2)

    return loss_real, (scalar_loss, beta_hat)


def _loss_nonlin_optimization(
    theta: ModelNonlinearLFR, args: _ArgsNonlinOptimization
) -> tuple:
    """Perform time-domain forward simulations and compute the loss."""
    theta = eqx.combine(theta, args.theta_static)

    # Simulate the model in time domain
    y_hat = theta._simulate(args.u, offset=args.offset, x0=args.x0)[0]
    Y_hat = jnp.fft.rfft(y_hat, axis=0)

    # Compute loss
    loss = _compute_weighted_residual(args.Y, Y_hat, args.Lambda)
    loss_real = (loss.real, loss.imag)
    scalar_loss = jnp.sum(jnp.abs(loss) ** 2)

    return loss_real, (scalar_loss,)


def inference_and_learning(
    bla: ModelBLA,
    data: InputOutputData,
    *,
    phi: AbstractFeatureMap,
    nw: int,
    lambda_w: float,
    fixed_point_iters: int,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser = SOLVER,
    max_iter: int = MAX_ITER_INFERENCE_AND_LEARNING,
    seed: int = SEED,
    epsilon: float = EPSILON,
) -> ModelNonlinearLFR:
    """Perform inference and learning.

    Parameters
    ----------
    bla : `ModelBLA`
        Parametric BLA model; is kept fixed during optimization.
    data : `InputOutputData`
        Measured input-output data object.
    phi : `AbstractFeatureMap`
        Any feature mapping that is linear in the parameters.
    nw : int
        Dimension of the latent signal `w`.
    lambda_w : float
        Regularization weight that controls the solution variance of `w`
        during inference.
    fixed_point_iters : int
        Number of fixed-point iterations for internal consistency.
    solver : `optx.AbstractLeastSquaresSolver` or `optx.AbstractMinimiser`
        Any least-squares solver or general minimization solver from the
        Optimistix or Optax libraries.
    max_iter : int
        Maximum number of optimization iterations.
    seed : int
        Random seed for parameter initialization.
    epsilon : float
        Numerical regularization constant for matrix inversion.

    Returns
    -------
    `ModelNonlinearLFR`
        Fully initialized NL-LFR model.

    """
    header = " NL-LFR inference and learning  "
    print(f"{header:=^72}")

    theta0, args = _prepare_inference_and_learning(
        bla, data, phi, nw, lambda_w, fixed_point_iters, seed, epsilon
    )
    bla_loss = _compute_bla_loss(bla, data)

    # Optimize the model parameters
    print("Starting iterative optimization...")
    print(f"    BLA loss: {bla_loss:.4e}")
    solve_result = solve(theta0, solver, args, _loss_inference_and_learning, max_iter)

    aux = solve_result.aux
    scalar_loss = aux[0]
    beta = aux[-1]

    model = ModelNonlinearLFR(
        A=args.theta_uy[0],
        B_u=args.theta_uy[1],
        C_y=args.theta_uy[2],
        D_yu=bla.D_yu,
        B_w=solve_result.theta.B_w_star,
        C_z=args.Tz_inv @ solve_result.theta.C_z_star,
        D_yw=solve_result.theta.D_yw_star,
        D_zu=args.Tz_inv @ solve_result.theta.D_zu_star,
        f_static=_create_basis_function_model_given_beta(nw, phi, beta),
        ts=data.time.ts,
        norm=bla.norm,
    )

    _misc.evaluate_model_performance(model, data, solve_result=solve_result)

    # Compute recursive loss of the NL-LFR model to check consistency with the
    # inference and learning solution. We simulate 2 periods and discard the
    # first one to ensure that transients have died out (see `offset`)
    y_hat = model._simulate(data.time.u, offset=data.time.u.shape[0])[0]
    Y_hat = jnp.fft.rfft(y_hat, axis=0)
    loss_recursive = _compute_weighted_residual(data.freq.Y, Y_hat, args.Lambda)
    scalar_loss_recursive = jnp.sum(jnp.abs(loss_recursive) ** 2)

    print("NL-LFR loss consistency check:")
    print(f"    Inference & learning loss = {scalar_loss:.4e}")
    print(f"    Recursive simulation loss = {scalar_loss_recursive:.4e}\n")

    return model


def optimize(
    model: ModelNonlinearLFR,
    data: InputOutputData,
    *,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser = SOLVER,
    max_iter: int = MAX_ITER_OPTIMIZATION,
    offset: int | None = None,
) -> ModelNonlinearLFR:
    """Refine the parameters of an NL-LFR model using time-domain simulations.

    Parameters
    ----------
    model : `ModelNonlinearLFR`
        Initial NL-LFR model to optimize.
    data : `InputOutputData`
        Measured input-output data object.
    solver : `optx.AbstractLeastSquaresSolver` or `optx.AbstractMinimiser`
        Any least-squares solver or general minimization solver from the
        Optimistix or Optax libraries.
    max_iter : int
        Maximum number of optimization iterations.
    offset : int, optional
        A non-negative integer `≤ N`. This value is used to select
        the unknown initial state of the time-domain simulations. Specifically,
        the initial state is selected from the steady-state BLA state
        simulations as `x0 = x_bla[-offset, :, :]`. The number of input samples
        are prepended accordingly. This warm-up ensures the system reaches
        steady-state before the main period begins, allowing for leakage-free
        DFT computations (the prepended samples are discarded). This approach
        is valid because the data is assumed to be periodic. If not specified,
        `offset` defaults to 10% of the data length.

    Returns
    -------
    `ModelNonlinearLFR`
        NL-LFR model with optimized parameters.

    """
    header = " NL-LFR optimization  "
    print(f"{header:=^72}")

    if offset is None:  # we start 10% 'ahead of time'
        offset = int(np.ceil(0.1 * data.time.u.shape[0]))

    theta0, args = _prepare_nonlin_optimization(data, model, offset)

    bla_loss = _compute_bla_loss(super(ModelNonlinearLFR, model), data)

    print("Starting iterative optimization...")
    print(f"    BLA loss: {bla_loss:.4e}")
    solve_result = solve(theta0, solver, args, _loss_nonlin_optimization, max_iter)

    model = eqx.combine(solve_result.theta, args.theta_static)

    _misc.evaluate_model_performance(
        model, data, solve_result=solve_result, offset=offset
    )

    return model


def construct(bla: ModelBLA, f_static: AbstractNonlinearFunction) -> ModelNonlinearLFR:
    """Construct an NL-LFR model given the BLA and a static nonlinear function object.

    Parameters
    ----------
    bla : `ModelBLA`
        Parametric BLA model.
    f_static : `AbstractNonlinearFunction`
        Nonlinear function object that defines the static relation between
        the latent signals `z` and `w`. Note that the `seed` attribute of
        `f_static` is used to randomly initialize the matrices `B_w`, `C_z`,
        `D_yw`, and `D_zu`.

    Returns
    -------
    `ModelNonlinearLFR`
        NL-LFR model with (partly random) initial parameters.

    """
    nw, nz = f_static.nw, f_static.nz
    ny, nu = bla.D_yu.shape
    nx = bla.A.shape[0]

    key = _misc.get_key(f_static.seed, "nonlin_lfr")

    key_B_w, key_C_z, key_D_yw, key_D_zu = jax.random.split(key, 4)

    # We want our initial loss to be very close to the BLA loss. To achieve
    # this, we make the elements of B_w and D_yw small enough so the
    # contribution of the static nonlinearity does not significantly affect
    # the initial loss. (Initializing B_w and D_yw with zeros could hamper
    # convergence because of zero gradients.)
    scale = 1e-6
    B_w = scale * jax.random.normal(key_B_w, (nx, nw))
    D_yw = scale * jax.random.normal(key_D_yw, (ny, nw))

    C_z = jax.random.normal(key_C_z, (nz, nx))
    D_zu = jax.random.normal(key_D_zu, (nz, nu))

    return ModelNonlinearLFR(
        bla.A,
        bla.B_u,
        C_y=bla.C_y,
        D_yu=bla.D_yu,
        B_w=B_w,
        C_z=C_z,
        D_yw=D_yw,
        D_zu=D_zu,
        f_static=f_static,
        ts=bla.ts,
        norm=bla.norm,
    )
