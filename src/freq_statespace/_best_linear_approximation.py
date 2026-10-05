"""Nonparametric BLA, parametric subspace identification, and optimizer."""
from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from scipy.linalg import solve_discrete_lyapunov

from freq_statespace import _misc
from freq_statespace._config import PRINT_EVERY, SOLVER, STABILITY_MARGIN, DeviceLike
from freq_statespace._data_manager import BLAEstimate, FrequencyData, InputOutputData
from freq_statespace._model_structures import ModelBLA
from freq_statespace._solve import SolveResult, solve
from freq_statespace.dep import fsid

if TYPE_CHECKING:
    from jaxtyping import Array, Float

    from freq_statespace._typing import ComplexArray, RealArray


MAX_ITER = 1000  # when changing, also update corresponding docstring!


class _StableBLAParameters(eqx.Module):
    """Unconstrained parameters of a BLA with a contractive A matrix."""

    theta_L: Float[Array, "n_lower_triangle_entries"]
    theta_W: Float[Array, "n_upper_triangle_entries"]
    B_u: Float[Array, "nx nu"]
    C_y: Float[Array, "ny nx"]
    D_yu: Float[Array, "ny nu"]


### Public functions ###


def subspace_id(
    data: InputOutputData,
    nx: int,
    nq: int | None = None,
    freq_weighting: bool = True,
    input_output_mode: bool = False,
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
    input_output_mode : bool
        Whether to parametrize the state-space model directly from the input-output
        spectra instead of the nonparametric BLA. This mode is automatically activated 
        if no BLA estimate is available, even if `input_output_mode` is set to `False`. 
        Defaults to `False`.
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
        msg = f"Subspace dimension nq={nq} must be greater than state dimension nx={nx}."
        raise ValueError(msg)

    freq_data = data.freq
    input_output_mode, freq_weighting = _validate_inputs(
        input_output_mode, freq_weighting, freq_data.G_bla, logging_enabled
    )
    
    # Run the subspace identification
    A, B_u, C_y, D_yu = _subspace_id(
        freq_data, nx, nq, freq_weighting, input_output_mode
    )
    model = ModelBLA(A, B_u, C_y, D_yu, 1 / freq_data.fs, data.norm)
    
    if logging_enabled:
        x_bla = _misc.compute_steady_state_bla_state(model, data)
        _misc.evaluate_model_performance(model, data, x0=x_bla[0, :, :], offset=0)

    return model


def optimize(
    model: ModelBLA,
    data: InputOutputData,
    *,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser = SOLVER,
    enforce_stability: bool = False,
    stability_margin: float = STABILITY_MARGIN,
    freq_weighting: bool = True,
    input_output_mode: bool = False,
    max_iter: int = MAX_ITER,
    print_every: int = PRINT_EVERY,
    return_solve_details: bool = False,
    device: DeviceLike = None,
) -> ModelBLA | tuple[ModelBLA, SolveResult]:
    """Optimize the BLA parameters via a frequency-domain fit to G_bla.

    Parameters
    ----------
    model : `ModelBLA`
        Initial BLA model to be optimized.
    data : `InputOutputData`
        Estimation data.
    solver : `optx.AbstractLeastSquaresSolver` or `optx.AbstractMinimiser`
        Any least-squares solver or general minimization solver from the
        Optimistix or Optax libraries. Defaults to 
        `optx.LevenbergMarquardt(rtol=1e-3, atol=1e-6)`.
    enforce_stability : bool
        Whether to constrain the state-transition matrix to be stable throughout
        optimization. Defaults to `False`.
    stability_margin : float
        Margin from the unit circle when ``enforce_stability`` is enabled. The
        optimizer enforces a spectral norm below ``1 - stability_margin``.
        Must lie in ``(0, 1)``. Defaults to ``1e-4``.
    freq_weighting : bool
        Whether to use frequency weighting based on the inverse of the total variance
        on the nonparametric BLA. Defaults to `True`.
    input_output_mode : bool
        Whether to optimize the state-space model directly from the input-output
        spectra instead of the nonparametric BLA. This mode is automatically activated 
        if no BLA estimate is available, even if `input_output_mode` is set to `False`. 
        Defaults to `False`.
    max_iter : int
        Maximum number of optimization iterations. Defaults to `1000`.
    print_every : int
        Frequency of printing iteration information. If set to `0`, only a
        summary is printed. If set to `-1`, no printing is done. Defaults to `1`.
    return_solve_details : bool
        Whether to return detailed information about the optimization process. This is
        useful for e.g. plotting convergence curves. Defaults to `False`.
    device : `DeviceLike`, optional
        Device on which to perform the computations. Can be either a device
        name (`"cpu"`, `"gpu"`, or `"tpu"`) or a specific JAX device. If not
        provided, the default JAX device is used.
        
    Returns
    -------
    `ModelBLA`
        BLA model with optimized parameters.
    `SolveResult`, optional
        More details about the optimization process, only returned if
        `return_solve_details` is `True`.

    """
    if enforce_stability:
        _validate_stability_margin(stability_margin)

    logging_enabled = print_every != -1
    
    if logging_enabled:
        header = (
            " Stability-enforced BLA optimization "
            if enforce_stability else " BLA optimization "
        )
        print(f"{header:=^72}")
    
    input_output_mode, freq_weighting = _validate_inputs(
        input_output_mode, freq_weighting, data.freq.G_bla, logging_enabled
    )
    
    # Ensure Normalizer is static
    model = eqx.tree_at(lambda tree: tree.norm, model, replace=None)
    
    # Run the optimization
    if enforce_stability:
        model, solve_result = _optimize_stable(
            model, data, input_output_mode, freq_weighting, logging_enabled,
            solver, max_iter, print_every, device, stability_margin
        )
    else:
        model, solve_result = _optimize(
            model, data, input_output_mode, freq_weighting,
            logging_enabled, solver, max_iter, print_every, device
        )
    
    # Add Normalizer back to the model
    model = eqx.tree_at(lambda tree: tree.norm, model, replace=data.norm)

    if logging_enabled:
        x_bla = _misc.compute_steady_state_bla_state(model, data)
        _misc.evaluate_model_performance(
            model, data, x0=x_bla[0, :, :], offset=0, solve_result=solve_result
        )

    if return_solve_details:
        return model, solve_result
    return model


### Internal helpers ###


def _subspace_id(
    freq_data: FrequencyData,
    nx: int,
    nq: int,
    freq_weighting: bool,
    input_output_mode: bool
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Perform actual subspace identification with validated inputs."""
    freqs = freq_data.freqs[freq_data.excited_bins]
    fs = freq_data.fs
    z = 2 * np.pi * freqs / fs
    
    if input_output_mode:
        n_bins = len(freqs)
        _, ny, n_realizations = freq_data.Y.shape
        nu = freq_data.U.shape[1]
        
        Y = np.transpose(freq_data.Y[freq_data.excited_bins], (0, 2, 1)).reshape(
            n_realizations * n_bins, ny
        )
        U = np.transpose(freq_data.U[freq_data.excited_bins], (0, 2, 1)).reshape(
            n_realizations * n_bins, nu
        )
        zj = np.repeat(np.exp(z * 1j), n_realizations)
        W = np.empty(0)  # no weighting in input-output mode
        
    else:
        G_bla = freq_data.G_bla
        n_bins, ny, nu = G_bla.G.shape

        # Convert BLA to "input-output form" for FSID algorithm compatibility
        Y = np.transpose(G_bla.G, (0, 2, 1)).reshape(nu * n_bins, ny)
        U = np.tile(np.eye(nu), (n_bins, 1))
        zj = np.repeat(np.exp(z * 1j), nu)

        # Create weighting matrix (inverse of total variance)
        if freq_weighting:
            W_temp = 1 / G_bla.var_tot

            # The four lines below are to ensure compatibility with fsid.gfdsid
            W_temp = np.transpose(np.sqrt(W_temp), (0, 2, 1)).reshape(nu * n_bins, ny)
            W = np.zeros((nu * n_bins, ny, ny))
            for k in range(nu * n_bins):
                np.fill_diagonal(W[k], W_temp[k])
        else:
            W = np.empty(0)

    # Ensure that zj, Y, and U are NumPy (i.e., non-JAX) arrays
    zj = np.asarray(zj, dtype=np.complex128)
    Y = np.asarray(Y, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)

    # Perform frequency-domain subspace identification
    fddata = (zj, Y, U)
    A, B_u, C_y, D_yu = fsid.gfdsid(fddata=fddata, n=nx, q=nq, estTrans=False, w=W)[:4]
    return A, B_u, C_y, D_yu


def _optimize(
    model: ModelBLA,
    data: InputOutputData,
    input_output_mode: bool,
    freq_weighting: bool,
    logging_enabled: bool,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser,
    max_iter: int,
    print_every: int,
    device: DeviceLike,
) -> tuple[ModelBLA, SolveResult]:
    """Perform actual optimization with validated inputs."""
    freqs = data.freq.freqs[data.freq.excited_bins]
    model = _normalize_states(model, data)
    theta0, theta_static = eqx.partition(model, eqx.is_inexact_array)

    if input_output_mode:
        U_nonpar = jnp.asarray(data.freq.U)[data.freq.excited_bins]
        Y_nonpar = jnp.asarray(data.freq.Y)[data.freq.excited_bins]
        args = (theta_static, U_nonpar, Y_nonpar, freqs)
        loss_fn = _loss_output_spectrum
   
    else:
        G_bla = data.freq.G_bla
        
        # Create weighting matrix (inverse of total variance)
        if freq_weighting:
            W = 1 / G_bla.var_tot
        else:
            W = jnp.ones_like(G_bla.G)

        args = (theta_static, jnp.asarray(G_bla.G), freqs, W)
        loss_fn = _loss_frequency_response

    # Optimize the model parameters
    if logging_enabled:
        print("Starting iterative optimization...")
    solve_result = solve(theta0, solver, args, loss_fn, max_iter, print_every, device)

    model = eqx.combine(solve_result.theta, theta_static)
    model = _normalize_states(model, data)
    return model, solve_result


def _optimize_stable(
    model: ModelBLA,
    data: InputOutputData,
    input_output_mode: bool,
    freq_weighting: bool,
    logging_enabled: bool,
    solver: optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser,
    max_iter: int,
    print_every: int,
    device: DeviceLike,
    stability_margin: float,
) -> tuple[ModelBLA, SolveResult]:
    """Optimize a BLA with a contractive state-transition parameterization."""
    theta0 = _stable_parameters_from_model(
        model, stability_margin
    )

    if input_output_mode:
        args = (
            model.ts,
            jnp.asarray(data.freq.U)[data.freq.excited_bins],
            jnp.asarray(data.freq.Y)[data.freq.excited_bins],
            data.freq.freqs[data.freq.excited_bins],
            stability_margin,
        )
        loss_fn = _loss_stable_output_spectrum
    else:
        G_bla = data.freq.G_bla
        W = 1 / G_bla.var_tot if freq_weighting else jnp.ones_like(G_bla.G)
        args = (
            model.ts,
            jnp.asarray(G_bla.G),
            data.freq.freqs[data.freq.excited_bins],
            W,
            stability_margin,
        )
        loss_fn = _loss_stable_frequency_response

    if logging_enabled:
        print("Starting iterative optimization...")
    solve_result = solve(
        theta0, solver, args, loss_fn, max_iter, print_every, device
    )

    model = _stable_model_from_parameters(
        solve_result.theta, model.ts, stability_margin, norm=None
    )
    model = _normalize_states(model, data)

    return model, solve_result


def _loss_frequency_response(theta_dyn: ModelBLA, args: tuple) -> tuple:
    """Compute the weighted loss between the nonparametric and parametric BLA."""
    theta_static, G_nonpar, freqs, W = args

    theta = eqx.combine(theta_dyn, theta_static)

    G_par = theta._frequency_response(freqs)
    loss = jnp.sqrt(W / G_nonpar.size) * (G_par - G_nonpar)
    return _misc.real_valued(loss), (_misc.scalar_valued(loss),)


def _loss_output_spectrum(theta_dyn: ModelBLA, args: tuple) -> tuple:
    """Compute the loss between the nonparametric and parametric output spectra."""
    theta_static, U_nonpar, Y_nonpar, freqs = args

    theta = eqx.combine(theta_dyn, theta_static)

    Y_par = theta._frequency_response(freqs) @ U_nonpar
    loss = jnp.sqrt(1 / Y_nonpar.size) * (Y_par - Y_nonpar)
    return _misc.real_valued(loss), (_misc.scalar_valued(loss),)


def _loss_stable_frequency_response(
    theta: _StableBLAParameters, args: tuple
) -> tuple:
    """Compute the complex-FRF residual for stable BLA parameters."""
    ts, G_nonpar, freqs, W, stability_margin = args
    model = _stable_model_from_parameters(theta, ts, stability_margin, norm=None)
    G_par = model._frequency_response(freqs)
    loss = jnp.sqrt(W / G_nonpar.size) * (G_par - G_nonpar)
    return _misc.real_valued(loss), (_misc.scalar_valued(loss),)


def _loss_stable_output_spectrum(
    theta: _StableBLAParameters, args: tuple
) -> tuple:
    """Compute the output-spectrum residual for stable BLA parameters."""
    ts, U_nonpar, Y_nonpar, freqs, stability_margin = args
    model = _stable_model_from_parameters(theta, ts, stability_margin, norm=None)
    Y_par = model._frequency_response(freqs) @ U_nonpar
    loss = jnp.sqrt(1 / Y_nonpar.size) * (Y_par - Y_nonpar)
    return _misc.real_valued(loss), (_misc.scalar_valued(loss),)


def _stable_model_from_parameters(
    theta: _StableBLAParameters,
    ts: float,
    stability_margin: float,
    norm,
) -> ModelBLA:
    """Construct a BLA whose A matrix is contractive by construction."""
    A = _stable_A_from_parameters(
        theta.theta_L, theta.theta_W, theta.B_u.shape[0], stability_margin
    )
    return ModelBLA(A, theta.B_u, theta.C_y, theta.D_yu, ts, norm)


def _stable_A_from_parameters(
    theta_L: Float[Array, "n_lower_triangle_entries"],
    theta_W: Float[Array, "n_upper_triangle_entries"],
    nx: int,
    stability_margin: float,
) -> Float[Array, "nx nx"]:
    """Map unconstrained variables to an A with norm below a fixed bound."""
    lower_rows, lower_cols = np.tril_indices(nx)
    diagonal = lower_rows == lower_cols
    upper_rows, upper_cols = np.triu_indices(nx, k=1)

    L_entries = theta_L.at[diagonal].set(jax.nn.softplus(theta_L[diagonal]))
    L = jnp.zeros((nx, nx), dtype=theta_L.dtype)
    L = L.at[lower_rows, lower_cols].set(L_entries)
    W = jnp.zeros((nx, nx), dtype=theta_W.dtype)
    W = W.at[upper_rows, upper_cols].set(theta_W)
    W = W.at[upper_cols, upper_rows].set(-theta_W)

    M = L @ L.T + W
    I = jnp.eye(nx)
    cayley_A = _misc.right_solve(I - M, I + M)
    return (1 - stability_margin) * cayley_A


def _stable_parameters_from_model(
    model: ModelBLA,
    stability_margin: float,
) -> _StableBLAParameters:
    """Create stable parameters from a possibly unstable BLA realization."""
    A = np.asarray(model.A)
    B_u = np.asarray(model.B_u)
    C_y = np.asarray(model.C_y)
    D_yu = np.asarray(model.D_yu)
    _validate_real_finite_model(A, B_u, C_y, D_yu)
    A = A.astype(float)

    stability_radius = 1 - stability_margin
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    if spectral_radius > stability_radius:
        A = A * (stability_radius / spectral_radius)

    # Convert Schur-stable A into Euclidean contractive coordinates. This is
    # exactly a similarity transformation, so B and C must follow it too.
    # P = _misc.solve_discrete_lyapunov(A, np.eye(A.shape[0]))
    P = solve_discrete_lyapunov(A.T, np.eye(A.shape[0]))
    P = 0.5 * (P + P.T)
    T = np.linalg.cholesky(P).T
    A = _misc.right_solve(T @ A, T)
    B_u = T @ B_u
    C_y = _misc.right_solve(C_y, T)

    # Keep the Lyapunov coordinates strictly within the radius implied by the
    # requested unit-circle margin before inverting the Cayley map.
    singular_value = np.linalg.svd(A, compute_uv=False)[0]
    target_norm = stability_radius - min(1e-6, 0.5 * stability_radius)
    if singular_value >= target_norm:
        A = A * (target_norm / singular_value)

    A_unbounded = A / stability_radius
    I = np.eye(A.shape[0])
    M = np.linalg.solve(I + A_unbounded, I - A_unbounded)
    symmetric_M = 0.5 * (M + M.T)
    W = 0.5 * (M - M.T)
    L = np.linalg.cholesky(symmetric_M)

    lower_rows, lower_cols = np.tril_indices(A.shape[0])
    diagonal = lower_rows == lower_cols
    theta_L = L[lower_rows, lower_cols]
    theta_L[diagonal] = _inverse_softplus(theta_L[diagonal])
    upper_rows, upper_cols = np.triu_indices(A.shape[0], k=1)

    return _StableBLAParameters(
        theta_L=jnp.asarray(theta_L),
        theta_W=jnp.asarray(W[upper_rows, upper_cols]),
        B_u=jnp.asarray(B_u),
        C_y=jnp.asarray(C_y),
        D_yu=jnp.asarray(D_yu),
    )


def _inverse_softplus(x: RealArray) -> RealArray:
    """Numerically stable inverse of softplus for strictly positive inputs."""
    return np.where(x > 20, x, np.log(np.expm1(x)))


def _validate_stability_margin(stability_margin: float) -> None:
    """Validate the unit-circle margin used by the enforced optimizer."""
    if not 0 < stability_margin < 1:
        msg = f"stability_margin must lie strictly between 0 and 1; got {stability_margin}."
        raise ValueError(msg)


def _validate_real_finite_model(*matrices: RealArray) -> None:
    """Reject models that cannot be used by the real-valued parameterization."""
    if any(np.iscomplexobj(matrix) for matrix in matrices):
        msg = "Stability-enforced optimization requires a real-valued model."
        raise ValueError(msg)
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
        msg = "Stability-enforced optimization requires finite model matrices."
        raise ValueError(msg)


def _normalize_states(model: ModelBLA, data: InputOutputData) -> ModelBLA:
    """Normalize BLA model states to have unit variance."""
    nx, nu = model.B_u.shape
    n_samples = data.time.u.shape[0]

    G_xu = ModelBLA(  # parametric u->x frequency response; not the true BLA
        A=model.A, B_u=model.B_u, C_y=np.eye(nx), D_yu=np.zeros((nx, nu)), 
        ts=model.ts, norm=model.norm,
    )._frequency_response(data.freq.freqs)  # shape (n_samples // 2 + 1, nx, nu)

    X = G_xu @ data.freq.U  # shape (n_samples // 2 + 1, nx, n_realizations)
    x = np.fft.irfft(X, n=n_samples, axis=0)  # shape (n_samples, nx, n_realizations)
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
        norm=model.norm
    )


def _validate_weighting(
    freq_weighting: bool,
    G_bla: BLAEstimate | None,
    input_output_mode: bool,
    print_warning: bool,
) -> bool:
    """Validate whether frequency weighting can be applied.

    Weighting is disabled if:
      - `input_output_mode` is active, or
      - `freq_weighting` is requested but BLA total variance is unavailable.
    """
    if not freq_weighting:
        return False

    # Case 1: Incompatible with input-output mode
    if input_output_mode:
        if print_warning:
            print(
                "Warning: Frequency weighting based on BLA total variance requested, "
                "but input-output mode is active. Proceeding without weighting."
            )
        return False

    # Case 2: BLA variance not available
    if G_bla is None or G_bla.var_tot is None:
        if print_warning:
            print(
                "Warning: Frequency weighting based on BLA total variance requested, "
                "but such estimate is not available. Proceeding without weighting."
            )
        return False

    return True


def _validate_inputs(
    input_output_mode: bool,
    freq_weighting: bool,
    G_bla: BLAEstimate | None,
    print_warning: bool
) -> tuple[bool, bool]:
    """Validate inputs for subspace identification."""
    # Switch to input-output mode if BLA data is missing
    if G_bla is None and not input_output_mode:
        if print_warning:
            print(
                "Warning: Nonparametric BLA estimate is not available. Proceeding "
                "with input-output mode for subspace identification."
            )
        input_output_mode = True
    
    # Validate frequency weighting settings
    freq_weighting = _validate_weighting(
        freq_weighting, G_bla, input_output_mode, print_warning
    )
    return input_output_mode, freq_weighting
