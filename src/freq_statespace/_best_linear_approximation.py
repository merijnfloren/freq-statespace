"""Nonparametric BLA, parametric subspace identification, and optimizer."""
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from . import _misc
from ._config import PRINT_EVERY, SOLVER, DeviceLike
from ._data_manager import FrequencyData, InputOutputData, NonparametricBLA
from ._model_structures import ModelBLA
from ._solve import SolveResult, solve
from .dep import fsid


MAX_ITER = 1000  # when changing, also update corresponding docstring!


### Public functions ###


def nonparametric_bla(U: np.ndarray, Y: np.ndarray) -> NonparametricBLA:
    """Compute nonparametric BLA and variance estimates from input-output data.

    Parameters
    ----------
    U : np.ndarray, shape (F, nu, R, P)
        DFT input spectrum at the excited frequencies across realizations and periods.
    Y : np.ndarray, shape (F, ny, R, P)
        DFT output spectrum at the excited frequencies across realizations and periods.

    Returns
    -------
    `NonparametricBLA`
        Nonparametric BLA estimate with frequency response and variance estimates.
        
    Raises
    ------
    ValueError
        If the number of realizations R is less than the number of inputs nu.

    """
    nu, R = U.shape[1:3]
    if R < nu:
        raise ValueError(
            "For multi-input systems, the number of realizations (R) must be "
            "at least equal to the number of inputs (nu) to compute the "
            "frequency response matrix."
        )
    
    G = _compute_frequency_response(U, Y)  # shape (F, ny, nu, M, P)
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
        raise ValueError(
            f"Subspace dimension nq={nq} must be greater than state dimension nx={nx}."
        )

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
    freq_weighting: bool = True,
    input_output_mode: bool = False,
    max_iter: int = MAX_ITER,
    print_every: int = PRINT_EVERY,
    return_solve_details: bool = False,
    device: DeviceLike = None
) -> ModelBLA | tuple[ModelBLA, SolveResult]:
    """Refine the parameters of the BLA using frequency-response computations.

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
    logging_enabled = print_every != -1
    
    if logging_enabled:
        header = " BLA optimization "
        print(f"{header:=^72}")
    
    input_output_mode, freq_weighting = _validate_inputs(
        input_output_mode, freq_weighting, data.freq.G_bla, logging_enabled
    )
    
    # Run the optimization
    model, solve_result = _optimize(
        model, data, input_output_mode, freq_weighting,
        logging_enabled, solver, max_iter, print_every, device
    )

    if logging_enabled:
        x_bla = _misc.compute_steady_state_bla_state(model, data)
        _misc.evaluate_model_performance(
            model, data, x0=x_bla[0, :, :], offset=0, solve_result=solve_result
        )

    if return_solve_details:
        return model, solve_result
    return model


### Internal helpers ###


def _compute_frequency_response(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute frequency response matrix G(k) = Y(k) * (U(k))^(-1).

    Parameters
    ----------
    U : np.ndarray, shape (F, nu, R, P)
        DFT input spectra at the excited frequencies across realizations and periods.
    Y : np.ndarray, shape (F, ny, R, P)
        DFT output spectra at the excited frequencies across realizations and periods.

    Returns
    -------
    G : np.ndarray, shape (F, ny, nu, M, P)
        Frequency Response Matrix:
        - F: number of frequency bins;
        - ny: number of outputs;
        - nu: number of inputs;
        - M: number of experiments (R // nu);
        - P: number of periods.

    """
    F, nu, R, P = U.shape
    ny = Y.shape[1]

    M = R // nu
    if M * nu != R:
        print(
            "Warning: Suboptimal number of realizations. Not all realizations "
            "are used to compute the frequency response matrix. Ideally, "
            "the number of realizations (R) should be an integer multiple "
            "of the number of inputs (nu)."
        )

    G = np.zeros((F, ny, nu, M, P), dtype=complex)

    for kf in range(F):
        for kr in range(M):
            for kp in range(P):
                start_idx = kr * nu
                end_idx = (kr + 1) * nu
                U_block = U[kf, :, start_idx:end_idx, kp]
                Y_block = Y[kf, :, start_idx:end_idx, kp]

                U_inv = np.linalg.solve(U_block, np.eye(nu))
                G[kf, :, :, kr, kp] = Y_block @ U_inv

    return G


def _subspace_id(
    freq_data: FrequencyData,
    nx: int,
    nq: int,
    freq_weighting: bool,
    input_output_mode: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform actual subspace identification with validated inputs."""
    freqs = freq_data.f[freq_data.f_idx]
    fs = freq_data.fs
    z = 2 * np.pi * freqs / fs
    
    if input_output_mode:
        F = len(freqs)
        _, ny, R = freq_data.Y.shape
        nu = freq_data.U.shape[1]
        
        Y = np.transpose(freq_data.Y[freq_data.f_idx], (0, 2, 1)).reshape(R * F, ny)
        U = np.transpose(freq_data.U[freq_data.f_idx], (0, 2, 1)).reshape(R * F, nu) 
        zj = np.repeat(np.exp(z * 1j), R)
        W = np.empty(0)  # no weighting in input-output mode
        
    else:
        G_bla = freq_data.G_bla
        F, ny, nu = G_bla.G.shape

        # Convert BLA to "input-output form" for FSID algorithm compatibility
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
    freqs = data.freq.f[data.freq.f_idx]
    model = _normalize_states(model, data)
    theta0, theta_static = eqx.partition(model, eqx.is_inexact_array)

    if input_output_mode:
        U_nonpar = jnp.asarray(data.freq.U)[data.freq.f_idx]
        Y_nonpar = jnp.asarray(data.freq.Y)[data.freq.f_idx]
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


def _normalize_states(model: ModelBLA, data: InputOutputData) -> ModelBLA:
    """Normalize BLA model states to have unit variance."""
    nx, nu = model.B_u.shape
    N = data.time.u.shape[0]

    G_xu = ModelBLA(  # parametric u->x frequency response; not the true BLA
        A=model.A, B_u=model.B_u, C_y=np.eye(nx), D_yu=np.zeros((nx, nu)), 
        ts=model.ts, norm=model.norm,
    )._frequency_response(data.freq.f)  # shape (N//2 + 1, nx, nu)

    X = G_xu @ data.freq.U  # shape (N//2 + 1, nx, R)
    x = np.fft.irfft(X, n=N, axis=0)  # shape (N, nx, R)
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
    G_bla: NonparametricBLA | None,
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
    G_bla: NonparametricBLA | None,
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
