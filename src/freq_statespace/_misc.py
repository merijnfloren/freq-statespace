"""Miscellaneous utility functions."""

import jax
import jax.numpy as jnp
import nonlinear_benchmarks
import numpy as np

from ._data_manager import InputOutputData, create_data_object
from ._model_structures import ModelBLA, ModelNonlinearLFR
from ._solve import SolveResult


def load_and_preprocess_silverbox_data() -> InputOutputData:
    """Load (from `nonlinear_benchmarks`) and preprocesses the Silverbox dataset.
    
    Returns
    -------
    data : `InputOutputData`
        Preprocessed Silverbox training data.

    """
    train = nonlinear_benchmarks.Silverbox()[0]
    u, y = train.u, train.y

    N = 8192  # number of samples per period
    R = 6  # number of random phase multisine realizations
    P = 1  # number of periods

    nu, ny = 1, 1  # SISO system

    fs = 1e7 / 2**14  # [Hz]
    f_idx = np.arange(1, 2 * 1342, 2)  # excited odd harmonics

    # Process data
    N_init = 164  # number of initial samples to be discarded
    N_z = 100  # number of zero samples separating the blocks visually
    N_tr = 400  # number of transient samples

    u_train = np.zeros((N, R))
    y_train = np.zeros((N, R))
    for k in range(R):
        if k == 0:
            u = u[N_init:]
            y = y[N_init:]
        else:
            idx = N_z + N_tr
            u = u[idx:]
            y = y[idx:]

        u_train[:, k] = u[:N]
        y_train[:, k] = y[:N]

        u = u[N:]
        y = y[N:]

    # Reshape data to required dimensions
    u_train = u_train.reshape(N, nu, R, P)
    y_train = y_train.reshape(N, ny, R, P)

    return create_data_object(u_train, y_train, f_idx, fs)


def extend_signal(u: jnp.ndarray, offset: int) -> jnp.ndarray:
    """Extend input signal by prepending last `offset` samples.

    Parameters
    ----------
    u : jnp.ndarray, shape (N, nu, R)
        Input signal.
    offset : int
        Number of samples to prepend.

    Returns
    -------
    u_ext : jnp.ndarray, shape (N + offset, nu, R)
        Extended input signal.

    """
    N = u.shape[0]
    repeats = (offset // N) + 1
    remainder = offset % N
    
    u_ext = jnp.tile(u, (repeats, 1, 1))
    if remainder > 0:
        u_ext = jnp.concatenate((u[-remainder:, ...], u_ext), axis=0)
    
    return u_ext


def compute_steady_state_bla_state(bla: ModelBLA, data: InputOutputData) -> jnp.ndarray:
    """Compute the steady-state BLA state trajectories."""
    N, nu = data.time.u.shape[:2]
    nx = bla.A.shape[0]
    
    G_xu = ModelBLA(  # parametric u->x frequency response; not the true BLA
        A=bla.A,
        B_u=bla.B_u,
        C_y=jnp.eye(nx),
        D_yu=jnp.zeros((nx, nu)), 
        ts=bla.ts,
        norm=bla.norm,
    )._frequency_response(data.freq.f)  # shape (N//2 + 1, nx, nu)
    X_bla = G_xu @ data.freq.U  # shape (N//2 + 1, nx, R)
    x_bla = jnp.fft.irfft(X_bla, n=N, axis=0)  # shape (N, nx, R)
    return x_bla


def evaluate_model_performance(
    model: ModelBLA | ModelNonlinearLFR,
    data: InputOutputData,
    *,
    x0: jnp.ndarray,
    offset: int,
    solve_result: SolveResult | None = None
) -> None:
    """Simulate `model` in the time domain and evaluate performance.

    Parameters
    ----------
    model : `ModelBLA` or `ModelNonlinearLFR`
        Model to be evaluated.
    data : `InputOutputData`
        Measured input-output data.
    x0 : jnp.ndarray of shape (nx, R)
        Initial state used at the beginning of each simulated realization.
    offset : int
        A non-negative number of initial samples to prepend to the input signal in
        order to let the system reach steady state before the main simulation
        begins. These prepended samples are discarded from the returned outputs.

        The value of `offset` must be consistent with `x0`: the provided initial
        state `x0` must correspond to the  state at time index `-offset`.
    solve_result : `SolveResult`, optional
        Result of the model solving process, containing timing and loss
        information.
        
    Raises
    ------
    TypeError
        If `model` is not of type `ModelBLA` or `ModelNonlinearLFR`.

    """
    if not isinstance(model, ModelBLA | ModelNonlinearLFR):
        raise TypeError("`model` must be either `ModelBLA` or `ModelNonlinearLFR`.")

    u, y = data.time.u, data.time.y
    N, ny, R = y.shape
        
    if offset > 0:
        u = extend_signal(u, offset)
        
    # Simulate model
    y_sim = model._simulate(u, x0)[0]
    y_sim = y_sim[offset:, ...]  # discard offset samples

    # Compute NRMSE per output channel
    error = y - y_sim
    mse = np.mean(error**2, axis=(0, 2))
    norm = np.mean(y**2, axis=(0, 2))
    nrmse = 100 * np.sqrt(mse / norm)  # as a percentage

    if solve_result is not None:
        avg_time = solve_result.iter_times.mean()
        unit = "s"
        if avg_time < 1:
            avg_time *= 1000
            unit = "ms"

        print(
            f"Optimization completed in {solve_result.wall_time:.2f}s "
            f"({solve_result.iter_count} iterations, "
            f"{avg_time:.2f}{unit}/iter).\n"
        )

    name = "NL-LFR" if isinstance(model, ModelNonlinearLFR) else "BLA"

    if ny == 1:
        print(f"{name} simulation error: {nrmse[0]:.2f}%.")
    else:
        print(f"{name} simulation error:")
        for k in range(ny):
            print(f"    output {k + 1}: {nrmse[k]:.2f}%.")
    print("")


def get_key(seed: int, tag: str) -> jax.Array:
    """Generate a deterministic key from a base seed and a tag.
    
    Parameters
    ----------
    seed : int
        Base seed.
    tag : str
        Tag to differentiate keys.
        
    Returns
    -------
    key : jax.Array
        A new random key that is a deterministic function of the inputs.

    """
    tag = hash(tag) & 0xFFFFFFFF  # ensure it's in 32-bit range
    return jax.random.fold_in(jax.random.key(seed), tag)


def real_valued(loss: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split complex loss into real and imaginary parts."""
    return loss.real, loss.imag


def scalar_valued(loss: jnp.ndarray) -> float:
    """Compute scalar loss from complex loss by summing squared magnitudes."""
    return jnp.sum(jnp.abs(loss) ** 2)
