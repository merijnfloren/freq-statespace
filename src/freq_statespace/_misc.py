"""Miscellaneous utility functions."""

import jax
import nonlinear_benchmarks
import numpy as np

from freq_statespace._data_manager import InputOutputData, create_data_object
from freq_statespace._model_structures import ModelBLA, ModelNonlinearLFR
from freq_statespace._solve import SolveResult


def load_and_preprocess_silverbox_data() -> InputOutputData:
    """Load (from `nonlinear_benchmarks`) and preprocesses the Silverbox dataset."""
    # Load data
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


def evaluate_model_performance(
    model: ModelBLA | ModelNonlinearLFR,
    data: InputOutputData,
    *,
    solve_result: SolveResult | None = None,
    offset: int | None = None,
) -> None:
    """Simulate model and prints NRMSEs, along with solver timings (if provided).

    Parameters
    ----------
    model : `ModelBLA` or `ModelNonlinearLFR`
        Model to be evaluated. Can be a BLA or NL-LFR model.
    data : `InputOutputData`
        Measured input-output data.
    solve_result : `SolveResult`, optional
        Result of the model solving process, containing timing and loss
        information.
    offset : int, optional
        A non-negative integer `≤ N`. This value is used to select
        the unknown initial state of the time-domain simulations. Specifically,
        the initial state is selected from the steady-state BLA state
        simulations as `x0 = x_bla[-offset, :, :]`. The number of input samples
        are prepended accordingly.  If `offset` is not specified, two entire
        periods are simulated from zero initial state. The first period is
        then discarded when computing the NRMSE.

    """
    if not isinstance(model, ModelBLA | ModelNonlinearLFR):
        raise TypeError("`model` must be either `ModelBLA` or `ModelNonlinearLFR`.")

    u, y = data.time.u, data.time.y
    N, ny = y.shape[:2]

    # Determine offset and initial state
    if offset is None:
        offset = N
        x0 = None
    else:
        if isinstance(model, ModelNonlinearLFR):
            bla = super(ModelNonlinearLFR, model)
        else:
            bla = model

        # Obtain steady-state BLA state simulations
        x_bla = bla._simulate(u, offset=N)[1]
        x0 = x_bla[-offset, :, :]

    # Simulate model
    y_sim = model._simulate(u, offset=offset, x0=x0)[0]

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
    """Generate a deterministic key from a base seed and a tag."""
    tag = hash(tag) & 0xFFFFFFFF  # ensure it's in 32-bit range
    return jax.random.fold_in(jax.random.key(seed), tag)
