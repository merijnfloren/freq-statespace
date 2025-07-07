"""Data structures in time and frequency domains, including metadata."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from freq_statespace import _best_linear_approximation


@dataclass(frozen=True)
class TimeData:
    """Normalized time-domain data container.

    Attributes
    ----------
    u : jnp.ndarray, shape (N, nu, R)
        Normalized input signals, averaged over periods.
    y : jnp.ndarray, shape (N, ny, R)
        Normalized output signals, averaged over periods.
    t : np.ndarray, shape (N,)
        Time vector of a single period.
    ts : float
        Sampling time in seconds.

    """

    u: jnp.ndarray
    y: jnp.ndarray
    t: np.ndarray
    ts: float


@dataclass(frozen=True)
class NonparametricBLA:
    """Nonparametric Best Linear Approximation (BLA) and distortion levels.

    Attributes
    ----------
    G : np.ndarray
        Nonparametric BLA estimate, shape (F, ny, nu).
    var_noise : np.ndarray, shape (F, ny, nu), optional
        Estimated noise variance per period. Is `None` if `P == 1`.
    var_tot : np.ndarray, shape (F, ny, nu), optional
        Estimated total variance per experiment (`R // nu`). Is `None`
        if `R // nu == 1`.

    """

    G: np.ndarray
    var_noise: np.ndarray | None
    var_tot: np.ndarray | None


@dataclass(frozen=True)
class FrequencyData:
    """Normalized frequency-domain data container.

    Attributes
    ----------
    G_bla : `NonparametricBLA`
    U : jnp.ndarray, shape (N//2 + 1, nu, R)
        Normalized input DFT, averaged over periods.
    Y : jnp.ndarray, shape (N//2 + 1, ny, R)
        Normalized output DFT, averaged over periods.
    Y_var_noise : jnp.ndarray, shape (F, ny), optional
        Estimated output noise variance per period. Is `None` if `P == 1`.
    f : np.ndarray, shape (N//2 + 1,)
        Complete frequency vector.
    f_idx : np.ndarray, shape (F,)
        Excited frequency indices.
    fs : float
        Sampling frequency in Hz.

    """

    G_bla: NonparametricBLA
    U: jnp.ndarray
    Y: jnp.ndarray
    Y_var_noise: jnp.ndarray | None
    f: np.ndarray
    f_idx: np.ndarray
    fs: float


@dataclass(frozen=True)
class Normalizer:
    """Normalization statistics for input/output signals.

    Attributes
    ----------
    u_mean : np.ndarray, shape (nu,)
        Input means.
    u_std : np.ndarray, shape (nu,)
        Input standard deviations.
    y_mean : np.ndarray, shape (ny,)
        Output means.
    y_std : np.ndarray, shape (ny,)
        Output standard deviations.

    """

    u_mean: np.ndarray
    u_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


@dataclass(frozen=True)
class InputOutputData:
    """Combined time and frequency domain data.

    Attributes
    ----------
    time : `TimeData`
    freq : `FrequencyData`
    norm : `Normalizer`

    """

    time: TimeData
    freq: FrequencyData
    norm: Normalizer


def create_data_object(
    u: np.ndarray,
    y: np.ndarray,
    f_idx: np.ndarray,
    fs: float
) -> InputOutputData:
    """Create InputOutputData object from time-domain signals and frequency metadata.

    Parameters
    ----------
    u : np.ndarray, shape (N, nu, R, P)
        Input time series, a (random-phase) (multi)sine with 4 dimensions:
        - N: Number of samples per period;
        - nu: Number of input channels;
        - R: Number of independent phase realizations. Each realization must
             have the same frequency content and amplitude characteristics;
        - P: Number of periods (copies of the same realization).
        It is fine if only a single realization and/or period is provided,
        but is is important to shape the data in the required 4D format.
    y : np.ndarray, shape (N, ny, R, P)
        Output time series. Similar structure as `u`, with ny as the number of
        output channels. All periods must be in steady-state,
        i.e., there should be almost no transient effects.
    f_idx : np.ndarray, shape (F,)
        Indices of excited frequencies, where `F ≤ N//2 + 1`.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    InputOutputData
        Processed (meta)data in time and frequency domains.

    """
    # Validate dimensions
    if u.ndim != 4:
        raise ValueError("`u` must have 4 dimensions: (N, nu, R, P).")
    if y.ndim != 4:
        raise ValueError("`y` must have 4 dimensions: (N, ny, R, P).")
    if u.shape[0] != y.shape[0]:
        raise ValueError("`u` and `y` must have same number of time samples.")
    if u.shape[2] != y.shape[2]:
        raise ValueError("`u` and `y` must have same number of realizations.")
    if u.shape[3] != y.shape[3]:
        raise ValueError("`u` and `y` must have same number of periods.")

    ts = 1 / fs
    N, _, R, P = y.shape
    t = np.arange(N) * ts

    # Normalize data (zero mean, unit variance)
    u_mean = u.mean(axis=(0, 2, 3), keepdims=True)
    y_mean = y.mean(axis=(0, 2, 3), keepdims=True)
    u_std = u.std(axis=(0, 2, 3), keepdims=True)
    y_std = y.std(axis=(0, 2, 3), keepdims=True)

    u = (u - u_mean) / u_std
    y = (y - y_mean) / y_std

    # Compute DFTs
    U = np.fft.rfft(u, axis=0)
    Y = np.fft.rfft(y, axis=0)
    f = np.arange(N//2 + 1) * fs / N

    # Compute output noise variance (will be used in loss functions)
    Y_avg = Y.mean(axis=3)
    if P > 1:
        sqr_error = np.abs(Y - Y_avg[..., None])**2  # shape (F, ny, R, P)
        tot_sqr_error = sqr_error.sum(axis=(2, 3))  # shape (F, ny)
        Y_var_noise = tot_sqr_error / (R * (P - 1))  # shape (F, ny)
        Y_var_noise = jnp.asarray(Y_var_noise)
    else:
        Y_var_noise = None

    # Compute nonparametric BLA
    G_bla = _best_linear_approximation.compute_nonparametric(U[f_idx], Y[f_idx])

    # We proceed with data that is averaged over periods
    u_avg, y_avg = u.mean(axis=3), y.mean(axis=3)
    U_avg = U.mean(axis=3)

    # Finally, we convert the input-output data to Jax arrays
    u_avg, y_avg = jnp.asarray(u_avg), jnp.asarray(y_avg)
    U_avg, Y_avg = jnp.asarray(U_avg), jnp.asarray(Y_avg)

    return InputOutputData(
        TimeData(u_avg, y_avg, t, ts),
        FrequencyData(G_bla, U_avg, Y_avg, Y_var_noise, f, f_idx, fs),
        Normalizer(u_mean.flatten(), u_std.flatten(),
                   y_mean.flatten(), y_std.flatten())
    )
