"""
Data structures for time and frequency domain system identification data.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimeDomain:
    """Normalized time-domain data container."""
    u: np.ndarray  # normalized inputs, shape (N, nu, R, P)
    y: np.ndarray  # normalized outputs, shape (N, ny, R, P)
    t: np.ndarray  # time vector of a single period, shape (N,)
    ts: float  # sampling time in seconds, shape


@dataclass(frozen=True)
class FrequencyDomain:
    """Normalized frequency-domain data container."""
    U: np.ndarray  # normalized input DFT, shape (F, nu, R, P)
    Y: np.ndarray  # normalized output DFT, shape (F, ny, R, P)
    f: np.ndarray  # complete frequency vector, shape (N//2 + 1,)
    f_idx: np.ndarray  # excited frequency indices, shape (F,)
    fs: float  # sampling frequency in Hz


@dataclass(frozen=True)
class Normalizer:
    """Stores normalization statistics for input/output signals."""
    u_mean: np.ndarray  # input means, shape (nu,)
    u_std: np.ndarray  # input means, shape (nu,)
    y_mean: np.ndarray  # output means, shape (ny,)
    y_std: np.ndarray  # output means, shape (ny,)


@dataclass(frozen=True)
class InputOutputData:
    """Combined time and frequency domain data."""
    time: TimeDomain
    freq: FrequencyDomain
    norm: Normalizer


def create_data_object(
    u: np.ndarray,
    y: np.ndarray,
    fs: float,
    f_idx: np.ndarray
) -> InputOutputData:
    """
    Create InputOutputData from time domain signals.

    Parameters
    ----------
    u : np.ndarray, shape (N, nu, R, P)
        Input time series
    y : np.ndarray, shape (N, ny, R, P)
        Output time series
    fs : float
        Sampling frequency
    f_idx : np.ndarray
        Excited frequency indices

    Returns
    -------
    InputOutputData
        Processed data with time and frequency domain representations
    """
    # Validate dimensions
    if u.ndim != 4:
        raise ValueError('u must have 4 dimensions: (N, nu, R, P).')
    if y.ndim != 4:
        raise ValueError('y must have 4 dimensions: (N, ny, R, P).')
    if u.shape[0] != y.shape[0]:
        raise ValueError('u and y must have same number of time samples.')
    if u.shape[2] != y.shape[2]:
        raise ValueError('u and y must have same number of realizations.')
    if u.shape[3] != y.shape[3]:
        raise ValueError('u and y must have same number of periods.')

    ts = 1 / fs
    N = u.shape[0]
    t = np.arange(N) * ts

    # Normalize data (zero mean, unit variance)
    u_mean = u.mean(axis=(0, 2, 3), keepdims=True)
    y_mean = y.mean(axis=(0, 2, 3), keepdims=True)
    u_std = u.std(axis=(0, 2, 3), keepdims=True)
    y_std = y.std(axis=(0, 2, 3), keepdims=True)

    u_norm = (u - u_mean) / u_std
    y_norm = (y - y_mean) / y_std

    # Compute DFT
    U = np.fft.rfft(u_norm, axis=0)
    Y = np.fft.rfft(y_norm, axis=0)
    f = np.arange(N//2 + 1) * fs / N

    return InputOutputData(
        TimeDomain(u_norm, y_norm, t, ts),
        FrequencyDomain(U, Y, f, f_idx, fs),
        Normalizer(u_mean.flatten(), u_std.flatten(),
                   y_mean.flatten(), y_std.flatten())
    )
