"""
Data structures for time and frequency domain system identification.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimeData:
    """
    Normalized time-domain data container.

    Attributes
    ----------
    u : np.ndarray
        Normalized input signals, shape (N, nu, R, P).
    y : np.ndarray
        Normalized output signals, shape (N, ny, R, P).
    t : np.ndarray
        Time vector of a single period, shape (N,).
    ts : float
        Sampling time in seconds.
    """
    u: np.ndarray
    y: np.ndarray
    t: np.ndarray
    ts: float


@dataclass(frozen=True)
class FrequencyData:
    """
    Normalized frequency-domain data container.

    Attributes
    ----------
    U : np.ndarray
        Normalized input DFT, shape (F, nu, R, P).
    Y : np.ndarray
        Normalized output DFT, shape (F, ny, R, P).
    f : np.ndarray
        Complete frequency vector, shape (N//2 + 1,).
    f_idx : np.ndarray
        Excited frequency indices, shape (F,).
    fs : float
        Sampling frequency in Hz.
    """
    U: np.ndarray
    Y: np.ndarray
    f: np.ndarray
    f_idx: np.ndarray
    fs: float


@dataclass(frozen=True)
class Normalizer:
    """
    Stores normalization statistics for input/output signals.

    Attributes
    ----------
    u_mean : np.ndarray
        Input means, shape (nu,).
    u_std : np.ndarray
        Input standard deviations, shape (nu,).
    y_mean : np.ndarray
        Output means, shape (ny,).
    y_std : np.ndarray
        Output standard deviations, shape (ny,).
    """
    u_mean: np.ndarray
    u_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


@dataclass(frozen=True)
class InputOutputData:
    """
    Combined time and frequency domain data.

    Attributes
    ----------
    time : TimeData
        Time-domain representation of the signal.
    freq : FrequencyData
        Frequency-domain representation of the signal.
    norm : Normalizer
        Normalization statistics used for scaling.
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
    """
    Create InputOutputData from time domain signals.

    Parameters
    ----------
    u : np.ndarray, shape (N, nu, R, P)
        Input time series, must have 4 dimensions:
        - N: number of time samples;
        - nu: number of inputs;
        - R: number of realizations;
        - P: number of periods.
    y : np.ndarray, shape (N, ny, R, P)
        Output time series. Similar structure as `u`, with ny as number of
        outputs.
    f_idx : np.ndarray, shape (F,)
        Indices of excited frequencies, where F is smaller than or equal to
        N//2 + 1.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    InputOutputData
        Processed (meta)data in time and frequency domains.
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
        TimeData(u_norm, y_norm, t, ts),
        FrequencyData(U, Y, f, f_idx, fs),
        Normalizer(u_mean.flatten(), u_std.flatten(),
                   y_mean.flatten(), y_std.flatten())
    )
