"""Data structures in time and frequency domains, including metadata."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from best_linear_approximation._exceptions import (
    InsufficientExperimentsError,
    NoiseCovarianceUnavailableWarning,
    TotalCovarianceUnavailableWarning,
)
from best_linear_approximation.robust import noisy_input
from best_linear_approximation._signal_validation import (
    MatchingAxes,
    SignalContract,
    SignalRanks,
    _validate_matching_axes,
)
from best_linear_approximation._spectra import compute_noise_covariance
from best_linear_approximation._spectral_validation import (
    resolve_excited_bins,
    validate_sampling_frequency,
)
from best_linear_approximation._uncertainty import Uncertainty

from freq_statespace._config import DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS

if TYPE_CHECKING:
    from jaxtyping import Array, Complex, Float
    from numpy.typing import NDArray

    from freq_statespace._typing import ComplexArray, RealArray


SIGNAL_CONTRACT = SignalContract(
    ranks=SignalRanks(r=None, u=4, y=4),
    matching_axes=(MatchingAxes(signals=("u", "y"), axes=(0, 2, 3)),),
)


@dataclass(frozen=True)
class TimeData:
    """Normalized time-domain data container.

    Attributes
    ----------
    u : Float[Array, "n_samples nu n_realizations"]
        Normalized input signals, averaged over periods.
    y : Float[Array, "n_samples ny n_realizations"]
        Normalized output signals, averaged over periods.
    t : RealArray, shape (n_samples,)
        Time vector of a single period.
    ts : float
        Sampling time in seconds.

    """

    u: Float[Array, "n_samples nu n_realizations"]
    y: Float[Array, "n_samples ny n_realizations"]
    t: RealArray
    ts: float


@dataclass(frozen=True)
class NonparametricBLA:
    """Nonparametric Best Linear Approximation (BLA) and distortion levels.

    Attributes
    ----------
    G : ComplexArray
        Nonparametric BLA estimate, shape (n_excited_bins, ny, nu).
    var_noise : RealArray, shape (n_excited_bins, ny, nu), optional
        Measurement-noise variance of the nonparametric BLA estimate.
    var_tot : RealArray, shape (n_excited_bins, ny, nu), optional
        Total variance of the nonparametric BLA estimate.

    """

    G: ComplexArray
    var_noise: RealArray | None
    var_tot: RealArray | None


@dataclass(frozen=True)
class FrequencyData:
    """Normalized frequency-domain data container.

    Attributes
    ----------
    G_bla : `NonparametricBLA`, optional
        Nonparametric BLA estimate with variance estimates. Is `None` if
        insufficient realizations are available to compute the frequency
        response matrix (i.e., `n_realizations < nu`).
    U : Complex[Array, "n_bins nu n_realizations"]
        Normalized input DFT, averaged over periods.
    Y : Complex[Array, "n_bins ny n_realizations"]
        Normalized output DFT, averaged over periods.
    Y_var_noise : Float[Array, "n_bins ny"], optional
        Estimated output measurement-noise variance. Is `None` when unavailable.
    freqs : RealArray, shape (n_samples // 2 + 1,)
        Complete frequency vector.
    excited_bins : NDArray[np.int_], shape (n_bins,)
        Excited frequency indices.
    fs : float
        Sampling frequency in Hz.

    """

    G_bla: NonparametricBLA | None
    U: Complex[Array, "n_bins nu n_realizations"]
    Y: Complex[Array, "n_bins ny n_realizations"]
    Y_var_noise: Float[Array, "n_bins ny"] | None
    freqs: RealArray
    excited_bins: NDArray[np.int_]
    fs: float


class Normalizer(eqx.Module):
    """Normalization statistics for input/output signals.

    Attributes
    ----------
    u_mean : RealArray, shape (nu,)
        Input means.
    u_std : RealArray, shape (nu,)
        Input standard deviations.
    y_mean : RealArray, shape (ny,)
        Output means.
    y_std : RealArray, shape (ny,)
        Output standard deviations.

    """

    u_mean: RealArray = eqx.field(converter=np.asarray)
    u_std: RealArray = eqx.field(converter=np.asarray)
    y_mean: RealArray = eqx.field(converter=np.asarray)
    y_std: RealArray = eqx.field(converter=np.asarray)


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
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float = DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS,
) -> InputOutputData:
    """Create InputOutputData object from time-domain signals and frequency metadata.

    Parameters
    ----------
    u : RealArray
        Input time series, a (random-phase) (multi)sine with shape
        ``(n_samples, nu, n_realizations, n_periods)``, where:
        - ``n_samples``: number of samples per period;
        - ``nu``: number of input channels;
        - ``n_realizations``: number of independent phase realizations. Each
            realization must have the same frequency content and amplitude
            characteristics;
        - ``n_periods``: number of periods (copies of the same realization).
        It is fine if only a single realization and/or period is provided,
        but is is important to shape the data in the required 4D format.
    y : RealArray
        Steady-state output time series with shape
        ``(n_samples, ny, n_realizations, n_periods)``,  where ``ny`` is the
        number of output channels.
    fs : float
        Sampling frequency in Hz.
    excited_bins : NDArray[np.int_] or float, optional
        If provided as an array, specifies strictly increasing indices of the
        excited non-DC, non-Nyquist ``rfft`` bins. Otherwise, if provided as a
        float in ``(0, 1)``, selects them automatically from ``u`` using the
        mean spectral magnitude across normalized input channels. Bins exceeding
        this fraction of the maximum magnitude are selected.

    Returns
    -------
    InputOutputData
        Processed (meta)data in time and frequency domains.

    """
    u, y, fs, excited_bins = _validate_input_arguments(u, y, fs, excited_bins)

    ts = 1 / fs
    n_samples = u.shape[0]
    t = np.arange(n_samples) * ts

    # Normalize data (zero mean, unit variance)
    u_mean = np.mean(u, axis=(0, 2, 3), keepdims=True)
    y_mean = np.mean(y, axis=(0, 2, 3), keepdims=True)
    u_std = np.std(u, axis=(0, 2, 3), keepdims=True)
    y_std = np.std(y, axis=(0, 2, 3), keepdims=True)

    u = (u - u_mean) / u_std
    y = (y - y_mean) / y_std

    G_bla = _create_nonparametric_bla(u, y, fs, excited_bins)

    # Compute DFTs
    U = np.fft.rfft(u, axis=0)
    Y = np.fft.rfft(y, axis=0)
    freqs = np.arange(n_samples // 2 + 1) * fs / n_samples
    Y_avg = np.mean(Y, axis=3)
    Y_var_noise = _compute_output_noise_variance(Y)

    # We proceed with data that is averaged over periods
    u_avg, y_avg = np.mean(u, axis=3), np.mean(y, axis=3)
    U_avg = np.mean(U, axis=3)

    # Finally, we convert the input-output data to JAX arrays
    u_avg, y_avg = jnp.asarray(u_avg), jnp.asarray(y_avg)
    U_avg, Y_avg = jnp.asarray(U_avg), jnp.asarray(Y_avg)

    return InputOutputData(
        TimeData(u_avg, y_avg, t, ts),
        FrequencyData(G_bla, U_avg, Y_avg, Y_var_noise, freqs, excited_bins, fs),
        Normalizer(u_mean.flatten(), u_std.flatten(),
                   y_mean.flatten(), y_std.flatten())
    )


def _validate_input_arguments(
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_] | float,
) -> tuple[RealArray, RealArray, float, NDArray[np.int_]]:
    """Validate and resolve data-creation arguments."""
    u, y = np.asarray(u), np.asarray(y)
    _validate_signal_contract(u, y)
    fs = validate_sampling_frequency(fs)
    excited_bins = resolve_excited_bins(excited_bins, u, fs)
    return u, y, fs, excited_bins


def _create_nonparametric_bla(
    u: RealArray,
    y: RealArray,
    fs: float,
    excited_bins: NDArray[np.int_],
) -> NonparametricBLA | None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NoiseCovarianceUnavailableWarning)
            warnings.filterwarnings("ignore", category=TotalCovarianceUnavailableWarning)
            bla = noisy_input(u, y, fs, excited_bins)
    except InsufficientExperimentsError:
        print(
            "Warning: Insufficient realizations (n_realizations < nu) to compute the "
            "nonparametric BLA. Identification can proceed in input-output mode, but "
            "the initial linear model may be suboptimal.",
        )
        return None

    var_noise = bla.G.noise.var
    var_tot = bla.G.total.var
    return NonparametricBLA(
        G=jnp.asarray(bla.G.value),
        var_noise=jnp.asarray(var_noise) if var_noise is not None else None,
        var_tot=jnp.asarray(var_tot) if var_tot is not None else None,
    )


def _compute_output_noise_variance(
    Y: Complex[Array, "n_bins ny n_realizations n_periods"],
) -> Float[Array, "n_bins ny"] | None:
    Y_with_input_axis = Y[:, :, None, :, :]
    Y_noise_cov = compute_noise_covariance(Y_with_input_axis)
    if Y_noise_cov is None:
        return None

    Y_var_noise = Uncertainty.from_cov(Y_noise_cov, (Y.shape[1],)).var
    return jnp.asarray(Y_var_noise)


def _validate_signal_contract(u: RealArray, y: RealArray) -> None:
    """Validate that the input and output signals have the correct dimensions."""
    if u.ndim != 4 or y.ndim != 4:
        raise ValueError(
            f"Input and output signals must be 4D arrays, got shapes u={u.shape} and y={y.shape}."
        )

    arrays_by_signal = {"u": u, "y": y}
    _validate_matching_axes(arrays_by_signal, SIGNAL_CONTRACT)
