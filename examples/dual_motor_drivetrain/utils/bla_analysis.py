"""Plot a nonparametric BLA beside a parametric frequency response."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from best_linear_approximation import FrequencyResponse, NonparametricBLA
    from best_linear_approximation._typing import RealArray
    from freq_statespace import ModelBLA
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def plot_bla_analysis(
    nonparametric_bla: NonparametricBLA,
    parametric_bla: ModelBLA,
) -> None:
    """Plot gain FRFs, uncertainty estimates, and residuals for every I/O pair.

    The parametric response is evaluated with the public
    :meth:`ModelBLA.frequency_response` method at exactly the excited BLA
    frequency lines.
    """
    excited_freqs = nonparametric_bla.freq.freqs[nonparametric_bla.freq.excited_bins]
    nonparametric_response = np.asarray(nonparametric_bla.G.value)
    parametric_response = np.asarray(parametric_bla.frequency_response(excited_freqs))
    if parametric_response.shape != nonparametric_response.shape:
        msg = "The parametric model response must have the same shape as the BLA response."
        raise ValueError(msg)

    ny, nu = nonparametric_response.shape[1:]
    is_multivariate = ny > 1 or nu > 1
    with plt.rc_context({"text.usetex": False}):
        figure, axes = plt.subplots(
            ny,
            nu,
            figsize=(5 * nu, 3.2 * ny + 0.7),
            layout="constrained",
            squeeze=False,
            sharex=True,
        )
        figure.suptitle("BLA comparison", fontsize=16, fontweight="bold")

        for output_channel in range(ny):
            for input_channel in range(nu):
                axis = axes[output_channel, input_channel]
                indices = (output_channel, input_channel)
                nonparametric_gain = np.abs(nonparametric_response[:, *indices])
                parametric_gain = np.abs(parametric_response[:, *indices])
                residual = np.abs(
                    parametric_response[:, *indices] - nonparametric_response[:, *indices]
                )

                _plot_level(
                    axis,
                    excited_freqs,
                    nonparametric_gain,
                    "nonparametric BLA",
                    color="black",
                )
                _plot_uncertainties(axis, nonparametric_bla.G, nonparametric_bla, indices)
                _plot_level(axis, excited_freqs, parametric_gain, "parametric model")
                _plot_level(axis, excited_freqs, residual, "model residual", linestyle=":")

                if is_multivariate and output_channel == 0:
                    axis.set_title(
                        rf"from input $\mathbf{{u}}_{{{input_channel + 1}}}$",
                        fontweight="bold",
                    )
                if input_channel == 0:
                    ylabel = (
                        rf"to output $\mathbf{{y}}_{{{output_channel + 1}}}$"
                        "\n\nmagnitude [dB]"
                        if is_multivariate
                        else "magnitude [dB]"
                    )
                    axis.set_ylabel(ylabel)

        _finish_figure(axes, excited_freqs)
        plt.show()


def _plot_uncertainties(
    axis: Axes,
    estimate: FrequencyResponse,
    result: NonparametricBLA,
    indices: tuple[int, int],
) -> None:
    """Plot the available total BLA uncertainty magnitude."""
    excited_freqs = result.freq.freqs[result.freq.excited_bins]
    standard_deviation = estimate.total.std
    if standard_deviation is not None:
        _plot_level(
            axis,
            excited_freqs,
            standard_deviation[:, *indices],
            "total std",
            linestyle="-.",
        )


def _finish_figure(axes: NDArray[Any], excited_freqs: RealArray) -> None:
    """Apply shared BLA-plot styling and a single legend."""
    positive_freqs = excited_freqs[excited_freqs > 0]
    if positive_freqs.size == 0:
        msg = "a logarithmic frequency axis requires at least one positive frequency"
        raise ValueError(msg)
    min_freq, max_freq = positive_freqs[0], positive_freqs[-1]
    if min_freq == max_freq:
        min_freq, max_freq = min_freq / 1.5, max_freq * 1.5

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlim(min_freq, max_freq)
        axis.grid(visible=False)
        axis.label_outer()
    for axis in axes[-1, :]:
        axis.set_xlabel("frequency [Hz]")
    axes.flat[0].legend()


def _plot_level(
    axis: Axes,
    freqs: RealArray,
    magnitude: RealArray,
    label: str,
    *,
    color: str | None = None,
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
) -> None:
    """Plot a finite, positive magnitude in dB."""
    valid = (magnitude > 0) & np.isfinite(magnitude)
    levels = np.full(magnitude.shape, np.nan)
    levels[valid] = 20 * np.log10(magnitude[valid])
    axis.plot(freqs, levels, label=label, color=color, linestyle=linestyle)
