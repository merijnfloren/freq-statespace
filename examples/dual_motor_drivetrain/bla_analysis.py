"""Plot nonparametric BLA estimates and their spectra."""

# ruff: noqa: INP001
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from numpy.typing import NDArray

    from best_linear_approximation import (
        FrequencyResponse,
        InputSpectrum,
        NonparametricBLA,
        OutputSpectrum,
    )
    from best_linear_approximation._typing import RealArray


def plot_bla_analysis(
    result: NonparametricBLA,
) -> None:
    """Create separate BLA, input-spectrum, and output-spectrum figures."""
    with plt.rc_context({"text.usetex": False}):
        _plot_bla(result)
        _plot_spectra(result, result.spectra.U, "input spectrum")
        _plot_spectra(result, result.spectra.Y, "output spectrum")
        plt.show()


def _plot_bla(result: NonparametricBLA) -> None:
    """Plot the BLA magnitude and phase matrices."""
    ny, nu = result.G.value.shape[1:]
    with plt.rc_context({"figure.constrained_layout.h_pad": 0.08}):
        figure, axes = plt.subplots(
            2 * ny,
            nu,
            figsize=(4.0 * nu, 5.3 * ny + 0.7),
            layout="constrained",
            squeeze=False,
            sharex=True,
        )
    title = figure.suptitle("best linear approximation", fontsize=16, fontweight="bold")
    excited_freqs = result.freq.freqs[result.freq.excited_bins]
    for output_channel in range(ny):
        for input_channel in range(nu):
            magnitude_axis = axes[output_channel, input_channel]
            phase_axis = axes[ny + output_channel, input_channel]
            indices = (output_channel, input_channel)
            _plot_level(
                magnitude_axis,
                excited_freqs,
                np.abs(result.G.value[:, *indices]),
                "magnitude",
                color="black",
            )
            _plot_uncertainties(magnitude_axis, result.G, result, indices)
            phase_axis.plot(
                excited_freqs,
                np.angle(result.G.value[:, *indices], deg=True),
                color="black",
            )
            if output_channel == 0:
                magnitude_axis.set_title(
                    rf"from input $\mathbf{{u}}_{{{input_channel + 1}}}$",
                    fontweight="bold",
                )
            if input_channel == 0:
                magnitude_axis.set_ylabel(
                    rf"$\mathbf{{to\ output\ }}\mathbf{{y}}_{{{output_channel + 1}}}$"
                    "\n\n"
                    "gain [dB]",
                )
                phase_axis.set_ylabel(
                    rf"$\mathbf{{to\ output\ }}\mathbf{{y}}_{{{output_channel + 1}}}$"
                    "\n\n"
                    "phase [deg]",
                )

    _finish_figure(axes, result, full_frequency_band=False)
    _center_title(figure, title)


def _plot_spectra(
    result: NonparametricBLA,
    spectrum: InputSpectrum | OutputSpectrum,
    figure_title: str,
) -> None:
    """Plot averaged spectra in one simple row or column."""
    n_channels = spectrum.value.shape[1]
    is_input = figure_title == "input spectrum"
    n_rows, n_columns = 1, n_channels
    figure_width = 4.0 * n_columns
    with plt.rc_context(
        {
            "figure.constrained_layout.h_pad": 0.15,
            "figure.constrained_layout.hspace": 0.1,
            "figure.constrained_layout.wspace": 0.1,
        },
    ):
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(figure_width, 2.8 * n_rows + 0.7),
            layout="constrained",
            squeeze=False,
            sharex=True,
            sharey=True,
        )
    title = figure.suptitle(figure_title, fontsize=16, fontweight="bold")
    flat_axes = axes.ravel()
    sample_axes = tuple(range(2, spectrum.value.ndim))
    magnitude = np.sqrt(np.mean(np.abs(spectrum.value) ** 2, axis=sample_axes))
    signal_name, symbol = ("input", "u") if is_input else ("output", "y")
    for channel, axis in enumerate(flat_axes):
        _plot_level(
            axis,
            result.freq.freqs,
            magnitude[:, channel],
            "spectrum (rms)",
            color="black",
        )
        _plot_uncertainties(axis, spectrum, result, (channel,))
        axis.set_title(
            rf"{signal_name} $\mathbf{{{symbol}}}_{{{channel + 1}}}$",
            fontsize=10,
            fontweight="bold",
        )
        if axis is flat_axes[0]:
            axis.set_ylabel("DFT-bin magnitude [dB]")

    _finish_figure(axes, result, full_frequency_band=True)
    _center_title(figure, title)


def _finish_figure(
    axes: NDArray[Any],
    result: NonparametricBLA,
    *,
    full_frequency_band: bool,
) -> None:
    """Apply shared labels, style, and legend to a figure."""
    frequency_range = (
        result.freq.freqs if full_frequency_band else result.freq.freqs[result.freq.excited_bins]
    )
    positive_freqs = frequency_range[frequency_range > 0]
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


def _center_title(figure: Figure, title: Text) -> None:
    """Center a figure title over its grid of axes."""
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.execute(figure)
    left = figure.axes[0].get_position().x0
    right = figure.axes[-1].get_position().x1
    title.set_x((left + right) / 2)


def _plot_uncertainties(
    axis: Axes,
    estimate: FrequencyResponse | InputSpectrum | OutputSpectrum,
    result: NonparametricBLA,
    indices: tuple[int, ...],
) -> None:
    """Plot the available marginal uncertainty standard deviations."""
    excited_freqs = result.freq.freqs[result.freq.excited_bins]
    styles = (
        ("noise", "noise std", "--"),
        ("total", "total std", "-."),
    )
    for name, label, linestyle in styles:
        uncertainty = getattr(estimate, name, None)
        standard_deviation = None if uncertainty is None else uncertainty.std
        if standard_deviation is None:
            continue
        freqs = result.freq.freqs if name == "noise" and estimate is not result.G else excited_freqs
        _plot_level(
            axis,
            freqs,
            standard_deviation[:, *indices],
            label,
            linestyle=linestyle,
        )


def _plot_level(  # noqa: PLR0913
    axis: Axes,
    freqs: RealArray,
    magnitude: RealArray,
    label: str,
    *,
    color: str | None = None,
    linestyle: str = "-",
) -> None:
    """Plot a finite, positive magnitude in dB."""
    valid = (magnitude > 0) & np.isfinite(magnitude)
    levels = np.full(magnitude.shape, np.nan)
    levels[valid] = 20 * np.log10(magnitude[valid])
    axis.plot(
        freqs,
        levels,
        label=label,
        color=color,
        linestyle=linestyle,
    )
