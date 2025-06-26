from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.frequency_response import compute_frequency_response
from src.training_data import InputOutputData, Normalizer
from src._model_structures import ModelBLA


@dataclass(frozen=True)
class NonparametricBLA:
    """Nonparametric Best Linear Approximation and metadeta."""
    G: np.ndarray  # BLA, shape (F, ny, nu)
    f: np.ndarray  # full frequency vector, shape (N//2 + 1,)
    f_idx: np.ndarray  # indices of excited frequencies, shape (F,)
    norm: Normalizer  # normalization statistics
    var_noise: Optional[np.ndarray]  # noise variance, shape (F, ny, nu)
    var_tot: Optional[np.ndarray]  # total variance, shape (F, ny, nu)

    def plot(self) -> None:
        """Plot BLA magnitude and distorion levels if available."""
        return _plot(self)


def compute_nonparametric(data: InputOutputData) -> NonparametricBLA:
    """
    Compute BLA and variance estimates from frequency response data.

    Parameters
    ----------
    G : np.ndarray, shape (F, ny, nu, M, P)
        Frequency response matrix over frequencies, outputs, inputs,
        realizations, and periods.

    Returns
    -------
    NonparametricBLA
        BLA with noise and total variance estimates.
    """

    freq_resp = compute_frequency_response(data)
    G = freq_resp.G

    M, P = G.shape[3:5]

    # Average over periods and compute noise variance
    G_P = G.mean(axis=4)
    if P > 1:
        var_noise = (np.abs(G - G_P[..., None])**2).sum(axis=4) / P / (P - 1)
    else:
        var_noise = None

    # Average over realizations and compute total variance
    G_bla = G_P.mean(axis=3)
    if M > 1:
        var_tot = (np.abs(G_P - G_bla[..., None])**2).sum(axis=3) / M / (M - 1)
        if var_noise is not None:
            var_noise = var_noise.mean(axis=3) / M
    else:
        var_tot = None

    return NonparametricBLA(
        G=G_bla,
        f=freq_resp.f,
        f_idx=freq_resp.f_idx,
        norm=freq_resp.norm,
        var_noise=var_noise,
        var_tot=var_tot
    )


def parametrize_with_fsid(bla: NonparametricBLA, nx: int, q: int) -> ModelBLA:
    """
    Initialize with frequency-domain subspace identification.

    Parameters
    ----------
    G_bla : np.ndarray, shape (F, ny, nu)
        Best Linear Approximation
    nx : int
        System order
    q : int
        Past/future horizon

    Returns
    -------
    np.ndarray
        Initial parameter estimates
    """
    pass


def _plot(bla: NonparametricBLA) -> None:
    """
    Plot BLA magnitude with distortion levels.

    Parameters
    ----------
    bla : NonparametricBLA
        BLA results to plot

    Returns
    -------
    None
        This function only produces a plot.
    """
    ny, nu = bla.G.shape[1:3]

    scale_matrix = bla.norm.y_std[:, None] / bla.norm.u_std[None, :]
    G = scale_matrix * bla.G

    # Calculate optimal figure size based on subplot grid
    fig_width = min(5 * nu, 15)  # max 15 inches wide
    fig_height = min(4 * ny, 12)  # max 12 inches tall
    _, axes = plt.subplots(ny, nu, figsize=(fig_width, fig_height))

    # Ensure axes is 2D for consistent indexing
    if ny == 1 and nu == 1:
        axes = np.array([[axes]])
    elif ny == 1 or nu == 1:
        axes = axes.reshape(ny, nu)

    f_data = bla.f[bla.f_idx]

    for i in range(ny):
        for j in range(nu):
            ax = axes[i, j]

            # Plot BLA magnitude
            mag_db = 20 * np.log10(np.abs(G[:, i, j]))
            ax.plot(f_data, mag_db, label='BLA magnitude')

            # Plot distortion levels if available
            if bla.var_noise is not None:
                std_noise = scale_matrix[i, j] * np.sqrt(bla.var_noise[:, i, j])  # noqa: E501
                noise_db = 20 * np.log10(std_noise)
                ax.plot(f_data, noise_db, label='Noise std')

            if bla.var_tot is not None:
                std_tot = scale_matrix[i, j] * np.sqrt(bla.var_tot[:, i, j])
                tot_db = 20 * np.log10(std_tot)
                ax.plot(f_data, tot_db, label='Total std')

            # Format subplot
            ax.set_title(f'G[{i+1},{j+1}]')
            ax.legend()
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Magnitude [dB]')

    plt.tight_layout()
    plt.show()
