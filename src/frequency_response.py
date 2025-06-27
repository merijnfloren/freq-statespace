"""
Frequency Response Analysis for Multi-Input Multi-Output Systems
"""
from dataclasses import dataclass

import numpy as np

from src.data_manager import FrequencyData, InputOutputData, Normalizer


@dataclass(frozen=True)
class FrequencyResponse:
    """Frequency response matrix and metadata."""
    G: np.ndarray  # FRM, shape (F, ny, nu, M, P)
    freq: FrequencyData  # normalized frequency-domain data
    norm: Normalizer  # normalization statistics


def compute_frequency_response(data: InputOutputData) -> FrequencyResponse:
    """
    Compute frequency response matrix G(z) = Y(z) * U(z)^(-1).

    Parameters
    ----------
    data : InputOutputData

    Returns
    -------
    FrequencyResponse
        FRM with metadata.

    Raises
    ------
    ValueError
        If R < nu (insufficient realizations for matrix inversion).
    """
    U = data.freq.U[data.freq.f_idx, ...]
    Y = data.freq.Y[data.freq.f_idx, ...]

    F, nu, R, P = U.shape
    ny = Y.shape[1]

    if R < nu:
        raise ValueError(
            'For multi-input systems, the number of realisations (R) must be '
            'at least equal to the number of inputs (nu) to compute the '
            'frequency response matrix.'
        )

    M = R // nu
    if M * nu != R:
        print(
            'Suboptimal number of realisations: not all realisations are '
            'used to compute the frequency response matrix. Ideally, the '
            'number of realisations (R) should be an integer multiple of '
            'the number of inputs (nu).'
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

    return FrequencyResponse(G, data.freq, data.norm)
