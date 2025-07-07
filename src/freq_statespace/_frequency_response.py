"""Frequency response analysis for multi-input multi-output systems."""

import numpy as np


def compute_frequency_response(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute frequency response matrix G(k) = Y(k) * (U(k))^(-1).

    Parameters
    ----------
    U : np.ndarray, shape (F, nu, R, P)
        DFT input spectra at the excited frequencies across realizations and
        periods.
    Y : np.ndarray, shape (F, ny, R, P)
        DFT output spectra at the excited frequencies across realizations and
        periods.

    Returns
    -------
    G : np.ndarray, shape (F, ny, nu, M, P)
        Frequency Response Matrix:
        - F: number of frequency bins;
        - ny: number of outputs;
        - nu: number of inputs;
        - M: number of experiments (R // nu);
        - P: number of periods.

    """
    F, nu, R, P = U.shape
    ny = Y.shape[1]

    if R < nu:
        raise ValueError(
            "For multi-input systems, the number of realizations (R) must be "
            "at least equal to the number of inputs (nu) to compute the "
            "frequency response matrix."
        )

    M = R // nu
    if M * nu != R:
        print(
            "Suboptimal number of realizations: not all realizations are "
            "used to compute the frequency response matrix. Ideally, the "
            "number of realizations (R) should be an integer multiple of "
            "the number of inputs (nu)."
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

    return G
