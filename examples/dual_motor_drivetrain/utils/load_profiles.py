"""Load-profile generators for the closed-loop identification example."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_random_load_profile(
    n_samples: int,
    fs: float,
    amplitude: float,
    correlation_time: float,
    seed: int | None,
    n_realizations: int = 1,
) -> NDArray[np.float64]:
    """Generate a piecewise-constant random load-torque profile.

    The duration of each constant level is exponentially distributed with
    mean ``correlation_time``. The levels themselves are sampled uniformly
    from ``[-amplitude, amplitude]``.

    Args:
        n_samples: Number of samples in each profile.
        fs: Sampling frequency in hertz.
        amplitude: Maximum magnitude of the load torque.
        correlation_time: Mean duration, in seconds, of a constant level.
        seed: Seed for the random-number generator.
        n_realizations: Number of independently generated profiles.

    Returns:
        Load-torque profiles with shape ``(n_samples, n_realizations)``.
    """
    if n_samples < 0:
        msg = "n_samples must be non-negative"
        raise ValueError(msg)
    if n_realizations < 1:
        msg = "n_realizations must be at least one"
        raise ValueError(msg)
    if fs <= 0.0:
        msg = "fs must be positive"
        raise ValueError(msg)
    if correlation_time < 0.0:
        msg = "correlation_time must be non-negative"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    if n_samples < 2:
        return amplitude * (2.0 * rng.random((n_samples, n_realizations)) - 1.0)

    tau_load = np.zeros((n_samples, n_realizations))
    dt = 1.0 / fs

    for realization in range(n_realizations):
        index = 0
        while index < n_samples:
            hold_samples = max(
                1,
                int(np.ceil(-correlation_time / dt * np.log(rng.random()))),
            )
            next_index = min(n_samples, index + hold_samples)

            level = amplitude * (2.0 * rng.random() - 1.0)
            tau_load[index:next_index, realization] = level
            index = next_index

    return tau_load
