"""Centralized default constants and types used across the package."""
from typing import Literal

import jax
import optimistix as optx


DeviceName = Literal["cpu", "gpu", "tpu"]
DeviceLike = DeviceName | jax.Device | None


PRINT_EVERY = 1  # keep in sync with docstrings
SEED = 42  # keep in sync with docstrings
STABILITY_MARGIN = 1e-4  # keep in sync with docstrings
SOLVER = optx.LevenbergMarquardt(rtol=1e-3, atol=1e-6)  # keep in sync with docstrings
DEFAULT_RELATIVE_THRESHOLD_EXCITED_BINS = 0.5  # keep in sync with docstrings