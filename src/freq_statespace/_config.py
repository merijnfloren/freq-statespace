"""Centralized default constants and types used across the package."""

from typing import Literal

import jax
import optimistix as optx


DeviceName = Literal["cpu", "gpu", "tpu"]
DeviceLike = DeviceName | jax.Device | None

PRINT_EVERY = 1
SEED = 42
SOLVER = optx.LevenbergMarquardt(rtol=1e-3, atol=1e-6)
