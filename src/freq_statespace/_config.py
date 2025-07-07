"""Centralized default constants used across the package."""

import optimistix as optx


SEED = 42
SOLVER = optx.LevenbergMarquardt(rtol=1e-3, atol=1e-6)
