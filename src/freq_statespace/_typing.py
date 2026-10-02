from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating[Any, Any]]
"""An array with a complex floating-point dtype."""


RealArray = NDArray[np.floating[Any]]
"""An array with a real floating-point dtype."""
