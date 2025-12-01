"""Top-level package for frequency-domain system identification.

The API is structured around three main namespaces:

- ``lin``:
    Tools for creating and optimizing BLA models.

- ``nonlin``:
    Tools for creating and optimizing nonlinear LFR models.

- ``static``:
    Static nonlinear function approximators and feature-map constructions.

Additional components:
- ``create_data_object`` for constructing ``InputOutputData`` objects, the starting
    point for all identification routines.
-  ``InputOutputData``, ``TimeData``, ``FrequencyData``, ``NonparametricBLA``,
    and ``Normalizer`` data objects.
- ``ModelBLA`` and ``ModelNonlinearLFR`` as the core model classes.
- ``SolveResult`` for detailed information about optimization routines.
- ``load_and_preprocess_silverbox_data`` for loading the Silverbox example dataset. 
    This is useful for quickly loading a low-dimensional benchmark dataset in the
    required format.
"""


from . import _best_linear_approximation as lin
from . import _nonlin_lfr as nonlin
from . import static
from ._data_manager import (
    FrequencyData,
    InputOutputData,
    NonparametricBLA,
    Normalizer,
    TimeData,
    create_data_object,
)
from ._misc import load_and_preprocess_silverbox_data
from ._model_structures import ModelBLA, ModelNonlinearLFR
from ._solve import SolveResult


__all__ = [
    "lin",
    "nonlin",
    "static",
    "FrequencyData",
    "InputOutputData",
    "NonparametricBLA",
    "Normalizer",
    "TimeData",
    "create_data_object",
    "load_and_preprocess_silverbox_data",
    "ModelBLA",
    "ModelNonlinearLFR",
    "SolveResult"
]
