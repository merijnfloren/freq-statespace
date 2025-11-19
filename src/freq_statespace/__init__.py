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
- ``ModelBLA`` and ``ModelNonlinearLFR`` as the core model classes.
- ``load_and_preprocess_silverbox_data`` for loading the Silverbox example dataset.
"""


from . import _best_linear_approximation as lin
from . import _nonlin_lfr as nonlin
from . import static
from ._data_manager import create_data_object
from ._misc import load_and_preprocess_silverbox_data
from ._model_structures import ModelBLA, ModelNonlinearLFR


__all__ = [
    "lin",
    "nonlin",
    "static",
    "load_and_preprocess_silverbox_data",
    "create_data_object",
    "ModelBLA",
    "ModelNonlinearLFR"
]
