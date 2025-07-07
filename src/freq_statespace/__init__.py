"""Top-level `freq_statespace` package.

This module exposes the main model structures, static nonlinear building blocks,
data utilities, and example preprocessing functions for frequency-domain system
identification.

Available submodules and components:
- `create_data_object`: Utility function to initialize the identification process by
   constructing an `InputOutputData` object.
- `lin`: Tools for creating and optimizing linear BLA models.
- `nonlin`: Tools for creating and optimizing nonlinear LFR models;
- `f_static`: Static nonlinear function approximators and feature maps;
- `ModelBLA`, `ModelNonlinearLFR`: Primary model structure classes;
- `load_and_preprocess_silverbox_data`: Example dataset loading and preprocessing.

"""


from . import _best_linear_approximation as lin
from . import _nonlin_lfr as nonlin
from . import f_static
from ._data_manager import create_data_object
from ._misc import load_and_preprocess_silverbox_data
from ._model_structures import ModelBLA, ModelNonlinearLFR


__all__ = [
    "lin",
    "nonlin",
    "f_static",
    "load_and_preprocess_silverbox_data",
    "create_data_object",
    "ModelBLA",
    "ModelNonlinearLFR"
]
