"""Submodule `static`.

This submodule provides the building blocks for the static nonlinear function.

Exposed components:
- `basis`: Feature maps (mapping `z` to `features`) that are linear in the parameters.
   They can be used for inference and learning and nonlinear optimization.
- `BasisFunctionModel`: Static nonlinear function (mapping `z` to `w`) based on a
   feature map.
- `NeuralNetwork`:  Static nonlinear function (mapping `z` to `w`) based on a neural
   network.
"""
from freq_statespace.static import _feature_maps as basis
from freq_statespace.static._nonlin_funcs import BasisFunctionModel, NeuralNetwork


__all__ = ["basis", "BasisFunctionModel", "NeuralNetwork"]
