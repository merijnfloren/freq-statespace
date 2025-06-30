"""
Static nonlinear functions in NL-LFR models.

This module defines abstract and concrete classes for nonlinear functions,
currently including basis function models and neural network models.
"""
from abc import abstractmethod
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from src.basis_functions import AbstractBasisFunction


class AbstractNonlinearFunction(eqx.Module, strict=True):
    """
    Abstract base class for nonlinear functions.
    This class serves as a blueprint for defining nonlinear functions with
    specific input and output dimensions.
    It requires subclasses to implement the `evaluate` and `num_parameters`
    methods.

    Attributes
    ----------
        nz (eqx.AbstractVar[int]): The input dimension of the nonlinear
                                   function.
        nw (eqx.AbstractVar[int]): The output dimension of the nonlinear
                                   function.
    """
    nz: eqx.AbstractVar[int]
    nw: eqx.AbstractVar[int]

    @abstractmethod
    def evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        """From size (-1, nz) to size (-1, nw)."""
        pass

    @abstractmethod
    def num_parameters(self) -> int:
        """Returns the total number of model parameters."""
        pass


class BasisFunctionModel(AbstractNonlinearFunction, strict=True):
    """
    BasisFunctionModel is a class that represents a nonlinear function model
    based on basis functions.

    Parameters
    ----------
    nw : int
        Output dimension of the function.
    phi : AbstractBasisFunction
        Basis function object that computes features from input data.
    key : PRNGKeyArray
        Random key used for initializing the weight matrix beta.
    """
    nz: int = eqx.field(init=False)
    nw: int
    beta: jnp.array = eqx.field(init=False)
    phi: AbstractBasisFunction
    key: PRNGKeyArray = eqx.field(repr=False)

    # Internal attribute (computed within the class)
    _num_parameters: int = eqx.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.nz = self.phi.nz
        self.beta = jax.random.uniform(
            self.key,
            (self.phi.num_features(), self.nw),
            minval=-1,
            maxval=1
        )
        self._num_parameters = self.beta.size

    def evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.phi.compute_features(z) @ self.beta

    def num_parameters(self) -> int:
        return self._num_parameters


class NeuralNetwork(AbstractNonlinearFunction, strict=True):
    """
    Nonlinear static function model based on a fully connected neural network.

    Parameters
    ----------
    nz : int
        Input dimension of the nonlinear function.
    nw : int
        Output dimension of the nonlinear function.
    num_layers : int
        Number of hidden layers in the MLP.
    num_neurons_per_layer : int
        Number of neurons per hidden layer.
    activation : Callable
        Activation function used in each hidden layer.
    key : PRNGKeyArray
        JAX PRNG key for random weight initialization.
    bias : bool, optional
        Whether to include bias terms in each layer (default: True).
    """
    nz: int
    nw: int
    num_layers: int = eqx.field(repr=False)
    num_neurons_per_layer: int = eqx.field(repr=False)
    activation: Callable = eqx.field(repr=False)
    key: PRNGKeyArray = eqx.field(repr=False)
    bias: bool = eqx.field(default=True, repr=False)
    model: eqx.nn.MLP = eqx.field(init=False)

    # Internal attribute (computed within the class)
    _num_parameters: int = eqx.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = eqx.nn.MLP(
            in_size=self.nz,
            out_size=self.nw,
            width_size=self.num_neurons_per_layer,
            depth=self.num_layers,
            activation=self.activation,
            use_bias=self.bias,
            key=self.key
        )
        self._num_parameters = sum(
            x.size for x in jax.tree_util.tree_leaves(self.model)
            if isinstance(x, jax.Array)
        )

    def evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.model)(z)

    def num_parameters(self) -> int:
        return self._num_parameters


def create_custom_basis_function_model(
    nw: int,
    phi: AbstractBasisFunction,
    beta: jnp.ndarray
) -> BasisFunctionModel:
    """
    Create a `BasisFunctionModel` instance with a user-defined weight matrix.

    This function bypasses random initialization of `beta` by replacing it
    with the provided matrix. A dummy key is still required for initialization,
    but it does not affect the final result.

    Parameters
    ----------
    nw : int
        Output dimension of the nonlinear function.
    phi : AbstractBasisFunction
        Basis function object that defines the feature mapping.
    beta : jnp.ndarray
        Custom weight matrix, shape (n_features, nw), where `n_features`
        is given by `phi.num_features()`.

    Returns
    -------
    BasisFunctionModel
        Instantiated model using the provided weight matrix.
    """
    dummy_key = jax.random.key(0)
    return eqx.tree_at(
        where=lambda tree: tree.beta,
        pytree=BasisFunctionModel(nw, phi, dummy_key),
        replace=beta
    )
