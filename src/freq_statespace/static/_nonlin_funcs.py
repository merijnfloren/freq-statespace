"""General static nonlinear function mappings (mapping `z` to `w`)."""

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from .. import _misc
from .._config import SEED
from ..static._feature_maps import AbstractFeatureMap


class AbstractNonlinearFunction(eqx.Module):
    """Abstract base class for nonlinear function mappings.

    Subclasses must provide the attributes `nw`, `nz`, `seed`, and
    `num_parameters`, and implement the method `_evaluate()`.
    """

    nw: eqx.AbstractVar[int]
    nz: eqx.AbstractVar[int]
    seed: eqx.AbstractVar[int]
    num_parameters: eqx.AbstractVar[int]

    @abstractmethod
    def _evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the nonlinear function.

        From inputs of shape (..., `nz`) to outputs of shape (..., `nw`).
        """
        ...


class BasisFunctionModel(AbstractNonlinearFunction):
    """Static nonlinear function based on an `AbstractFeatureMap`.

    This class combines an `AbstractFeatureMap` with a coefficient matrix `beta`
    to implement and evaluate a static nonlinear mapping that is linear in its
    parameters.
    """

    nw: int
    nz: int
    beta: jnp.ndarray
    phi: AbstractFeatureMap
    num_parameters: int
    seed: int = eqx.field(repr=False)

    def __init__(
        self,
        nw: int,
        phi: AbstractFeatureMap,
        seed: int = SEED,
    ) -> None:
        """Initialize a static nonlinear basis-function model.

        Parameters
        ----------
        nw : int
            Number of output features (dimension of latent signal `w`).
        phi : AbstractFeatureMap
            Nonlinear feature map that is linear in the parameters.
        seed : int, optional
            Used for randomly initializing (i) the nonlinear coefficient
            matrix `beta` and (ii) the matrices `B_w`, `C_z`, `D_yw`, and
            `D_zu` (initialized externally, not by this class). Defaults
            to `SEED`.

        """
        self.nw = nw
        self.phi = phi
        self.nz = phi.nz
        self.seed = seed

        self.beta = jax.random.uniform(
            key=_misc.get_key(self.seed, "basis_function_model"),
            shape=(self.phi.num_features, self.nw),
            minval=-1.0,
            maxval=1.0,
        )
        self.num_parameters = self.beta.size

    def _evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.phi._compute_features(z) @ self.beta


class NeuralNetwork(AbstractNonlinearFunction):
    """Fully connected feedforward neural network.

    This class wraps an `eqx.nn.MLP` and exposes a simple interface for
    evaluating the network.
    """

    nw: int
    nz: int
    model: eqx.nn.MLP
    num_parameters: int
    layers: int = eqx.field(repr=False)
    neurons_per_layer: int = eqx.field(repr=False)
    activation: Callable = eqx.field(repr=False)
    seed: int = eqx.field(repr=False)
    bias: bool = eqx.field(repr=False)
    

    def __init__(
        self,
        nz: int,
        nw: int,
        layers: int,
        neurons_per_layer: int,
        activation: Callable,
        seed: int = SEED,
        bias: bool = True,
    ) -> None:
        """Initialize a fully connected feedforward neural network.

        Parameters
        ----------
        nz : int
            Number of input features (dimension of latent signal `z`).
        nw : int
            Number of output features (dimension of latent signal `w`).
        layers : int
            Number of hidden layers.
        neurons_per_layer : int
            Number of neurons per hidden layer.
        activation : Callable, from `jax.nn`
            Activation function used in hidden layers.
        seed : int, optional
            Used for randomly initializing (i) the neural network parameters and
            (ii) the matrices `B_w`, `C_z`, `D_yw`, and `D_zu` (initialized
            externally, not by this class). Defaults to `SEED`.
        bias : bool, optional
            Whether to include bias terms. Defaults to `True`.

        """
        self.nz = nz
        self.nw = nw
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.seed = seed
        self.bias = bias

        self.model = eqx.nn.MLP(
            in_size=self.nz,
            out_size=self.nw,
            width_size=self.neurons_per_layer,
            depth=self.layers,
            activation=self.activation,
            use_bias=self.bias,
            key=_misc.get_key(self.seed, "neural_network"),
        )
        self.num_parameters = sum(
            x.size
            for x in jax.tree_util.tree_leaves(self.model)
            if isinstance(x, jax.Array)
        )

    def _evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.model)(z)
