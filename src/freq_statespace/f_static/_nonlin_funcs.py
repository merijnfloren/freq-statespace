"""General static nonlinear function mappings (mapping `z` to `w`)."""

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from freq_statespace import _misc
from freq_statespace._config import SEED
from freq_statespace.f_static._feature_maps import AbstractFeatureMap


class AbstractNonlinearFunction(eqx.Module, strict=True):
    """Abstract base class for nonlinear function mappings.

    Subclasses must provide the attributes `nw`, `nz` and `seed`,
    and must implement the methods `_evaluate` and `num_parameters`.
    """

    nw: eqx.AbstractVar[int]
    nz: eqx.AbstractVar[int]
    seed: eqx.AbstractVar[int]

    @abstractmethod
    def _evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        """From size (-1, `nz`) to size (-1, `nw`)."""
        pass

    @abstractmethod
    def num_parameters(self) -> int:
        """Return the total number of model parameters."""
        pass


class BasisFunctionModel(AbstractNonlinearFunction, strict=True):
    """Static nonlinear function based on an `AbstractFeatureMap`.

    Attributes
    ----------
    nw : int
        Number of output features (dimension of latent signal `w`).
    phi : `AbstractFeatureMap`
        Basis function that is linear in the parameters.
    seed : int
        Used for randomly initializing:
        1. Nonlinear coefficient matrix `beta`;
        2. The matrices `B_w`, `C_z`, `D_yw`, and `D_zu` (initialized
           externally, not by this class).

    """

    nw: int
    phi: AbstractFeatureMap
    seed: int = eqx.field(default=SEED, repr=False)

    # Post-init attributes
    nz: int = eqx.field(init=False)
    beta: jnp.array = eqx.field(init=False)
    _num_parameters: int = eqx.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Parametrize `beta` given random seed."""
        key = _misc.get_key(self.seed, "nonlin_funcs")
        self.beta = jax.random.uniform(
            key, shape=(self.phi.num_features(), self.nw), minval=-1, maxval=1
        )
        self.nz = self.phi.nz
        self._num_parameters = self.beta.size

    def _evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.phi._compute_features(z) @ self.beta

    def num_parameters(self) -> int:
        """Return the size of `beta`."""
        return self._num_parameters


class NeuralNetwork(AbstractNonlinearFunction, strict=True):
    """Fully connected feedforward neural network.

    Attributes
    ----------
    nz : int
        Number of input features (dimension of latent signal `z`).
    nw : int
        Number of output features (dimension of latent signal `w`).
    num_layers : int
        Number of hidden layers.
    num_neurons_per_layer : int
        Number of neurons per hidden layer.
    activation : Callable, from `jax.nn`
        Activation function used in each hidden layer.
    seed : int
        Used for randomly initializing:
        1. Nonlinear coefficient matrix `beta`;
        2. The matrices `B_w`, `C_z`, `D_yw`, and `D_zu` (initialized
           externally, not by this class).
    bias : bool
        Whether to include bias terms in each layer.

    """

    nz: int = eqx.field(repr=False)
    nw: int = eqx.field(repr=False)
    num_layers: int = eqx.field(repr=False)
    num_neurons_per_layer: int = eqx.field(repr=False)
    activation: Callable = eqx.field(repr=False)
    seed: int = eqx.field(default=SEED, repr=False)
    bias: bool = eqx.field(default=True, repr=False)

    # Post-init attributes
    model: eqx.nn.MLP = eqx.field(init=False)
    _num_parameters: int = eqx.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Parametrize neural network given random seed."""
        key = _misc.get_key(self.seed, "nonlin_funcs")
        self.model = eqx.nn.MLP(
            in_size=self.nz,
            out_size=self.nw,
            width_size=self.num_neurons_per_layer,
            depth=self.num_layers,
            activation=self.activation,
            use_bias=self.bias,
            key=key,
        )
        self._num_parameters = sum(
            x.size
            for x in jax.tree_util.tree_leaves(self.model)
            if isinstance(x, jax.Array)
        )

    def _evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.model)(z)

    def num_parameters(self) -> int:
        """Return the number of neural network parameters."""
        return self._num_parameters
