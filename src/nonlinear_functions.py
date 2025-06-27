from abc import abstractmethod
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from src.basis_functions import AbstractBasisFunction


class AbstractNonlinearFunction(eqx.Module, strict=True):
    nz: eqx.AbstractVar[int]
    nw: eqx.AbstractVar[int]

    @abstractmethod
    def evaluate(self, z: jnp.ndarray) -> jnp.ndarray:
        """From size (-1, nz) to size (-1, nw)"""
        pass

    @abstractmethod
    def num_parameters(self) -> int:
        pass


class BasisFunctionModel(AbstractNonlinearFunction, strict=True):
    nz: int = eqx.field(init=False)
    nw: int
    beta: jnp.array = eqx.field(init=False)
    phi: AbstractBasisFunction
    key: PRNGKeyArray = eqx.field(repr=False)

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
    nz: int
    nw: int
    num_layers: int = eqx.field(repr=False)
    num_neurons_per_layer: int = eqx.field(repr=False)
    activation: Callable = eqx.field(repr=False)
    key: PRNGKeyArray = eqx.field(repr=False)
    bias: bool = eqx.field(default=True, repr=False)
    model: eqx.nn.MLP = eqx.field(init=False)

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
    """ Instantiates BasisFunctionModel for a custom beta. """
    dummy_key = jax.random.key(0)
    return eqx.tree_at(
        where=lambda tree: tree.beta,
        pytree=BasisFunctionModel(nw, phi, dummy_key),
        replace=beta
    )
