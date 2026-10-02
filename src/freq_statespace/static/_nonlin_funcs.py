"""General static nonlinear function mappings (mapping `z` to `w`)."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from freq_statespace import _misc
from freq_statespace._activations import activation_from_config, activation_to_config
from freq_statespace._config import SEED
from freq_statespace._serialize import NONLINEAR_FUNCTION_REGISTRY, Serializable
from freq_statespace.static._feature_maps import AbstractFeatureMap

if TYPE_CHECKING:
    from jaxtyping import Array, Float


class AbstractNonlinearFunction(eqx.Module, Serializable):
    """Abstract base class for nonlinear function mappings.

    Subclasses must provide the attributes `nw`, `nz`, `seed`, and `num_parameters`,
    and implement `_evaluate()`.

    The tagged-config serialization hooks are inherited from `Serializable`.
    """

    nw: eqx.AbstractVar[int]
    nz: eqx.AbstractVar[int]
    seed: eqx.AbstractVar[int]
    num_parameters: eqx.AbstractVar[int]

    @abstractmethod
    def _evaluate(
        self, z: Float[Array, "... nz"]
    ) -> Float[Array, "... nw"]:
        """Evaluate the nonlinear function.

        From inputs of shape (..., `nz`) to outputs of shape (..., `nw`).
        """
        ...

    
@NONLINEAR_FUNCTION_REGISTRY.register
class BasisFunctionModel(AbstractNonlinearFunction):
    """Static nonlinear function based on an `AbstractFeatureMap`.

    This class combines an `AbstractFeatureMap` with a coefficient matrix `beta`
    to implement and evaluate a static nonlinear mapping that is linear in its
    parameters.
    """

    nw: int
    nz: int
    beta: Float[Array, "n_features nw"]
    phi: AbstractFeatureMap
    num_parameters: int
    seed: int = eqx.field(repr=False)
    _type_name: ClassVar[str] = "basis_function_model"

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
            Used for randomly initializing (i) the nonlinear coefficient matrix
            `beta` and (ii) the matrices `B_w`, `C_z`, `D_yw`, and `D_zu` 
            (initialized externally, not by this class). Defaults to `42`.

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
        
    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> Self:
        from freq_statespace._serialize import FEATURE_MAP_REGISTRY

        phi = FEATURE_MAP_REGISTRY.from_config(config["phi"])
        if not isinstance(phi, AbstractFeatureMap):
            msg = "Deserialized `phi` is not a feature map."
            raise TypeError(msg)
        return cls(
            nw=config["nw"],
            phi=phi,
            seed=config["seed"],
        )

    def _evaluate(
        self, z: Float[Array, "n_samples nz"]
    ) -> Float[Array, "n_samples nw"]:
        return self.phi._compute_features(z) @ self.beta
    
    def _config_payload(self) -> dict[str, Any]:
        config = {
            "nw": self.nw,
            "nz": self.nz,
            "seed": self.seed,  # not strictly needed, but included for completeness
            "phi": self.phi.to_config(),
        }
        return config


@NONLINEAR_FUNCTION_REGISTRY.register
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
    _type_name: ClassVar[str] = "neural_network"

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
        activation : Callable, from `jax.nn` or a `functools.partial` wrapping one
            Activation function used in hidden layers, must be an elementwise function
            from `jax.nn` or a `functools.partial` wrapping one (e.g. to specify keyword
            arguments). For a complete list of supported activations, see 
            'ELEMENTWISE_ACTIVATIONS' in `freq_statespace._activations`.
        seed : int, optional
            Used for randomly initializing (i) the neural network parameters and
            (ii) the matrices `B_w`, `C_z`, `D_yw`, and `D_zu` (initialized
            externally, not by this class). Defaults to `42`.
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
        
    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> Self:
        return cls(
            nz=config["nz"],
            nw=config["nw"],
            layers=config["layers"],
            neurons_per_layer=config["neurons_per_layer"],
            activation=activation_from_config(config["activation"]),
            seed=config["seed"],
            bias=config["bias"],
        )

    def _evaluate(
        self, z: Float[Array, "n_samples nz"]
    ) -> Float[Array, "n_samples nw"]:
        return jax.vmap(self.model)(z)
    
    def _config_payload(self) -> dict[str, Any]:
        return {
            "nw": self.nw,
            "nz": self.nz,
            "layers": self.layers,
            "neurons_per_layer": self.neurons_per_layer,
            "activation": activation_to_config(self.activation),
            "bias": self.bias,
            "seed": self.seed,  # not strictly needed, but included for completeness
        }
