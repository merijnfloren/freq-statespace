"""Helpers for serializing supported JAX activation functions."""
from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any

import numpy as np


ELEMENTWISE_JAX_ACTIVATIONS: tuple[str, ...] = (
    "celu",
    "elu",
    "gelu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "leaky_relu",
    "mish",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softplus",
    "sparse_plus",
    "squareplus",
    "swish",
    "tanh",
)


def _supported_activation_registry() -> dict[str, Callable]:
    import jax

    return {
        name: getattr(jax.nn, name)
        for name in ELEMENTWISE_JAX_ACTIVATIONS
        if hasattr(jax.nn, name)
    }


def _jsonify_activation_value(value: Any) -> Any:
    """Convert activation kwargs into JSON-safe scalar/list values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonify_activation_value(v) for v in value]
    if isinstance(value, list):
        return [_jsonify_activation_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify_activation_value(v) for k, v in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(
        "Activation kwargs must be JSON-serializable scalars, lists, or dictionaries."
    )


def activation_to_config(activation: Callable) -> dict[str, Any]:
    """Serialize a supported JAX activation function.

    Supported inputs are:
    - a bare elementwise function from `jax.nn`
    - `functools.partial` wrapping one of those functions with keyword arguments
    """
    registry = _supported_activation_registry()
    kwargs: dict[str, Any] = {}

    if isinstance(activation, functools.partial):
        if activation.args:
            raise TypeError(
                "Activation serialization only supports keyword arguments; "
                "positional partial arguments are not supported."
            )
        func = activation.func
        kwargs = dict(activation.keywords or {})
    else:
        func = activation

    name = getattr(func, "__name__", None)
    if name is None or registry.get(name) is not func:
        raise TypeError(
            "Activation must be an elementwise `jax.nn` function or a "
            "`functools.partial` wrapping one."
        )

    json.dumps({k: _jsonify_activation_value(v) for k, v in kwargs.items()})
    return {
        "name": name,
        "kwargs": {k: _jsonify_activation_value(v) for k, v in kwargs.items()},
    }


def activation_from_config(data: str | dict[str, Any]) -> Callable:
    """Deserialize a supported JAX activation function.

    A plain string is accepted for backward compatibility with older saved configs.
    """
    registry = _supported_activation_registry()

    if isinstance(data, str):
        name = data
        kwargs: dict[str, Any] = {}
    elif isinstance(data, dict):
        name = data.get("name")
        kwargs = data.get("kwargs", {})
        if not isinstance(name, str):
            raise ValueError("Activation config must contain a string 'name'.")
        if not isinstance(kwargs, dict):
            raise ValueError("Activation config field 'kwargs' must be a dictionary.")
    else:
        raise TypeError(
            f"Activation config must be a string or dictionary, got {type(data)}."
        )

    func = registry.get(name)
    if func is None:
        supported = ", ".join(sorted(registry))
        raise ValueError(
            f"Unsupported activation '{name}'. Supported activations are: {supported}."
        )

    return functools.partial(func, **kwargs) if kwargs else func
