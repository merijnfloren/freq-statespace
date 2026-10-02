"""Support for saving and loading ModelBLA and ModelNonlinearLFR instances."""
from __future__ import annotations

import io
import json
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import equinox as eqx


if TYPE_CHECKING:
    from freq_statespace._model_structures import ModelBLA, ModelNonlinearLFR


T = TypeVar("T", bound="Serializable")


class Serializable(ABC):
    """Shared tagged-config serialization contract."""

    _type_name: ClassVar[str]

    @classmethod
    def type_name(cls) -> str:
        return cls._type_name

    @classmethod
    @abstractmethod
    def _from_config(cls: type[T], config: dict[str, Any]) -> T:
        """Build a skeleton object with the right PyTree structure."""
        ...

    @abstractmethod
    def _config_payload(self) -> dict[str, Any]:
        """Return the subclass-specific config payload."""
        ...

    def to_config(self) -> dict[str, Any]:
        return {
            "type_name": self.type_name(),
            "config": self._config_payload(),
        }


class SerializationRegistry:
    """Registry for tagged config deserialization."""

    def __init__(self, family_name: str) -> None:
        self.family_name = family_name
        self._types: dict[str, type[Serializable]] = {}

    def register(self, cls: type[T]) -> type[T]:
        type_name = cls.type_name()
        if type_name in self._types:
            msg = f"Duplicate {self.family_name} serialization type_name: {type_name}."
            raise ValueError(msg)
        self._types[type_name] = cls
        return cls

    def from_config(self, data: dict[str, Any]) -> Serializable:
        if not isinstance(data, dict):
            msg = f"{self.family_name} config must be a dictionary, got {type(data)}."
            raise TypeError(msg)

        type_name = data.get("type_name")
        config = data.get("config")
        if not isinstance(type_name, str):
            msg = f"{self.family_name} config must contain a string 'type_name'."
            raise ValueError(msg)
        if not isinstance(config, dict):
            msg = f"{self.family_name} config must contain a dictionary 'config'."
            raise ValueError(msg)

        cls = self._types.get(type_name)
        if cls is None:
            msg = f"Unknown {self.family_name} serialization type_name: {type_name}."
            raise ValueError(msg)

        return cls._from_config(config)


MODEL_REGISTRY = SerializationRegistry("model")
NONLINEAR_FUNCTION_REGISTRY = SerializationRegistry("nonlinear function")
FEATURE_MAP_REGISTRY = SerializationRegistry("feature map")


def save_model(model: ModelBLA | ModelNonlinearLFR, path: str | Path) -> None:
    """Save a ModelBLA or ModelNonlinearLFR instance given a file path.
    
    The model is serialized to a zip file containing a JSON config and an Equinox 
    weights file.
    
    Parameters
    ----------
    model : ModelBLA or ModelNonlinearLFR
        The model to be saved.
    path : str or Path
        File path where the model should be saved. If the suffix is missing, '.zip'
        will be added automatically. Parent directories will be created if they do not 
        exist yet. If a file already exists at the path, it will be overwritten.
        
    Returns
    -------
    None
    
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".zip")

    path.parent.mkdir(parents=True, exist_ok=True)
    config = model.to_config()

    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, model)

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps(config, indent=2, sort_keys=True))
        zf.writestr("weights.eqx", buf.getvalue())


def load_model(path: str | Path) -> ModelBLA | ModelNonlinearLFR:
    """Load a ModelBLA or ModelNonlinearLFR instance from a file path.
    
    Parameters
    ----------
    path : str or Path
        File path from which the model should be loaded. Must point to a zip file
        containing a JSON config and an Equinox weights file, as produced by 
        `save_model()`.
        
    Returns
    -------
    ModelBLA or ModelNonlinearLFR
        The deserialized model instance.

    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        config = json.loads(zf.read("config.json"))
        weights = io.BytesIO(zf.read("weights.eqx"))

    skeleton = MODEL_REGISTRY.from_config(config)
    model = eqx.tree_deserialise_leaves(weights, skeleton)
    return model
