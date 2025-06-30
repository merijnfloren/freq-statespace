"""
Miscellaneous utility functions for general use.
"""

from dataclasses import fields
from typing import Any


def print_attributes(dataclass_instance: Any) -> None:
    """
    Print the names of all fields in a dataclass instance.

    For each field, its name is printed. If the value is `None`,
    this is indicated explicitly by appending "(None)".

    Parameters
    ----------
    dataclass_instance : Any
        An instance of a dataclass. The function assumes that the object
        is a valid dataclass (can also be an Equinox module!).
    """
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)
        if value is None:
            print(f"{field.name} (None)")
        else:
            print(field.name)
