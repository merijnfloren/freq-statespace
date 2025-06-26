"""
Miscellaneous utility functions for general use.
"""

from dataclasses import dataclass, fields


def print_attributes(dataclass_instance: dataclass) -> None:
    """
    Print the names of all fields in a dataclass instance.

    Fields with a value of None are marked explicitly.
    """
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)
        if value is None:
            print(f"{field.name} (None)")
        else:
            print(field.name)
