"""
Miscellaneous utility functions for general use.
"""

from dataclasses import dataclass, fields

import numpy as np


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


def compute_sample_mean_and_noise_var(
    X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    R = X.shape[2]
    P = X.shape[3]

    X_mean = np.mean(np.abs(X), axis=(2, 3))
    X_p = np.mean(X, axis=3, keepdims=True)

    if P > 1:
        X_var_p = np.sum(
            np.abs(X - np.tile(X_p, (1, 1, 1, P)))**2, axis=-1
        ) / P / (P - 1)
        X_var = np.mean(X_var_p, axis=2) / R
        X_std = np.sqrt(X_var)
    else:
        X_std = None

    return X_mean, X_std
