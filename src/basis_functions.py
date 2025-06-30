"""
Basis function classes for nonlinear feature expansion.

This module defines a common interface for basis functions used in
nonlinear modeling and provides concrete implementations for:

- Generic multivariate polynomials;
- Legendre polynomials (orthogonal);
- Chebyshev polynomials (orthogonal).
"""

from abc import abstractmethod
from itertools import combinations_with_replacement

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class AbstractBasisFunction(eqx.Module, strict=True):
    nz: eqx.AbstractVar[int]

    @abstractmethod
    def compute_features(self, z: jnp.ndarray) -> jnp.ndarray:
        """From size (-1, nz) to size (-1, num_features())."""
        pass

    @abstractmethod
    def num_features(self) -> int:
        """Returns the total number of model features."""
        pass


class Polynomial(AbstractBasisFunction, strict=True):
    """
    Multivariate polynomial basis function with configurable structure.

    Parameters
    ----------
    nz : int
        Number of input dimensions.
    degree : int
        Maximum degree of the polynomial terms.
    type : str, default='full'
        Type of polynomial degrees to include:
        - 'full': all degrees up to `degree`
        - 'odd' : only odd degrees (1, 3, 5, ...)
        - 'even': only even degrees (2, 4, 6, ...)
    cross_terms : bool, default=True
        Whether to include cross-variable terms (e.g., z1 * z2).
    offset : bool, default=True
        Whether to include a constant offset (bias term).
    linear : bool, default=True
        Whether to include linear terms (degree-1 monomials).
    tanh_clip : bool, default=True
        Whether to apply elementwise tanh to inputs before feature computation.
    """
    nz: int
    degree: int
    type: str = 'full'
    cross_terms: bool = True
    offset: bool = True
    linear: bool = True
    tanh_clip: bool = True

    # Internal attributes (computed within the class)
    _num_features: int = eqx.field(init=False, repr=False)
    _combination_matrix: jnp.ndarray = eqx.field(init=False, repr=False)

    def __post_init__(self):

        if self.type == 'full':
            active_degrees = range(1 if self.linear else 2, self.degree + 1)
        elif self.type == 'odd':
            active_degrees = range(1 if self.linear else 3, self.degree + 1, 2)
        elif self.type == 'even':
            active_degrees = range(2, self.degree + 1, 2)
        else:
            raise ValueError(
                'Invalid polynomial type. Must be "full", "odd", or "even".'
            )

        max_degree = active_degrees[-1]
        combination_matrix = []
        combination_row = np.full((max_degree,), -1, dtype=int)
        num_features = 1 if self.offset else 0
        for _, deg in enumerate(active_degrees):
            if self.cross_terms:
                combinations = tuple(
                    combinations_with_replacement(
                        range(self.nz), deg
                    )
                )
            else:
                combinations = tuple(
                    (i,) * deg for i in range(self.nz)
                )

            num_features += len(combinations)
            for _, comb in enumerate(combinations):
                combination_row[:deg] = comb
                combination_matrix.append(combination_row.copy())

        self._num_features = num_features
        self._combination_matrix = jnp.array(combination_matrix)

    def compute_features(self, z: jnp.ndarray) -> jnp.ndarray:

        N, nz = z.shape

        if nz != self.nz:
            raise ValueError(
                'Input size does not match the basis function size.'
            )

        if self.tanh_clip:
            z = jnp.tanh(z)

        # Augment input to make it consistent with the combination matrix
        z_augmented = jnp.hstack((z, jnp.ones((N, 1))))

        def _compute_phi_z(combination_idx):
            combination = self._combination_matrix[combination_idx]
            z_selected = jnp.take(z_augmented, combination.ravel(), axis=1)
            return jnp.prod(z_selected, axis=1)

        num_combs = self._combination_matrix.shape[0]
        phi_z = jax.vmap(_compute_phi_z, out_axes=1)(jnp.arange(num_combs))

        return jnp.hstack((jnp.ones((N, 1)), phi_z)) if self.offset else phi_z

    def num_features(self) -> int:
        return self._num_features


class LegendrePolynomial(AbstractBasisFunction, strict=True):
    """
    Multivariate Legendre polynomial basis function.

    Constructs a feature expansion using Legendre polynomials, which are
    orthogonal over the interval [-1, 1] with unit weighting in the univariate
    case.

    Parameters
    ----------
    nz : int
        Number of input dimensions.
    degree : int
        Maximum polynomial degree per input dimension.
    offset : bool, default=True
        Whether to include a constant offset (bias term).
    tanh_clip : bool, default=True
        Whether to apply elementwise tanh to inputs before feature computation.
    """
    nz: int
    degree: int
    offset: bool = True
    tanh_clip: bool = True
    _num_features: int = eqx.field(init=False, repr=False)

    def __post_init__(self):
        self._num_features = (
            self.nz * self.degree + (1 if self.offset else 0)
        )

    def compute_features(self, z: jnp.ndarray) -> jnp.ndarray:

        def _compute_phi_z(k, state):
            phi_z, phi_z_previous, phi_z_two_before = state
            phi_z_current = (
                (2 * k - 1) / k * z * phi_z_previous
                - (k - 1) / k * phi_z_two_before
            )
            phi_z = phi_z.at[k-2, ...].set(phi_z_current)
            return phi_z, phi_z_current, phi_z_previous

        if self.tanh_clip:
            z = jnp.tanh(z)

        N = z.shape[0]
        phi_z0 = jnp.zeros((self.degree - 1, N, self.nz))
        loop_init = (phi_z0, z, jnp.ones_like(z))

        phi_z = jax.lax.fori_loop(
            2, self.degree + 1, _compute_phi_z, loop_init, unroll=True
        )[0]
        phi_z = jnp.transpose(phi_z, (1, 2, 0)).reshape(N, -1)
        phi_z = jnp.hstack((jnp.ones((N, 1)), z, phi_z))
        return phi_z if self.offset else phi_z[:, 1:]

    def num_features(self) -> int:
        return self._num_features


class ChebyshevPolynomial(AbstractBasisFunction, strict=True):
    """
    Multivariate Chebyshev polynomial basis function.

    Constructs a feature expansion using Chebyshev polynomials, which are
    orthogonal over the interval [-1, 1] with respect to a weight function
    that depends on the polynomial type.

    Parameters
    ----------
    nz : int
        Number of input dimensions.
    degree : int
        Maximum polynomial degree per input dimension.
    type : int
        Type of Chebyshev polynomial:
        - 1: First kind (orthogonal w.r.t. 1 / sqrt(1 - x²))
        - 2: Second kind (orthogonal w.r.t. sqrt(1 - x²))
    offset : bool, default=True
        Whether to include a constant offset (bias term).
    tanh_clip : bool, default=True
        Whether to apply elementwise tanh to inputs before feature computation.
    """
    nz: int
    degree: int
    type: int
    offset: bool = True
    tanh_clip: bool = True
    _num_features: int = eqx.field(init=False, repr=False)

    def __post_init__(self):
        if self.type not in [1, 2]:
            raise ValueError('Invalid Chebyshev polynomial type.')
        self._num_features = (
            self.nz * self.degree + (1 if self.offset else 0)
        )

    def compute_features(self, z: jnp.ndarray) -> jnp.ndarray:

        def _compute_phi_z(k, state):
            phi_z, phi_z_previous, phi_z_two_before = state
            phi_z_current = 2 * z * phi_z_previous - phi_z_two_before
            phi_z = phi_z.at[k-2, ...].set(phi_z_current)
            return phi_z, phi_z_current, phi_z_previous

        if self.tanh_clip:
            z = jnp.tanh(z)

        N = z.shape[0]
        phi_z0 = jnp.zeros((self.degree - 1, N, self.nz))
        loop_init = (phi_z0, self.type * z, jnp.ones_like(z))

        phi_z = jax.lax.fori_loop(
            2, self.degree + 1, _compute_phi_z, loop_init, unroll=True
        )[0]
        phi_z = jnp.transpose(phi_z, (1, 2, 0)).reshape(N, -1)
        phi_z = jnp.hstack((jnp.ones((N, 1)), self.type * z, phi_z))
        return phi_z if self.offset else phi_z[:, 1:]

    def num_features(self) -> int:
        return self._num_features
