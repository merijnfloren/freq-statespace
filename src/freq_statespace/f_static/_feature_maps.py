"""Nonlinear feature mappings (`z` to `features`) that are linear in the parameters."""

from abc import abstractmethod
from itertools import combinations_with_replacement

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class AbstractFeatureMap(eqx.Module):
    """Abstract base class for feature mappings.

    Subclasses must provide the attributes `nz` and `num_features`,
    and must implement the methods `_compute_features()` and `num_features()`.
    """

    nz: eqx.AbstractVar[int]
    num_features: eqx.AbstractVar[int]

    @abstractmethod
    def _compute_features(self, z: jnp.ndarray) -> jnp.ndarray:
        """Compute the nonlinear feature mapping.

        From inputs of shape (..., `nz`) to outputs of shape (..., `num_features`).
        """
        pass


class Polynomial(AbstractFeatureMap, strict=True):
    """Flexible polynomial feature map.

    This class constructs and evaluates polynomial basis features up to a
    specified degree, with optional cross-terms, offset, linear terms, and
    `tanh` clipping of the inputs.
    """

    nz: int
    degree: int
    type: str
    cross_terms: bool
    offset: bool
    linear: bool
    tanh_clip: bool
    num_features: int
    combination_matrix: jnp.ndarray = eqx.field(repr=False)

    def __init__(
        self,
        nz: int,
        degree: int,
        type: str = "full",
        cross_terms: bool = True,
        offset: bool = True,
        linear: bool = True,
        tanh_clip: bool = True,
    ) -> None:
        """Initialize a flexible polynomial feature map.

        Parameters
        ----------
        nz : int
            Number of input features (dimension of latent signal `z`).
        degree : int
            Maximum polynomial degree.
        type : str, optional
            Type of polynomial. Must be one of `"full"`, `"odd"`, or `"even"`.
            Defaults to `"full"`.
        cross_terms : bool, optional
            Whether to include cross-terms in the polynomial features.
            Defaults to `True`.
        offset : bool, optional
            Whether to include a constant offset term in the features.
            Defaults to `True`.
        linear : bool, optional
            Whether to include linear terms in the polynomial features.
            Defaults to `True`.
        tanh_clip : bool, optional
            Whether to apply `tanh` clipping to the input features before
            constructing polynomial terms. Defaults to `True`.

        """
        self.nz = nz
        self.degree = degree
        self.type = type
        self.cross_terms = cross_terms
        self.offset = offset
        self.linear = linear
        self.tanh_clip = tanh_clip

        # Construct the feature architecture based on the specified parameters.
        if self.type == "full":
            active_degrees = range(1 if self.linear else 2, self.degree + 1)
        elif self.type == "odd":
            active_degrees = range(1 if self.linear else 3, self.degree + 1, 2)
        elif self.type == "even":
            active_degrees = range(2, self.degree + 1, 2)
        else:
            raise ValueError(
                'Invalid polynomial `type`. Must be "full", "odd", or "even".'
            )

        max_degree = active_degrees[-1]
        combination_matrix = []
        combination_row = np.full((max_degree,), -1, dtype=int)

        num_features = 1 if self.offset else 0
        for _, deg in enumerate(active_degrees):
            if self.cross_terms:
                combinations = tuple(
                    combinations_with_replacement(range(self.nz), deg)
                )
            else:
                combinations = tuple((i,) * deg for i in range(self.nz))

            num_features += len(combinations)
            for _, comb in enumerate(combinations):
                combination_row[:deg] = comb
                combination_matrix.append(combination_row.copy())

        self.num_features = num_features
        self.combination_matrix = jnp.array(combination_matrix)

    def _compute_features(self, z: jnp.ndarray) -> jnp.ndarray:
        N, nz = z.shape
        if nz != self.nz:
            raise ValueError(
                "Input size does not match the basis function size: "
                "`z.shape[1] != nz`."
            )

        if self.tanh_clip:
            z = jnp.tanh(z)

        # Augment input to make it consistent with the combination matrix
        z_augmented = jnp.hstack((z, jnp.ones((N, 1))))

        def _compute_phi_z(combination_idx):
            combination = self.combination_matrix[combination_idx]
            z_selected = jnp.take(z_augmented, combination.ravel(), axis=1)
            return jnp.prod(z_selected, axis=1)

        num_combs = self.combination_matrix.shape[0]
        phi_z = jax.vmap(_compute_phi_z, out_axes=1)(jnp.arange(num_combs))

        return jnp.hstack((jnp.ones((N, 1)), phi_z)) if self.offset else phi_z


class LegendrePolynomial(AbstractFeatureMap, strict=True):
    """Legendre polynomial feature map.

    This class constructs and evaluates Legendre polynomial features up to a
    specified degree, with optional offset terms and optional `tanh` clipping
    of the inputs.
    """

    nz: int
    degree: int
    offset: bool
    tanh_clip: bool
    num_features: int

    def __init__(
        self,
        nz: int,
        degree: int,
        offset: bool = True,
        tanh_clip: bool = True,
    ) -> None:
        """Initialize a Legendre polynomial feature map.

        Parameters
        ----------
        nz : int
            Number of input features (dimension of latent signal `z`).
        degree : int
            Maximum polynomial degree.
        offset : bool, optional
            Whether to include a constant offset term. Defaults to `True`.
        tanh_clip : bool, optional
            Whether to apply `tanh` clipping to the inputs. Defaults to `True`.

        """
        self.nz = nz
        self.degree = degree
        self.offset = offset
        self.tanh_clip = tanh_clip
        self.num_features = self.nz * self.degree + (1 if self.offset else 0)

    def _compute_features(self, z: jnp.ndarray) -> jnp.ndarray:

        def _compute_phi_z(k, state):
            phi_z, phi_z_previous, phi_z_two_before = state
            phi_z_current = (
                (2 * k - 1) / k * z * phi_z_previous
                - (k - 1) / k * phi_z_two_before
            )
            phi_z = phi_z.at[k - 2, ...].set(phi_z_current)
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


class ChebyshevPolynomial(AbstractFeatureMap, strict=True):
    """Chebyshev polynomial feature map.

    This class constructs and evaluates Chebyshev polynomial features of the
    first or second kind up to a specified degree, with optional offset terms
    and optional `tanh` clipping of the inputs.
    """

    nz: int
    degree: int
    type: int
    offset: bool
    tanh_clip: bool
    num_features: int

    def __init__(
        self,
        nz: int,
        degree: int,
        type: int,
        offset: bool = True,
        tanh_clip: bool = True,
    ) -> None:
        """Initialize a Chebyshev polynomial feature map.

        Parameters
        ----------
        nz : int
            Number of input features (dimension of latent signal `z`).
        degree : int
            Maximum polynomial degree.
        type : int
            Polynomial type:
            - `1`: First kind (orthogonal w.r.t. 1/sqrt(1 - x²))
            - `2`: Second kind (orthogonal w.r.t. sqrt(1 - x²))
        offset : bool, optional
            Whether to include a constant offset term. Defaults to `True`.
        tanh_clip : bool, optional
            Whether to apply `tanh` clipping to the inputs. Defaults to `True`.

        """
        if type not in (1, 2):
            raise ValueError('Invalid polynomial `type`. Must be `1` or `2`.')

        self.nz = nz
        self.degree = degree
        self.type = type
        self.offset = offset
        self.tanh_clip = tanh_clip
        self.num_features = self.nz * self.degree + (1 if self.offset else 0)

    def _compute_features(self, z: jnp.ndarray) -> jnp.ndarray:

        def _compute_phi_z(k, state):
            phi_z, phi_z_previous, phi_z_two_before = state
            phi_z_current = 2 * z * phi_z_previous - phi_z_two_before
            phi_z = phi_z.at[k - 2, ...].set(phi_z_current)
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
