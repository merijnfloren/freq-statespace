from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.nonlinear_functions import AbstractNonlinearFunction


class ModelBLA(eqx.Module):
    A: jnp.ndarray = eqx.field(converter=jnp.asarray)
    B_u: jnp.ndarray = eqx.field(converter=jnp.asarray)
    C_y: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_yu: jnp.ndarray = eqx.field(converter=jnp.asarray)
    ts: float

    def simulate(
        self,
        u: np.ndarray,
        *,
        handicap: int = 0
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

        def _make_step(k, state):
            """
            Simulate one step forward, parallelised over realisations.
            """

            X, Y_accum, X_accum = state
            U = jax.lax.dynamic_slice(u, (k, 0, 0), (1, nu, R)).squeeze(axis=0)

            # Model equations
            X_next = self.A @ X + self.B_u @ U
            Y = self.C_y @ X + self.D_yu @ U
            return X_next, Y_accum.at[k, ...].set(Y), X_accum.at[k, ...].set(X)

        # Convert input to JAX array
        u = jnp.asarray(u)

        # Validate handicap
        if not isinstance(handicap, int) or handicap < 0:
            raise ValueError(f"'handicap' must be a non-negative integer, got {handicap}.")  # noqa: E501

        # Extend input signal if needed
        if handicap > 0:
            u = jnp.concatenate((u[-handicap:, ...], u), axis=0)

        N, nu, R = u.shape
        ny, nx = self.C_y.shape

        loop_init = (
            jnp.zeros((nx, R)),  # initial state
            jnp.zeros((N, ny, R)),  # Y_accum
            jnp.zeros((N, nx, R))  # X_accum
        )
        Y, X = jax.lax.fori_loop(0, N, _make_step, loop_init)[1:]
        return Y[handicap:, ...], X[handicap:, ...]

    def frequency_response(
        self,
        f: Union[np.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:

        def G(k):
            G_x = jnp.linalg.solve(zj[k] * I_nx - self.A, B_u)
            return C_y @ G_x + self.D_yu

        fs = 1 / self.ts
        z = 2 * jnp.pi * f / fs
        zj = jnp.exp(z * 1j)

        I_nx = jnp.eye(self.A.shape[0])
        B_u = self.B_u.astype(complex)  # to suppress a warning
        C_y = self.C_y.astype(complex)  # to suppress a warning
        return jax.vmap(G)(jnp.arange(len(f)))

    def num_parameters(self) -> int:
        return self.A.size + self.B_u.size + self.C_y.size + self.D_yu.size


class ModelNonlinearLFR(ModelBLA):
    B_w: jnp.ndarray = eqx.field(converter=jnp.asarray)
    C_z: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_yw: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_zu: jnp.ndarray = eqx.field(converter=jnp.asarray)
    f_static: AbstractNonlinearFunction

    def simulate(
        self,
        u: np.ndarray,
        *,
        handicap: int = 0
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        def _make_step(k, state):
            """
            Simulate one step forward, parallelised over realisations.
            """

            X, Y_accum, X_accum, W_accum, Z_accum = state
            U = jax.lax.dynamic_slice(
                u, (k, 0, 0), (1, nu, R)
            ).squeeze(axis=0)

            # Model equations
            Z = self.C_z @ X + self.D_zu @ U
            W = self.f_static.evaluate(Z.T).T
            X_next = self.A @ X + self.B_u @ U + self.B_w @ W
            Y = self.C_y @ X + self.D_yu @ U + self.D_yw @ W
            return (X_next,
                    Y_accum.at[k, ...].set(Y),
                    X_accum.at[k, ...].set(X),
                    W_accum.at[k, ...].set(W),
                    Z_accum.at[k, ...].set(Z))

        u = jnp.asarray(u)

        # Validate handicap
        if not isinstance(handicap, int) or handicap < 0:
            raise ValueError(f"'handicap' must be a non-negative integer, got {handicap}.")  # noqa: E501

        # Extend input signal if needed
        if handicap > 0:
            u = jnp.concatenate((u[-handicap:, ...], u), axis=0)

        N, nu, R = u.shape
        nz, nx = self.C_z.shape
        ny, nw = self.D_yw.shape

        loop_init = (
            jnp.zeros((nx, R)),  # initial state
            jnp.zeros((N, ny, R)),  # Y_accum
            jnp.zeros((N, nx, R)),  # X_accum
            jnp.zeros((N, nw, R)),  # W_accum
            jnp.zeros((N, nz, R)),  # Z_accum
        )
        Y, X, W, Z = jax.lax.fori_loop(0, N, _make_step, loop_init)[1:]
        return (Y[handicap:, ...], X[handicap:, ...],
                W[handicap:, ...], Z[handicap:, ...])

    def num_parameters(self) -> int:
        return (
            self.B_w.size + self.C_z.size + self.D_yw.size + self.D_zu.size
            + super().num_parameters() + self.f_static.num_parameters()
        )
