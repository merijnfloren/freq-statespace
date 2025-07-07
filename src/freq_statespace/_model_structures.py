"""BLA and NL-LFR model classes, optimized for use with JAX and Equinox."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from freq_statespace._data_manager import Normalizer
from freq_statespace.f_static._nonlin_funcs import AbstractNonlinearFunction


class ModelBLA(eqx.Module):
    """BLA model class.

    Attributes
    ----------
    A : jnp.ndarray, shape (nx, nx)
    B_u : jnp.ndarray, shape (nx, nu)
    C_y : jnp.ndarray, shape (ny, nx)
    D_yu : jnp.ndarray, shape (ny, nu)
    ts : float
        Sampling time (in seconds) of the discrete system.
    norm : Normalizer
        Contains means and standard deviations of input-output signals.

    """

    A: jnp.ndarray = eqx.field(converter=jnp.asarray)
    B_u: jnp.ndarray = eqx.field(converter=jnp.asarray)
    C_y: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_yu: jnp.ndarray = eqx.field(converter=jnp.asarray)
    ts: float
    norm: Normalizer = eqx.field(static=True)

    def _simulate(
        self, u: np.ndarray, *, offset: int = 0, x0: np.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate the BLA model in the time domain.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        u : jnp.ndarray, shape (N, nu, R)
            Normalized input signal where:
            - N: number of time steps
            - nu: number of inputs
            - R: number of realizations
        offset : int
            Non-negative integer, must be `≤ N`. If greater than 0, the last
            `offset` input samples are prepended before the simulation starts.
            These prepended samples are used for simulation warm-up and are not
            included in the returned tuple.
        x0 : jnp.ndarray of shape (nx, R), optional
            Initial state of the system. If not provided, the simulation starts
            from a zero state. If `offset > 0`, the simulation first prepends
            the `offset` input samples; `x0` then refers to the initial state
            of the offset samples.

        Returns
        -------
        Y : jnp.ndarray, shape (N, ny, R)
            Simulated output trajectories.
        X : jnp.ndarray, of shape (N, nx, R)
            Simulated state trajectories.
        W : jnp.ndarray, of shape (N, nw, R)
            Static nonlinear function outputs.
        Z : jnp.ndarray, of shape (N, nz, R)
            Static nonlinear function inputs.

        Raises
        ------
        ValueError
            If `offset` is not a non-negative integer `≤ N`.

        """
        def _make_step(k, state):
            X, Y_accum, X_accum = state
            U = jax.lax.dynamic_slice(u, (k, 0, 0), (1, nu, R)).squeeze(axis=0)

            # Model equations
            X_next = self.A @ X + self.B_u @ U
            Y = self.C_y @ X + self.D_yu @ U
            return X_next, Y_accum.at[k, ...].set(Y), X_accum.at[k, ...].set(X)

        # Validate offset
        if not isinstance(offset, int) or offset < 0 or offset > u.shape[0]:
            raise ValueError(
                f"'offset' must be a non-negative integer smaller"
                f" or equal to N (={u.shape[0]}), got {offset}."
            )

        # Extend input signal if needed
        if offset > 0:
            u = jnp.concatenate((u[-offset:, ...], u), axis=0)

        N, nu, R = u.shape
        ny, nx = self.C_y.shape

        if x0 is None:
            x0 = jnp.zeros((nx, R))

        loop_init = (
            x0,
            jnp.zeros((N, ny, R)),  # Y_accum
            jnp.zeros((N, nx, R)),  # X_accum
        )
        Y, X = jax.lax.fori_loop(0, N, _make_step, loop_init)[1:]
        return Y[offset:, ...], X[offset:, ...]

    def simulate(
        self,
        u: np.ndarray,
        *,
        x0: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate the BLA model in the time domain.

        Parameters
        ----------
        u : np.ndarray, shape (N, nu) or (N,)
            Input signal of shape (N, nu). If the system has only one input
            channel, `u` may be a 1D array of shape (N,).
        x0 : np.ndarray, shape (nx,), optional
            Initial state of the system. If not provided, the initial state is
            assumed to be zero.

        Returns
        -------
        y : np.ndarray, shape (N, ny) or (N,)
            Simulated output trajectories. Shape is (N,) in case `ny == 1`.
        t : np.ndarray, shape (N,)
            Time vector corresponding to the simulation.
        x : np.ndarray, shape (N, nx) or (N,)
            Simulated state trajectories. Shape is (N,) in case `nx == 1`.

        """
        # Validate if `u` is 1D or 2D, then extend to 3D
        if u.ndim == 1:
            u = u[:, None, None]
        elif u.ndim == 2:
            u = u[:, :, None]
        else:
            raise ValueError(f"`u` must be 1D or 2D, got {u.ndim}D.")

        # Validate initial state
        if x0 is not None:
            if x0.ndim != 1:
                raise ValueError(f"`x0` must be 1D, got {x0.ndim}D.")
            if x0.shape[0] != self.A.shape[0]:
                raise ValueError(
                    f"`x0` must have shape ({self.A.shape[0]},), got {x0.shape}."
                )
            x0 = x0.reshape(-1, 1)
        else:
            x0 = jnp.zeros((self.A.shape[0], 1))

        # Normalize input signal
        u_mean = self.norm.u_mean.reshape(1, -1)
        u_std = self.norm.u_std.reshape(1, -1)
        u = (u - u_mean) / u_std

        u = jnp.asarray(u)
        y, x = self._simulate(u, offset=0, x0=x0)

        # Denormalize output signal
        y_mean = self.norm.y_mean.reshape(1, -1, 1)
        y_std = self.norm.y_std.reshape(1, -1, 1)
        y = y * y_std + y_mean

        y, x = jnp.squeeze(y), jnp.squeeze(x)

        t = np.arange(y.shape[0]) * self.ts

        return np.asarray(y), t, np.asarray(x)

    def _frequency_response(
        self,
        f: np.ndarray,
    ) -> jnp.ndarray:
        """Compute the frequency response of the system.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        f : np.ndarray, shape (F_f,)
            Frequency points in Hz.

        Returns
        -------
        G : jnp.ndarray
            Frequency response matrix of shape (F_f, ny, nu).

        """
        def G(k):
            G_x = jnp.linalg.solve(zj[k] * I_nx - self.A, B_u)
            return C_y @ G_x + self.D_yu

        fs = 1 / self.ts
        z = 2 * jnp.pi * f / fs
        zj = jnp.exp(z * 1j)

        I_nx = jnp.eye(self.A.shape[0])
        B_u = self.B_u.astype(complex)  # to suppress a warning
        C_y = self.C_y.astype(complex)  # to suppress a warning
        return jax.vmap(G)(np.arange(len(f)))

    def num_parameters(self) -> int:
        """Return the total number of model parameters."""
        return self.A.size + self.B_u.size + self.C_y.size + self.D_yu.size


class ModelNonlinearLFR(ModelBLA):
    """NL-LFR model class.

    Inherits from `ModelBLA` and adds linear matrices `B_w`, `C_z`, `D_yw`, `D_zu`, and
    static nonlinear feedback.

    Attributes
    ----------
    B_w : jnp.ndarray, shape (nx, nw)
    C_z : jnp.ndarray, shape (nz, nx)
    D_yw : jnp.ndarray, shape (ny, nw)
    D_zu : jnp.ndarray, shape (nz, nu)
    f_static : `AbstractNonlinearFunction`
        Static nonlinear function mapping `z` to `w`.

    """

    B_w: jnp.ndarray = eqx.field(converter=jnp.asarray)
    C_z: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_yw: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_zu: jnp.ndarray = eqx.field(converter=jnp.asarray)
    f_static: AbstractNonlinearFunction

    def _simulate(
        self, u: np.ndarray, *, offset: int = 0, x0: np.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate the NL-LFR model in the time domain.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        u : jnp.ndarray, shape (N, nu, R)
            Normalized input signal where:
            - N: number of time steps
            - nu: number of inputs
            - R: number of realizations
        offset : int
            Non-negative integer, must be `≤ N`. If greater than 0, the last
            `offset` input samples are prepended before the simulation starts.
            These prepended samples are used for simulation warm-up and are not
            included in the returned tuple.
        x0 : jnp.ndarray of shape (nx, R), optional
            Initial state of the system. If not provided, the simulation starts
            from a zero state. If `offset > 0`, the simulation first prepends
            the `offset` input samples; `x0` then refers to the initial state
            of the offset samples.

        Returns
        -------
        Y : jnp.ndarray, shape (N, ny, R)
            Simulated output trajectories.
        X : jnp.ndarray, of shape (N, nx, R)
            Simulated state trajectories.
        W : jnp.ndarray, of shape (N, nw, R)
            Static nonlinear function outputs.
        Z : jnp.ndarray, of shape (N, nz, R)
            Static nonlinear function inputs.

        Raises
        ------
        ValueError
            If `offset` is not a non-negative integer `≤ N`.

        """
        def _make_step(k, state):
            X, Y_accum, X_accum, W_accum, Z_accum = state
            U = jax.lax.dynamic_slice(u, (k, 0, 0), (1, nu, R)).squeeze(axis=0)

            # Model equations
            Z = self.C_z @ X + self.D_zu @ U
            W = self.f_static._evaluate(Z.T).T
            X_next = self.A @ X + self.B_u @ U + self.B_w @ W
            Y = self.C_y @ X + self.D_yu @ U + self.D_yw @ W
            return (
                X_next,
                Y_accum.at[k, ...].set(Y),
                X_accum.at[k, ...].set(X),
                W_accum.at[k, ...].set(W),
                Z_accum.at[k, ...].set(Z),
            )

        # Validate offset
        if not isinstance(offset, int) or offset < 0 or offset > u.shape[0]:
            raise ValueError(
                f"'offset' must be a non-negative integer smaller"
                f" or equal to N (={u.shape[0]}), got {offset}."
            )

        # Extend input signal if needed
        if offset > 0:
            u = jnp.concatenate((u[-offset:, ...], u), axis=0)

        N, nu, R = u.shape
        nz, nx = self.C_z.shape
        ny, nw = self.D_yw.shape

        if x0 is None:
            x0 = jnp.zeros((nx, R))

        loop_init = (
            x0,
            jnp.zeros((N, ny, R)),  # Y_accum
            jnp.zeros((N, nx, R)),  # X_accum
            jnp.zeros((N, nw, R)),  # W_accum
            jnp.zeros((N, nz, R)),  # Z_accum
        )
        Y, X, W, Z = jax.lax.fori_loop(0, N, _make_step, loop_init)[1:]
        return (Y[offset:, ...], X[offset:, ...], W[offset:, ...], Z[offset:, ...])

    def simulate(
        self,
        u: np.ndarray,
        *,
        x0: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate the NL-LFR model in the time domain.

        Parameters
        ----------
        u : np.ndarray, shape (N, nu) or (N,)
            Input signal of shape (N, nu). If the system has only one input
            channel, `u` may be a 1D array of shape (N,).
        x0 : np.ndarray, shape (nx,), optional
            Initial state of the system. If not provided, the initial state is
            assumed to be zero.

        Returns
        -------
        y : np.ndarray, shape (N, ny) or (N,)
            Simulated output trajectories. Shape is (N,) in case `ny == 1`.
        t : np.ndarray, shape (N,)
            Time vector corresponding to the simulation.
        x : np.ndarray, shape (N, nx) or (N,)
            Simulated state trajectories. Shape is (N,) in case `nx == 1`.
        w : np.ndarray, shape (N, nw), or (N,)
            Static nonlinear function outputs. Shape is (N,) in case `nw == 1`.
        z : np.ndarray, shape (N, nz) or (N,)
            Static nonlinear function inputs. Shape is (N,) in case `nz == 1`.

        """
        # Validate if `u` is 1D or 2D, then extend to 3D
        if u.ndim == 1:
            u = u[:, None, None]
        elif u.ndim == 2:
            u = u[:, :, None]
        else:
            raise ValueError(f"`u` must be 1D or 2D, got {u.ndim}D.")

        # Validate initial state
        if x0 is not None:
            if x0.ndim != 1:
                raise ValueError(f"`x0` must be 1D, got {x0.ndim}D.")
            if x0.shape[0] != self.A.shape[0]:
                raise ValueError(
                    f"`x0` must have shape ({self.A.shape[0]},), got {x0.shape}."
                )
            x0 = x0.reshape(-1, 1)
        else:
            x0 = jnp.zeros((self.A.shape[0], 1))

        # Normalize input signal
        u_mean = self.norm.u_mean.reshape(1, -1)
        u_std = self.norm.u_std.reshape(1, -1)
        u = (u - u_mean) / u_std

        y, x, w, z = self._simulate(u, offset=0, x0=x0)

        # Denormalize output signal
        y_mean = self.norm.y_mean.reshape(1, -1, 1)
        y_std = self.norm.y_std.reshape(1, -1, 1)
        y = y * y_std + y_mean

        y, x = jnp.squeeze(y), jnp.squeeze(x)
        w, z = jnp.squeeze(w), jnp.squeeze(z)

        t = np.arange(y.shape[0]) * self.ts

        return np.asarray(y), t, np.asarray(x), np.asarray(w), np.asarray(z)

    def num_parameters(self) -> int:
        """Return the total number of model parameters."""
        return (
            self.B_w.size
            + self.C_z.size
            + self.D_yw.size
            + self.D_zu.size
            + super().num_parameters()
            + self.f_static.num_parameters()
        )
