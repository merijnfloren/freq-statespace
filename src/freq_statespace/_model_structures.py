"""BLA and NL-LFR model classes, optimized for use with JAX and Equinox."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import _misc
from ._data_manager import Normalizer
from .static._nonlin_funcs import AbstractNonlinearFunction


class ModelBLA(eqx.Module):
    """BLA model class.

    Parameters
    ----------
    A : jnp.ndarray, shape (nx, nx)
        State transition matrix.
    B_u : jnp.ndarray, shape (nx, nu)
        Input-to-state matrix.
    C_y : jnp.ndarray, shape (ny, nx)
        State-to-output matrix.
    D_yu : jnp.ndarray, shape (ny, nu)
        Input-to-output matrix.
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
        self,
        u: jnp.ndarray,
        x0: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate the BLA model in the time domain.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        u : jnp.ndarray, shape (N, nu, R)
            Normalized input signal.
        x0 : jnp.ndarray of shape (nx, R)
            Initial state of the system. 

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

        """
        def _make_step(k, state):
            X, Y_accum, X_accum = state
            U = jax.lax.dynamic_slice(u, (k, 0, 0), (1, nu, R)).squeeze(axis=0)

            # Model equations
            X_next = self.A @ X + self.B_u @ U
            Y = self.C_y @ X + self.D_yu @ U
            return X_next, Y_accum.at[k, ...].set(Y), X_accum.at[k, ...].set(X)

        N, nu, R = u.shape
        ny, nx = self.C_y.shape

        loop_init = (
            x0,
            jnp.zeros((N, ny, R)),  # Y_accum
            jnp.zeros((N, nx, R)),  # X_accum
        )
        Y, X = jax.lax.fori_loop(0, N, _make_step, loop_init)[1:]
        return Y, X

    def simulate(
        self,
        u: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        offset: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the BLA model in the time domain for arbitrary input signals.

        Parameters
        ----------
        u : np.ndarray, shape (N,), (N, nu), (N, nu, R), or (N, nu, R, P)
            Input signal. This array can be 1D up to 4D, with:
            - N : number of time steps  
            - nu : number of inputs  
            - R : number of realizations  
            - P : number of periods  

            The input does not need to be normalized; this is handled internally.
            Since this is a public method (not used within an optimization loop), 
            the input also does not need to be periodic.

            If the fourth dimension P is provided, it is assumed that the input is
            periodic. In that case, all periods are internally concatenated into
            the first dimension, effectively simulating multiple periods sequentially.

        x0 : np.ndarray, shape (nx,) or (nx, R), optional
            Initial state for simulation. If not provided, the initial state is
            assumed to be zero.

        offset : int, optional
            Should only be provided if the input signal `u` is periodic. A non-negative
            integer representing the number of initial samples to prepend to the input
            signal to allow the system to reach steady-state before the main simulations
            begin. Those samples are not returned. If not provided, no samples are 
            prepended.

        Returns
        -------
        y : np.ndarray, shape (N, ny), (N, ny, R), or (N, ny, R, P)
            Simulated output time series, at least 2D, with ny as the number of
            output channels.

        t : np.ndarray, shape (N,)
            Time vector corresponding to one simulation of length N.

        x : np.ndarray, shape (N, nx), (N, nx, R), or (N, nx, R, P)
            Simulated state trajectories, at least 2D, with nx as the number of
            state variables.

        Raises
        ------
        ValueError
            If `u` has an invalid shape.
        ValueError
            If `x0` has an invalid shape.
        ValueError
            If `offset` is not a non-negative integer.

        """
        return _simulate_core(self, u, x0=x0, offset=offset, with_wz=False)

    def _frequency_response(self, f: np.ndarray) -> jnp.ndarray:
        """Compute the frequency response of the system.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        f : np.ndarray, shape (freqs,)
            Frequency points in Hz.

        Returns
        -------
        G : jnp.ndarray
            Frequency response matrix of shape (freqs, ny, nu).

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

    Inherits from `ModelBLA` and adds linear matrices `B_w`, `C_z`, `D_yw`, `D_zu`,
    and static nonlinear feedback.
    """

    B_w: jnp.ndarray = eqx.field(converter=jnp.asarray)
    C_z: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_yw: jnp.ndarray = eqx.field(converter=jnp.asarray)
    D_zu: jnp.ndarray = eqx.field(converter=jnp.asarray)
    func_static: AbstractNonlinearFunction
    
    # Reference to initial BLA, used for selecting initial states 
    _bla : ModelBLA = eqx.field(repr=False)
    
    def __init__(
        self,
        A: jnp.ndarray,
        B_u: jnp.ndarray,
        C_y: jnp.ndarray,
        D_yu: jnp.ndarray,
        B_w: jnp.ndarray,
        C_z: jnp.ndarray,
        D_yw: jnp.ndarray,
        D_zu: jnp.ndarray,
        func_static: AbstractNonlinearFunction,
        ts: float,
        norm: Normalizer
    ) -> None:
        """Initialize NL-LFR model.

        Parameters
        ----------
        A : jnp.ndarray, shape (nx, nx)
            State transition matrix.
        B_u : jnp.ndarray, shape (nx, nu)
            Input-to-state matrix.
        C_y : jnp.ndarray, shape (ny, nx)
            State-to-output matrix.
        D_yu : jnp.ndarray, shape (ny, nu)
            Input-to-output matrix.
        B_w : jnp.ndarray, shape (nx, nw)
            Feedback input-to-state matrix.
        C_z : jnp.ndarray, shape (nz, nx)
            State-to-static nonlinear function matrix.
        D_yw : jnp.ndarray, shape (ny, nw)
            Feedback input-to-output matrix.
        D_zu : jnp.ndarray, shape (nz, nu)
            Input-to-static nonlinear function matrix.
        func_static : AbstractNonlinearFunction
            Static nonlinear function mapping `z` to `w`.
        ts : float
            Sampling time (in seconds) of the discrete system.
        norm : Normalizer, optional
            Contains means and standard deviations of input-output signals.

        """
        super().__init__(A, B_u, C_y, D_yu, ts, norm)
        self.B_w = B_w
        self.C_z = C_z
        self.D_yw = D_yw
        self.D_zu = D_zu
        self.func_static = func_static 
        self._bla = ModelBLA(A, B_u, C_y, D_yu, ts, norm)

    def _simulate(
        self,
        u: jnp.ndarray,
        x0: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate the NL-LFR model in the time domain.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        u : jnp.ndarray, shape (N, nu, R)
            Normalized input signal.
        x0 : jnp.ndarray of shape (nx, R)
            Initial state of the system.

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

        """
        def _make_step(k, state):
            X, Y_accum, X_accum, W_accum, Z_accum = state
            U = jax.lax.dynamic_slice(u, (k, 0, 0), (1, nu, R)).squeeze(axis=0)

            # Model equations
            Z = self.C_z @ X + self.D_zu @ U
            W = self.func_static._evaluate(Z.T).T
            X_next = self.A @ X + self.B_u @ U + self.B_w @ W
            Y = self.C_y @ X + self.D_yu @ U + self.D_yw @ W
            return (
                X_next,
                Y_accum.at[k, ...].set(Y),
                X_accum.at[k, ...].set(X),
                W_accum.at[k, ...].set(W),
                Z_accum.at[k, ...].set(Z),
            )

        N, nu, R = u.shape
        nz, nx = self.C_z.shape
        ny, nw = self.D_yw.shape

        loop_init = (
            x0,
            jnp.zeros((N, ny, R)),  # Y_accum
            jnp.zeros((N, nx, R)),  # X_accum
            jnp.zeros((N, nw, R)),  # W_accum
            jnp.zeros((N, nz, R)),  # Z_accum
        )
        
        Y, X, W, Z = jax.lax.fori_loop(0, N, _make_step, loop_init)[1:]
        return Y, X, W, Z

    def simulate(
        self,
        u: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        offset: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the BLA model in the time domain for arbitrary input signals.

        Parameters
        ----------
        u : np.ndarray, shape (N,), (N, nu), (N, nu, R), or (N, nu, R, P)
            Input signal. This array can be 1D up to 4D, with:
            - N : number of time steps  
            - nu : number of inputs  
            - R : number of realizations  
            - P : number of periods  

            The input does not need to be normalized; this is handled internally.
            Since this is a public method (not used within an optimization loop), 
            the input also does not need to be periodic.

            If the fourth dimension P is provided, it is assumed that the input is
            periodic. In that case, all periods are internally concatenated into
            the first dimension, effectively simulating multiple periods sequentially.

        x0 : np.ndarray, shape (nx,) or (nx, R), optional
            Initial state for simulation. If not provided, the initial state is
            assumed to be zero.

        offset : int, optional
            Should only be provided if the input signal `u` is periodic. A non-negative
            integer representing the number of initial samples to prepend to the input
            signal to allow the system to reach steady-state before the main simulations
            begin. Those samples are not returned. If not provided, no samples are 
            prepended.

        Returns
        -------
        y : np.ndarray, shape (N, ny), (N, ny, R), or (N, ny, R, P)
            Simulated output time series, at least 2D, with ny as the number of
            output channels.

        t : np.ndarray, shape (N,)
            Time vector corresponding to one simulation of length N.

        x : np.ndarray, shape (N, nx), (N, nx, R), or (N, nx, R, P)
            Simulated state trajectories, at least 2D, with nx as the number of
            state variables.
        w : np.ndarray, shape (N, nw), (N, nw, R), or (N, nw, R, P)
            Simulated static nonlinear function outputs, at least 2D, with nw as
            the number of static nonlinear outputs.
        z : np.ndarray, shape (N, nz), (N, nz, R), or (N, nz, R, P)
            Simulated static nonlinear function inputs, at least 2D, with nz as
            the number of static nonlinear inputs.

        Raises
        ------
        ValueError
            If `u` has an invalid shape.
        ValueError
            If `x0` has an invalid shape.
        ValueError
            If `offset` is not a non-negative integer.

        """
        y, t, x, w, z = _simulate_core(self, u, x0=x0, offset=offset, with_wz=True)
        return y, t, x, w, z

    def num_parameters(self) -> int:
        """Return the total number of model parameters."""
        return (
            self.B_w.size + self.C_z.size + self.D_yw.size + self.D_zu.size
            + super().num_parameters() + self.func_static.num_parameters
        )
        
        
def _simulate_core(
    model: ModelBLA | ModelNonlinearLFR,
    u: np.ndarray,
    *,
    x0: np.ndarray | None,
    offset: int | None,
    with_wz: bool,
):
    """Simulate either a BLA or NL-LFR model for arbitrary input signals."""
    _validate_user_inputs(model, u, offset, x0)

    u_dim = u.ndim

    # Ensure `u` is 4D: (N, nu, R, P)
    u = u.reshape(u.shape + (1,) * (4 - u.ndim))
    N, nu, R, P = u.shape

    # Stack periods into the first dimension: (N * P, nu, R)
    u = jnp.transpose(u, (0, 3, 1, 2)).reshape(N * P, nu, R, order="F")

    if offset is not None:
        u = _misc.extend_signal(u, offset)

    nx = model.A.shape[0]
    x0 = jnp.zeros((nx, R)) if x0 is None else jnp.asarray(x0)

    # Normalize input
    u_mean = model.norm.u_mean.reshape(1, -1, 1)
    u_std = model.norm.u_std.reshape(1, -1, 1)
    u = (u - u_mean) / u_std

    u = jnp.asarray(u)

    # Call model-specific simulator
    if with_wz:
        y, x, w, z = model._simulate(u, x0)
    else:
        y, x = model._simulate(u, x0)
        w = z = None

    # Remove offset samples from outputs
    if offset is not None:
        y = y[offset:, ...]
        x = x[offset:, ...]
        if with_wz:
            w = w[offset:, ...]
            z = z[offset:, ...]

    # Denormalize output
    y_mean = model.norm.y_mean.reshape(1, -1, 1)
    y_std = model.norm.y_std.reshape(1, -1, 1)
    y = y * y_std + y_mean

    # Helper to reshape back to match input structure
    def _reshape_back(arr):
        arr = jnp.reshape(arr, (N, P, -1, R), order="F").transpose((0, 2, 3, 1))
        if u_dim in (1, 2):
            arr = jnp.squeeze(arr, axis=(2, 3))
        elif u_dim == 3:
            arr = jnp.squeeze(arr, axis=3)
        return arr

    y = _reshape_back(y)
    x = _reshape_back(x)
    if with_wz:
        w = _reshape_back(w)
        z = _reshape_back(z)

    t = np.arange(y.shape[0]) * model.ts

    if with_wz:
        return np.asarray(y), t, np.asarray(x), np.asarray(w), np.asarray(z)
    else:
        return np.asarray(y), t, np.asarray(x)


def _validate_user_inputs(
    model: ModelBLA | ModelNonlinearLFR,
    u: np.ndarray | jnp.ndarray,
    offset: int | None,
    x0: jnp.ndarray | None,
) -> None:
    
    if u.ndim != 1 and u.ndim != 2 and u.ndim != 3 and u.ndim != 4:
        raise ValueError(f"`u` must have 1 to 4 dimensions, got {u.ndim}D.")
    
    if x0 is not None:
        # Check if 1D or 2D
        if x0.ndim != 1 and x0.ndim != 2:
            raise ValueError(f"`x0` must be 1D or 2D, got {x0.ndim}D.")
        
        # Check consistency with `u`
        if u.ndim >= 3:
            if x0.ndim == 1:
                raise ValueError(
                    "`x0` must be 2D to match number of realizations in `u`."
                )
            if u.shape[2] != x0.shape[-1]:
                raise ValueError(
                    f"`x0` has {x0.shape[-1]} realizations, but `u` has "
                    f"{u.shape[2]} realizations."
                )
        else:
            if x0.ndim == 2:
                raise ValueError(
                    f"`x0` must be 1D since `u` is {u.ndim}D < 3D."
                )
            
        # Check consistency of state dimension
        if x0.shape[0] != model.A.shape[0]:
            raise ValueError(
                f"`x0` must have shape ({model.A.shape[0]}, ...), got {x0.shape}."
            )
    else:
        if u.ndim >= 3:
            x0 = jnp.zeros((model.A.shape[0], u.shape[2]))
        else:
            x0 = jnp.zeros((model.A.shape[0],))
            
    if offset is not None:
        if not (isinstance(offset, int) and offset >= 0):
            raise ValueError("`offset` must be a non-negative integer.")
 