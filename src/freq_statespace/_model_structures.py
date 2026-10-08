"""BLA and NL-LFR model classes, optimized for use with JAX and Equinox."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Self

from freq_statespace import _misc
from freq_statespace._data_manager import Normalizer
from freq_statespace._serialize import MODEL_REGISTRY, Serializable
from freq_statespace.static._nonlin_funcs import AbstractNonlinearFunction

if TYPE_CHECKING:
    from jaxtyping import Array, Complex, Float

    from freq_statespace._typing import RealArray


def _as_matrix(value: RealArray | float) -> Float[Array, "n_rows n_columns"]:
    """Convert a scalar or array-like value to a two-dimensional JAX array."""
    return jnp.atleast_2d(jnp.asarray(value))


def _identity_normalizer(nu: int, ny: int) -> Normalizer:
    """Create normalization statistics that leave input and output signals unchanged."""
    return Normalizer(
        u_mean=np.zeros(nu),
        u_std=np.ones(nu),
        y_mean=np.zeros(ny),
        y_std=np.ones(ny),
    )


@MODEL_REGISTRY.register
class ModelBLA(eqx.Module, Serializable):
    """BLA model class.

    Parameters
    ----------
    A : Float[Array, "nx nx"] or float
        State transition matrix.
    B_u : Float[Array, "nx nu"] or float
        Input-to-state matrix.
    C_y : Float[Array, "ny nx"] or float
        State-to-output matrix.
    D_yu : Float[Array, "ny nu"] or float
        Input-to-output matrix.
    ts : float
        Sampling time (in seconds) of the discrete system.
    norm : Normalizer, optional
        Means and standard deviations of input-output signals. If omitted,
        inputs and outputs are left unnormalized.

    """

    A: Float[Array, "nx nx"] = eqx.field(converter=_as_matrix)
    B_u: Float[Array, "nx nu"] = eqx.field(converter=_as_matrix)
    C_y: Float[Array, "ny nx"] = eqx.field(converter=_as_matrix)
    D_yu: Float[Array, "ny nu"] = eqx.field(converter=_as_matrix)
    ts: float
    norm: Normalizer
    _type_name: ClassVar[str] = "model_bla"

    def __init__(
        self,
        A: RealArray | float,
        B_u: RealArray | float,
        C_y: RealArray | float,
        D_yu: RealArray | float,
        ts: float,
        norm: Normalizer | None = None,
    ) -> None:
        """Initialize a BLA model.

        Scalars are accepted for all system matrices, which initializes a
        first-order SISO model. Array-like values are converted to matrices.

        Parameters
        ----------
        A : RealArray or float
            State transition matrix.
        B_u : RealArray or float
            Input-to-state matrix.
        C_y : RealArray or float
            State-to-output matrix.
        D_yu : RealArray or float
            Input-to-output matrix.
        ts : float
            Sampling time in seconds.
        norm : Normalizer, optional
            Input/output normalization statistics. If omitted, zero means and
            unit standard deviations are created using the model dimensions.

        """
        self.A = _as_matrix(A)
        self.B_u = _as_matrix(B_u)
        self.C_y = _as_matrix(C_y)
        self.D_yu = _as_matrix(D_yu)
        self.ts = ts
        nu = self.B_u.shape[1]
        ny = self.C_y.shape[0]
        self.norm = _identity_normalizer(nu, ny) if norm is None else norm
    
    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> Self:
        """Create a dummy PyTree with the same structure as a to-be-loaded model."""
        nu, ny, nx = config["nu"], config["ny"], config["nx"]
        norm = Normalizer(
            u_mean=np.zeros(nu),
            u_std=np.zeros(nu),
            y_mean=np.zeros(ny),
            y_std=np.zeros(ny),
        )
        return cls(
            A=jnp.zeros((nx, nx)),
            B_u=jnp.zeros((nx, nu)),
            C_y=jnp.zeros((ny, nx)),
            D_yu=jnp.zeros((ny, nu)),
            ts=0.0,
            norm=norm,
        )

    def simulate(
        self,
        u: RealArray,
        *,
        x0: RealArray | None = None,
        offset: int | None = None
    ) -> tuple[RealArray, RealArray, RealArray]:
        """Simulate the BLA model in the time domain for arbitrary input signals.

        Parameters
        ----------
        u : RealArray
            Shape (n_samples,), (n_samples, nu), (n_samples, nu, n_realizations),
            or (n_samples, nu, n_realizations, n_periods).
            Input signal. This array can be 1D up to 4D, with:
            - n_samples : number of time samples
            - nu : number of inputs  
            - n_realizations : number of realizations
            - n_periods : number of periods

            The input does not need to be normalized; this is handled internally.
            Since this is a public method (not used within an optimization loop), 
            the input also does not need to be periodic.

            If the fourth dimension n_periods is provided, it is assumed that the input is
            periodic. In that case, all periods are internally concatenated into
            the first dimension, effectively simulating multiple periods sequentially.

        x0 : RealArray, shape (nx,) or (nx, n_realizations), optional
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
        y : RealArray
            Shape (n_samples, ny), (n_samples, ny, n_realizations), or
            (n_samples, ny, n_realizations, n_periods).
            Simulated output time series, at least 2D, with ny as the number of
            output channels.

        t : RealArray, shape (n_samples,)
            Time vector corresponding to one simulation of length n_samples.

        x : RealArray
            Shape (n_samples, nx), (n_samples, nx, n_realizations), or
            (n_samples, nx, n_realizations, n_periods).
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
    
    def num_parameters(self) -> int:
        """Return the total number of model parameters."""
        return self.A.size + self.B_u.size + self.C_y.size + self.D_yu.size
    
    def _simulate(
        self,
        u: Float[Array, "n_samples nu n_realizations"],
        x0: Float[Array, "nx n_realizations"],
    ) -> tuple[
        Float[Array, "n_samples ny n_realizations"],
        Float[Array, "n_samples nx n_realizations"],
    ]:
        """Simulate the BLA model in the time domain.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        u : Float[Array, "n_samples nu n_realizations"]
            Normalized input signal.
        x0 : Float[Array, "nx n_realizations"]
            Initial state of the system. 

        Returns
        -------
        Y : Float[Array, "n_samples ny n_realizations"]
            Simulated output trajectories.
        X : Float[Array, "n_samples nx n_realizations"]
            Simulated state trajectories.
        W : Float[Array, "n_samples nw n_realizations"]
            Static nonlinear function outputs.
        Z : Float[Array, "n_samples nz n_realizations"]
            Static nonlinear function inputs.

        """
        def _make_step(k, state):
            X, Y_accum, X_accum = state
            U = jax.lax.dynamic_slice(u, (k, 0, 0), (1, nu, n_realizations)).squeeze(axis=0)

            # Model equations
            X_next = self.A @ X + self.B_u @ U
            Y = self.C_y @ X + self.D_yu @ U
            return X_next, Y_accum.at[k, ...].set(Y), X_accum.at[k, ...].set(X)

        n_samples, nu, n_realizations = u.shape
        ny, nx = self.C_y.shape

        loop_init = (
            x0,
            jnp.zeros((n_samples, ny, n_realizations)),  # Y_accum
            jnp.zeros((n_samples, nx, n_realizations)),  # X_accum
        )
        Y, X = jax.lax.fori_loop(0, n_samples, _make_step, loop_init)[1:]
        return Y, X

    def _frequency_response(
        self, freqs: RealArray
    ) -> Complex[Array, "n_bins ny nu"]:
        """Compute the frequency response of the system.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        freqs : RealArray, shape (n_bins,)
            Frequency points in Hz.

        Returns
        -------
        G : Complex[Array, "n_bins ny nu"]
            Frequency response matrix of shape (freqs, ny, nu).

        """
        def G(k):
            G_x = jnp.linalg.solve(zj[k] * I_nx - self.A, B_u)
            return C_y @ G_x + self.D_yu

        fs = 1 / self.ts
        z = 2 * jnp.pi * freqs / fs
        zj = jnp.exp(z * 1j)

        I_nx = jnp.eye(self.A.shape[0])
        B_u = self.B_u.astype(complex)  # to suppress a warning
        C_y = self.C_y.astype(complex)  # to suppress a warning
        return jax.vmap(G)(np.arange(len(freqs)))

    def frequency_response(
        self, freqs: RealArray
    ) -> Complex[Array, "n_bins ny nu"]:
        """Compute the frequency response in the original signal coordinates.

        Parameters
        ----------
        freqs : RealArray, shape (n_bins,)
            Frequency points in Hz.

        Returns
        -------
        Complex[Array, "n_bins ny nu"]
            Frequency response matrix in the original input and output units.
            The response describes deviations about the input and output means
            stored in ``norm``.

        """
        G_normalized = self._frequency_response(freqs)
        y_std = jnp.asarray(self.norm.y_std)[None, :, None]
        u_std = jnp.asarray(self.norm.u_std)[None, None, :]
        return G_normalized * y_std / u_std
    
    def _config_payload(self) -> dict[str, Any]:
        """Convert structural information to a dictionary for serialization."""
        return {
            "nu": self.B_u.shape[1],
            "ny": self.C_y.shape[0],
            "nx": self.A.shape[0],
        }
    

@MODEL_REGISTRY.register
class ModelNonlinearLFR(ModelBLA):
    """NL-LFR model class.

    Inherits from `ModelBLA` and adds linear matrices `B_w`, `C_z`, `D_yw`, `D_zu`,
    and static nonlinear feedback.
    """

    B_w: Float[Array, "nx nw"] = eqx.field(converter=_as_matrix)
    C_z: Float[Array, "nz nx"] = eqx.field(converter=_as_matrix)
    D_yw: Float[Array, "ny nw"] = eqx.field(converter=_as_matrix)
    D_zu: Float[Array, "nz nu"] = eqx.field(converter=_as_matrix)
    func_static: AbstractNonlinearFunction
    
    # Keep a private reference to the original BLA for initial state selection
    # during simulation. The optimize() routine may be called multiple times
    # (e.g. when re-running a Jupyter cell), which would otherwise overwrite the
    # original BLA. We retain this reference because the initial BLA has useful
    # properties (e.g. stability) for state initialization. During optimization,
    # the focus is on the overall NL-LFR performance, without explicitly maintaining
    # the numerical properties of the BLA component.
    _bla: ModelBLA = eqx.field(repr=False)
    _type_name: ClassVar[str] = "model_nllfr"
    
    def __init__(
        self,
        A: RealArray | float,
        B_u: RealArray | float,
        C_y: RealArray | float,
        D_yu: RealArray | float,
        B_w: RealArray | float,
        C_z: RealArray | float,
        D_yw: RealArray | float,
        D_zu: RealArray | float,
        func_static: AbstractNonlinearFunction,
        ts: float,
        norm: Normalizer | None = None,
    ) -> None:
        """Initialize NL-LFR model.

        Parameters
        ----------
        A : Float[Array, "nx nx"] or float
            State transition matrix.
        B_u : Float[Array, "nx nu"] or float
            Input-to-state matrix.
        C_y : Float[Array, "ny nx"] or float
            State-to-output matrix.
        D_yu : Float[Array, "ny nu"] or float
            Input-to-output matrix.
        B_w : Float[Array, "nx nw"] or float
            Feedback input-to-state matrix.
        C_z : Float[Array, "nz nx"] or float
            State-to-static nonlinear function matrix.
        D_yw : Float[Array, "ny nw"] or float
            Feedback input-to-output matrix.
        D_zu : Float[Array, "nz nu"] or float
            Input-to-static nonlinear function matrix.
        func_static : AbstractNonlinearFunction
            Static nonlinear function mapping `z` to `w`.
        ts : float
            Sampling time (in seconds) of the discrete system.
        norm : Normalizer, optional
            Input/output normalization statistics. If omitted, zero means and
            unit standard deviations are created using the model dimensions.

        """
        super().__init__(A, B_u, C_y, D_yu, ts, norm)
        self.B_w = _as_matrix(B_w)
        self.C_z = _as_matrix(C_z)
        self.D_yw = _as_matrix(D_yw)
        self.D_zu = _as_matrix(D_zu)
        self.func_static = func_static 
        self._bla = ModelBLA(
            self.A,
            self.B_u,
            self.C_y,
            self.D_yu,
            self.ts,
            self.norm,
        )
        
    @classmethod
    def _from_config(cls, config: dict[str, Any]) -> Self:
        """Create a dummy PyTree with the same structure as a to-be-loaded model."""
        from freq_statespace._serialize import NONLINEAR_FUNCTION_REGISTRY

        bla = ModelBLA._from_config(config)
        nu, ny, nx = config["nu"], config["ny"], config["nx"]
        func_static = NONLINEAR_FUNCTION_REGISTRY.from_config(config["func_static"])
        if not isinstance(func_static, AbstractNonlinearFunction):
            msg = "Deserialized func_static is not a nonlinear function."
            raise TypeError(msg)

        func_config = config["func_static"]["config"]
        nw = func_config["nw"]
        nz = func_config["nz"]
        return cls(
            A=bla.A,
            B_u=bla.B_u,
            C_y=bla.C_y,
            D_yu=bla.D_yu,
            B_w=jnp.zeros((nx, nw)),
            C_z=jnp.zeros((nz, nx)),
            D_yw=jnp.zeros((ny, nw)),
            D_zu=jnp.zeros((nz, nu)),
            func_static=func_static,
            ts=bla.ts,
            norm=bla.norm,
        )
        
    def simulate(
        self,
        u: RealArray,
        *,
        x0: RealArray | None = None,
        offset: int | None = None
    ) -> tuple[RealArray, RealArray, RealArray, RealArray, RealArray]:
        """Simulate the BLA model in the time domain for arbitrary input signals.

        Parameters
        ----------
        u : RealArray
            Shape (n_samples,), (n_samples, nu), (n_samples, nu, n_realizations),
            or (n_samples, nu, n_realizations, n_periods).
            Input signal. This array can be 1D up to 4D, with:
            - n_samples : number of time samples
            - nu : number of inputs  
            - n_realizations : number of realizations
            - n_periods : number of periods

            The input does not need to be normalized; this is handled internally.
            Since this is a public method (not used within an optimization loop), 
            the input also does not need to be periodic.

            If the fourth dimension n_periods is provided, it is assumed that the input is
            periodic. In that case, all periods are internally concatenated into
            the first dimension, effectively simulating multiple periods sequentially.

        x0 : RealArray, shape (nx,) or (nx, n_realizations), optional
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
        y : RealArray
            Shape (n_samples, ny), (n_samples, ny, n_realizations), or
            (n_samples, ny, n_realizations, n_periods).
            Simulated output time series, at least 2D, with ny as the number of
            output channels.

        t : RealArray, shape (n_samples,)
            Time vector corresponding to one simulation of length n_samples.

        x : RealArray
            Shape (n_samples, nx), (n_samples, nx, n_realizations), or
            (n_samples, nx, n_realizations, n_periods).
            Simulated state trajectories, at least 2D, with nx as the number of
            state variables.
        w : RealArray
            Shape (n_samples, nw), (n_samples, nw, n_realizations), or
            (n_samples, nw, n_realizations, n_periods).
            Simulated static nonlinear function outputs, at least 2D, with nw as
            the number of static nonlinear outputs.
        z : RealArray
            Shape (n_samples, nz), (n_samples, nz, n_realizations), or
            (n_samples, nz, n_realizations, n_periods).
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

    def _frequency_response(
        self, freqs: RealArray
    ) -> Complex[Array, "n_bins ny nu"]:
        """Compute the direct linear-path frequency response.

        This method does not include the nonlinear feedback and therefore does
        not represent the frequency response of the complete NL-LFR model.
        """
        _print_nllfr_frequency_response_warning()
        return super()._frequency_response(freqs)

    def frequency_response(
        self, freqs: RealArray
    ) -> Complex[Array, "n_bins ny nu"]:
        """Compute the direct linear-path response in original signal coordinates.

        Parameters
        ----------
        freqs : RealArray, shape (n_bins,)
            Frequency points in Hz.

        Returns
        -------
        Complex[Array, "n_bins ny nu"]
            Frequency response of the direct linear path in the original input
            and output units. It does not include the nonlinear feedback.

        """
        _print_nllfr_frequency_response_warning()
        G_normalized = ModelBLA._frequency_response(self, freqs)
        y_std = jnp.asarray(self.norm.y_std)[None, :, None]
        u_std = jnp.asarray(self.norm.u_std)[None, None, :]
        return G_normalized * y_std / u_std

    def _simulate(
        self,
        u: Float[Array, "n_samples nu n_realizations"],
        x0: Float[Array, "nx n_realizations"],
    ) -> tuple[
        Float[Array, "n_samples ny n_realizations"],
        Float[Array, "n_samples nx n_realizations"],
        Float[Array, "n_samples nw n_realizations"],
        Float[Array, "n_samples nz n_realizations"],
    ]:
        """Simulate the NL-LFR model in the time domain.

        To be used within an optimization loop, as it assumes normalized data.

        Parameters
        ----------
        u : Float[Array, "n_samples nu n_realizations"]
            Normalized input signal.
        x0 : Float[Array, "nx n_realizations"]
            Initial state of the system.

        Returns
        -------
        Y : Float[Array, "n_samples ny n_realizations"]
            Simulated output trajectories.
        X : Float[Array, "n_samples nx n_realizations"]
            Simulated state trajectories.
        W : Float[Array, "n_samples nw n_realizations"]
            Static nonlinear function outputs.
        Z : Float[Array, "n_samples nz n_realizations"]
            Static nonlinear function inputs.

        """
        def _make_step(k, state):
            X, Y_accum, X_accum, W_accum, Z_accum = state
            U = jax.lax.dynamic_slice(u, (k, 0, 0), (1, nu, n_realizations)).squeeze(axis=0)

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

        n_samples, nu, n_realizations = u.shape
        nz, nx = self.C_z.shape
        ny, nw = self.D_yw.shape

        loop_init = (
            x0,
            jnp.zeros((n_samples, ny, n_realizations)),  # Y_accum
            jnp.zeros((n_samples, nx, n_realizations)),  # X_accum
            jnp.zeros((n_samples, nw, n_realizations)),  # W_accum
            jnp.zeros((n_samples, nz, n_realizations)),  # Z_accum
        )
        
        Y, X, W, Z = jax.lax.fori_loop(0, n_samples, _make_step, loop_init)[1:]
        return Y, X, W, Z
    
    def _config_payload(self) -> dict[str, Any]:
        config = super()._config_payload()
        config["func_static"] = self.func_static.to_config()
        return config
        

def _print_nllfr_frequency_response_warning() -> None:
    """Print a warning about the direct linear-path response of an NL-LFR."""
    print(
        "Warning: The ModelNonlinearLFR frequency response returns only the "
        "direct linear path and does not include nonlinear feedback. An NL-LFR "
        "has no unique frequency response without a specified operating point "
        "and linearization."
    )


def _simulate_core(
    model: ModelBLA | ModelNonlinearLFR,
    u: RealArray,
    *,
    x0: RealArray | None,
    offset: int | None,
    with_wz: bool,
):
    """Simulate either a BLA or NL-LFR model for arbitrary input signals."""
    _validate_user_inputs(model, u, offset, x0)

    u_dim = u.ndim

    # Ensure `u` is 4D: (n_samples, nu, n_realizations, n_periods)
    u = u.reshape(u.shape + (1,) * (4 - u.ndim))
    n_samples, nu, n_realizations, n_periods = u.shape

    # Stack periods into the first dimension: (n_samples * n_periods, nu, n_realizations)
    u = jnp.transpose(u, (0, 3, 1, 2)).reshape(n_samples * n_periods, nu, n_realizations, order="F")

    if offset is not None:
        u = _misc.extend_signal(u, offset)

    nx = model.A.shape[0]
    x0 = jnp.zeros((nx, n_realizations)) if x0 is None else jnp.asarray(x0)

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
        arr = jnp.reshape(
            arr, (n_samples, n_periods, -1, n_realizations), order="F"
        ).transpose((0, 2, 3, 1))
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
    u: RealArray | Float[Array, "..."],
    offset: int | None,
    x0: Float[Array, "..."] | None,
) -> None:
    
    nu = u.shape[1] if u.ndim > 1 else 1
    if nu != model.B_u.shape[1] or nu != model.D_yu.shape[1]:
        msg = (
            f"Input signal has {nu} channel(s), but model expects "
            f"{model.B_u.shape[1]} channel(s)."
        )
        raise ValueError(msg)
    
    if u.ndim != 1 and u.ndim != 2 and u.ndim != 3 and u.ndim != 4:
        msg = f"`u` must have 1 to 4 dimensions, got {u.ndim}D."
        raise ValueError(msg)
    
    if x0 is not None:
        # Check if 1D or 2D
        if x0.ndim != 1 and x0.ndim != 2:
            msg = f"`x0` must be 1D or 2D, got {x0.ndim}D."
            raise ValueError(msg)
        
        # Check consistency with `u`
        if u.ndim >= 3:
            if x0.ndim == 1:
                msg = "`x0` must be 2D to match number of realizations in `u`."
                raise ValueError(msg)
            if u.shape[2] != x0.shape[-1]:
                msg = (
                    f"`x0` has {x0.shape[-1]} realizations, but `u` has "
                    f"{u.shape[2]} realizations."
                )
                raise ValueError(msg)
        else:
            if x0.ndim == 2:
                msg = f"`x0` must be 1D since `u` is {u.ndim}D < 3D."
                raise ValueError(msg)
            
        # Check consistency of state dimension
        if x0.shape[0] != model.A.shape[0]:
            msg = f"`x0` must have shape ({model.A.shape[0]}, ...), got {x0.shape}."
            raise ValueError(msg)
    else:
        if u.ndim >= 3:
            x0 = jnp.zeros((model.A.shape[0], u.shape[2]))
        else:
            x0 = jnp.zeros((model.A.shape[0],))
            
    if offset is not None:
        if not (isinstance(offset, int) and offset >= 0):
            msg = "`offset` must be a non-negative integer."
            raise ValueError(msg)
 
