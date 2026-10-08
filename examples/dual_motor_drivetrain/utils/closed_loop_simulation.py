"""Closed-loop model simulation for the dual-motor drivetrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from freq_statespace import ModelBLA, ModelNonlinearLFR

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from freq_statespace._typing import RealArray


def simulate_closed_loop(
    r_1: RealArray,
    r_2: RealArray,
    K_left: ModelBLA | ModelNonlinearLFR,
    K_right: ModelBLA | ModelNonlinearLFR,
    plant: ModelBLA | ModelNonlinearLFR,
) -> NDArray[np.floating]:
    """Simulate the closed loop shown in the drivetrain block diagram.

    Both controllers receive the top-motor tracking error
    ``e = r_1 - y_top``. Their outputs are the left and right plant inputs,
    respectively, and ``r_2`` is added to those two inputs. The plant output
    order is ``[left, top, right]``.

    Parameters
    ----------
    r_1
        Top-motor reference with shape ``(N,)``, ``(N, 1)``,
        ``(N, 1, R)``, or ``(N, 1, R, P)``.
    r_2
        Additive left/right plant-input excitation with shape ``(N, 2)``,
        ``(N, 2, R)``, or ``(N, 2, R, P)``. Channel order is ``[left,
        right]``.
    K_left, K_right
        Single-input, single-output left and right controller models.
    plant
        Strictly proper two-input, three-output drivetrain model.

    Returns
    -------
    NDArray[np.floating]
        Simulated ``[left, top, right]`` output. The realization and period
        dimensions match those supplied in ``r_1``.

    Raises
    ------
    TypeError
        If a model is not a supported model type.
    ValueError
        If model dimensions, sample times, or the reference shape are
        incompatible, or if the plant has direct feedthrough.

    """
    _validate_models(K_left, K_right, plant)
    r_1_array = np.asarray(r_1)
    r_1_4d = _as_4d(r_1_array, "r_1", 1)
    r_2_4d = _as_4d(np.asarray(r_2), "r_2", 2)
    _validate_signal_shapes(r_1_4d, r_2_4d)
    n_samples, _, n_realizations, n_periods = r_1_4d.shape
    r_1_flat = jnp.asarray(_stack_periods(r_1_4d))
    r_2_flat = jnp.asarray(_stack_periods(r_2_4d))

    initial_state = (
        jnp.zeros((plant.A.shape[0], n_realizations)),
        jnp.zeros((K_left.A.shape[0], n_realizations)),
        jnp.zeros((K_right.A.shape[0], n_realizations)),
    )
    zero_plant_input = jnp.zeros((2, n_realizations))

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ):
        plant_state, K_left_state, K_right_state = carry
        r_1_current, r_2_current = inputs
        y_current, _ = _step_model_jax(plant, plant_state, zero_plant_input)
        error = r_1_current - y_current[[1], :]
        u_left, K_left_next = _step_model_jax(K_left, K_left_state, error)
        u_right, K_right_next = _step_model_jax(K_right, K_right_state, error)
        plant_input = jnp.concatenate((u_left, u_right), axis=0) + r_2_current
        _, plant_next = _step_model_jax(plant, plant_state, plant_input)
        return (plant_next, K_left_next, K_right_next), y_current

    scan = jax.jit(lambda r_1_scan, r_2_scan: jax.lax.scan(step, initial_state, (r_1_scan, r_2_scan)))
    _, y_flat = scan(r_1_flat, r_2_flat)
    return _restore_periods(np.asarray(y_flat), n_samples, n_periods, r_1_array.ndim)


def _step_model_jax(
    model: ModelBLA | ModelNonlinearLFR,
    state: jax.Array,
    input_signal: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate one normalized model time step with JAX arrays."""
    u_mean = jnp.asarray(model.norm.u_mean).reshape(-1, 1)
    u_std = jnp.asarray(model.norm.u_std).reshape(-1, 1)
    y_mean = jnp.asarray(model.norm.y_mean).reshape(-1, 1)
    y_std = jnp.asarray(model.norm.y_std).reshape(-1, 1)
    normalized_input = (input_signal - u_mean) / u_std
    if isinstance(model, ModelNonlinearLFR):
        z = model.C_z @ state + model.D_zu @ normalized_input
        w = model.func_static._evaluate(z.T).T
        normalized_output = model.C_y @ state + model.D_yu @ normalized_input + model.D_yw @ w
        next_state = model.A @ state + model.B_u @ normalized_input + model.B_w @ w
    else:
        normalized_output = model.C_y @ state + model.D_yu @ normalized_input
        next_state = model.A @ state + model.B_u @ normalized_input
    return normalized_output * y_std + y_mean, next_state


def _as_4d(signal: NDArray[np.floating], name: str, n_channels: int) -> NDArray[np.floating]:
    """Validate a public input signal and represent it as ``(N, nu, R, P)``."""
    if signal.ndim < 1 or signal.ndim > 4:
        msg = f"`{name}` must have 1 to 4 dimensions, got {signal.ndim}D."
        raise ValueError(msg)
    signal_channels = signal.shape[1] if signal.ndim > 1 else 1
    if signal_channels != n_channels:
        msg = f"`{name}` must have {n_channels} channel(s), got {signal_channels}."
        raise ValueError(msg)
    return signal.reshape(signal.shape + (1,) * (4 - signal.ndim))


def _validate_signal_shapes(r_1: NDArray[np.floating], r_2: NDArray[np.floating]) -> None:
    """Ensure both references describe the same simulations."""
    r_1_shape = (r_1.shape[0], r_1.shape[2], r_1.shape[3])
    r_2_shape = (r_2.shape[0], r_2.shape[2], r_2.shape[3])
    if r_2_shape != r_1_shape:
        msg = f"`r_2` has simulation shape {r_2_shape}, but `r_1` has simulation shape {r_1_shape}."
        raise ValueError(msg)


def _validate_models(
    K_left: ModelBLA | ModelNonlinearLFR,
    K_right: ModelBLA | ModelNonlinearLFR,
    plant: ModelBLA | ModelNonlinearLFR,
) -> None:
    """Validate dimensions, sample times, and a well-posed interconnection."""
    for name, model, expected_shape in (
        ("K_left", K_left, (1, 1)),
        ("K_right", K_right, (1, 1)),
        ("plant", plant, (2, 3)),
    ):
        if not isinstance(model, (ModelBLA, ModelNonlinearLFR)):
            msg = f"`{name}` must be a ModelBLA or ModelNonlinearLFR."
            raise TypeError(msg)
        model_shape = (model.B_u.shape[1], model.C_y.shape[0])
        if model_shape != expected_shape:
            msg = f"`{name}` must have shape {expected_shape} (nu, ny), got {model_shape}."
            raise ValueError(msg)
    if not np.isclose(K_left.ts, K_right.ts) or not np.isclose(K_left.ts, plant.ts):
        msg = "K_left, K_right, and plant must have the same sampling time."
        raise ValueError(msg)
    if np.any(np.asarray(plant.D_yu) != 0):
        msg = "`plant.D_yu` must be zero to avoid a closed-loop algebraic loop."
        raise ValueError(msg)


def _stack_periods(signal: NDArray[np.floating]) -> NDArray[np.floating]:
    """Stack periods in the sequential order used by model simulation."""
    n_samples, n_inputs, n_realizations, n_periods = signal.shape
    return signal.transpose(0, 3, 1, 2).reshape(
        n_samples * n_periods, n_inputs, n_realizations, order="F"
    )


def _restore_periods(
    signal: NDArray[np.floating],
    n_samples: int,
    n_periods: int,
    reference_ndim: int,
) -> NDArray[np.floating]:
    """Restore the public output shape after sequential period simulation."""
    output = np.reshape(signal, (n_samples, n_periods, 3, -1), order="F").transpose(0, 2, 3, 1)
    if reference_ndim <= 2:
        return np.squeeze(output, axis=(2, 3))
    if reference_ndim == 3:
        return np.squeeze(output, axis=3)
    return output
