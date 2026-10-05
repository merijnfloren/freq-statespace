"""Closed-loop simulation for the dual-motor drivetrain example."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from freq_statespace import ModelBLA, ModelNonlinearLFR

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from freq_statespace._typing import RealArray


def simulate(
    K_left: ModelBLA | ModelNonlinearLFR,
    K_right: ModelBLA | ModelNonlinearLFR,
    plant: ModelBLA | ModelNonlinearLFR,
    r: RealArray,
    r_1_right: RealArray,
    r_2: RealArray,
) -> NDArray[np.floating]:
    """Simulate the dual-motor drivetrain closed loop with a compiled recurrence.

    The interconnection follows ``dual-motor-drivetrain-block-diagram.drawio.svg``.
    The left controller receives ``r - y_left`` and the right controller receives
    ``-r + r_1_right - y_right``. Their outputs are respectively added to the
    left and right channels of ``r_2`` before driving the plant. The sequential
    time recurrence is compiled with :func:`jax.lax.scan`, while all realizations
    are processed in parallel.

    Parameters
    ----------
    K_left, K_right : ModelBLA or ModelNonlinearLFR
        Single-input, single-output models for the left and right controllers.
    plant : ModelBLA or ModelNonlinearLFR
        Strictly proper two-input, three-output drivetrain model. Input channels
        are ordered ``[left, right]`` and output channels are ordered
        ``[left, top, right]``.
    r : RealArray
        Left reference signal with shape ``(n_samples,)``, ``(n_samples, 1)``,
        ``(n_samples, 1, n_realizations)``, or
        ``(n_samples, 1, n_realizations, n_periods)``.
    r_1_right : RealArray
        Right reference signal with the same shape as ``r``.
    r_2 : RealArray
        Additive plant-input excitation with shape ``(n_samples, 2)``,
        ``(n_samples, 2, n_realizations)``, or
        ``(n_samples, 2, n_realizations, n_periods)``. Channel 0 is left and
        channel 1 is right.

    Returns
    -------
    y : NDArray[np.floating]
        Simulated plant outputs in ``[left, top, right]`` order. Its shape
        matches the public simulation shape implied by ``r_2``.

    Raises
    ------
    TypeError
        If a supplied model is not a supported model type.
    ValueError
        If model dimensions, sample times, or signal shapes are inconsistent, or
        if the plant has nonzero direct feedthrough.

    """
    _validate_models(K_left, K_right, plant)

    r_4d = _as_4d(r, "r", 1)
    r_1_right_4d = _as_4d(r_1_right, "r_1_right", 1)
    r_2_4d = _as_4d(r_2, "r_2", 2)
    _validate_signal_shapes(r_4d, r_1_right_4d, r_2_4d)

    n_samples, _, n_realizations, n_periods = r_2_4d.shape
    r_flat = jnp.asarray(_stack_periods(r_4d))
    r_1_right_flat = jnp.asarray(_stack_periods(r_1_right_4d))
    r_2_flat = jnp.asarray(_stack_periods(r_2_4d))
    initial_state = (
        jnp.zeros((plant.A.shape[0], n_realizations)),
        jnp.zeros((K_left.A.shape[0], n_realizations)),
        jnp.zeros((K_right.A.shape[0], n_realizations)),
    )
    zero_plant_input = jnp.zeros((2, n_realizations))

    def _step(carry, inputs):
        plant_state, K_left_state, K_right_state = carry
        r_current, r_1_right_current, r_2_current = inputs

        y_current, _ = _step_model_jax(plant, plant_state, zero_plant_input)
        e_left = r_current - y_current[[0], :]
        e_right = -r_current + r_1_right_current - y_current[[2], :]
        u_left, K_left_next = _step_model_jax(K_left, K_left_state, e_left)
        u_right, K_right_next = _step_model_jax(K_right, K_right_state, e_right)
        u = jnp.concatenate(
            (u_left + r_2_current[[0], :], u_right + r_2_current[[1], :]),
            axis=0,
        )
        _, plant_next = _step_model_jax(plant, plant_state, u)
        return (plant_next, K_left_next, K_right_next), y_current

    scan = jax.jit(
        lambda r_scan, r_1_right_scan, r_2_scan: jax.lax.scan(
            _step, initial_state, (r_scan, r_1_right_scan, r_2_scan)
        )
    )
    _, y_flat = scan(r_flat, r_1_right_flat, r_2_flat)
    return _restore_periods(np.asarray(y_flat), n_samples, n_periods, r_2.ndim)


def _step_model_jax(
    model: ModelBLA | ModelNonlinearLFR,
    x: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate one model time step with JAX arrays."""
    u_mean = jnp.asarray(model.norm.u_mean).reshape(-1, 1)
    u_std = jnp.asarray(model.norm.u_std).reshape(-1, 1)
    y_mean = jnp.asarray(model.norm.y_mean).reshape(-1, 1)
    y_std = jnp.asarray(model.norm.y_std).reshape(-1, 1)
    u_normalized = (u - u_mean) / u_std
    if isinstance(model, ModelNonlinearLFR):
        z = model.C_z @ x + model.D_zu @ u_normalized
        w = model.func_static._evaluate(z.T).T
        y_normalized = model.C_y @ x + model.D_yu @ u_normalized + model.D_yw @ w
        x_next = model.A @ x + model.B_u @ u_normalized + model.B_w @ w
    else:
        y_normalized = model.C_y @ x + model.D_yu @ u_normalized
        x_next = model.A @ x + model.B_u @ u_normalized
    return y_normalized * y_std + y_mean, x_next


def _as_4d(signal: RealArray, name: str, nu: int) -> NDArray[np.floating]:
    """Validate a public simulation input and represent it as a 4D array."""
    signal = np.asarray(signal)
    if signal.ndim < 1 or signal.ndim > 4:
        msg = f"`{name}` must have 1 to 4 dimensions, got {signal.ndim}D."
        raise ValueError(msg)
    n_signal_inputs = signal.shape[1] if signal.ndim > 1 else 1
    if n_signal_inputs != nu:
        msg = f"`{name}` must have {nu} input channel(s), got {n_signal_inputs}."
        raise ValueError(msg)
    return signal.reshape(signal.shape + (1,) * (4 - signal.ndim))


def _validate_signal_shapes(
    r: NDArray[np.floating],
    r_1_right: NDArray[np.floating],
    r_2: NDArray[np.floating],
) -> None:
    """Ensure all three signals describe the same simulations."""
    reference_shape = (r.shape[0], r.shape[2], r.shape[3])
    for name, signal in (("r_1_right", r_1_right), ("r_2", r_2)):
        signal_shape = (signal.shape[0], signal.shape[2], signal.shape[3])
        if signal_shape != reference_shape:
            msg = (
                f"`{name}` has simulation shape {signal_shape}, but `r` has "
                f"simulation shape {reference_shape}."
            )
            raise ValueError(msg)


def _validate_models(
    K_left: ModelBLA | ModelNonlinearLFR,
    K_right: ModelBLA | ModelNonlinearLFR,
    plant: ModelBLA | ModelNonlinearLFR,
) -> None:
    """Validate model types, dimensions, and compatible sample times."""
    for name, model, expected_shape in (
        ("K_left", K_left, (1, 1)),
        ("K_right", K_right, (1, 1)),
        ("plant", plant, (2, 3)),
    ):
        if not isinstance(model, ModelBLA):
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
    """Match the period-stacking convention of ModelBLA.simulate."""
    n_samples, nu, n_realizations, n_periods = signal.shape
    return signal.transpose(0, 3, 1, 2).reshape(
        n_samples * n_periods, nu, n_realizations, order="F"
    )


def _restore_periods(
    signal: NDArray[np.floating],
    n_samples: int,
    n_periods: int,
    input_ndim: int,
) -> NDArray[np.floating]:
    """Restore the public signal shape after sequential period simulation."""
    signal = np.reshape(signal, (n_samples, n_periods, 3, -1), order="F").transpose(0, 2, 3, 1)
    if input_ndim == 2:
        return np.squeeze(signal, axis=(2, 3))
    if input_ndim == 3:
        return np.squeeze(signal, axis=3)
    return signal
