"""Closed-loop simulation for the dual-motor drivetrain example."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> NDArray[np.floating]:
    """Simulate the dual-motor drivetrain closed loop.

    The interconnection follows ``dual-motor-drivetrain-block-diagram.drawio.svg``.
    The left controller receives ``r + y_left`` and the right controller receives
    ``r_1_right + y_right``. Their outputs are respectively added to the left and
    right channels of ``r_2`` before driving the plant. The returned channels are
    ordered ``[left, top, right]``.

    Parameters
    ----------
    K_left, K_right : ModelBLA or ModelNonlinearLFR
        Single-input, single-output models for the left and right controllers.
    plant : ModelBLA or ModelNonlinearLFR
        Two-input, three-output drivetrain model. Its input channels must be
        ordered ``[left, right]`` and its output channels ``[left, top, right]``.
    r : RealArray
        Left reference signal with shape ``(n_samples,)``, ``(n_samples, 1)``,
        ``(n_samples, 1, n_realizations)``, or
        ``(n_samples, 1, n_realizations, n_periods)``.
    r_1_right : RealArray
        Right reference signal with the same shape as ``r``. The block diagram
        obtains this signal by applying a gain of -1 to the common reference,
        but it is supplied independently here.
    r_2 : RealArray
        Additive plant-input excitation with shape ``(n_samples, 2)``,
        ``(n_samples, 2, n_realizations)``, or
        ``(n_samples, 2, n_realizations, n_periods)``. Channel 0 is left and
        channel 1 is right.
    tolerance : float, default=1e-9
        Absolute convergence tolerance for a per-sample fixed-point solve of
        direct feedthrough in the closed loop.
    max_iterations : int, default=100
        Maximum fixed-point iterations at each time sample and realization.

    Returns
    -------
    y : NDArray[np.floating]
        Simulated plant outputs, ordered ``[left, top, right]``. Its shape is
        ``(n_samples, 3)``, ``(n_samples, 3, n_realizations)``, or
        ``(n_samples, 3, n_realizations, n_periods)``, matching ``r_2``.

    Raises
    ------
    TypeError
        If a supplied model is not a supported model type.
    ValueError
        If model dimensions, sample times, signal shapes, or solver options are
        inconsistent, or if the direct-feedthrough loop does not converge.

    """
    _validate_models(K_left, K_right, plant)
    _validate_solver_options(tolerance, max_iterations)

    r_4d = _as_4d(r, "r", 1)
    r_1_right_4d = _as_4d(r_1_right, "r_1_right", 1)
    r_2_4d = _as_4d(r_2, "r_2", 2)
    _validate_signal_shapes(r_4d, r_1_right_4d, r_2_4d)

    n_samples, _, n_realizations, n_periods = r_2_4d.shape
    r_flat = _stack_periods(r_4d)
    r_1_right_flat = _stack_periods(r_1_right_4d)
    r_2_flat = _stack_periods(r_2_4d)
    n_total_samples = r_flat.shape[0]

    K_left_state = np.zeros((K_left.A.shape[0], n_realizations))
    K_right_state = np.zeros((K_right.A.shape[0], n_realizations))
    plant_state = np.zeros((plant.A.shape[0], n_realizations))
    y_flat = np.empty((n_total_samples, 3, n_realizations))
    y_previous = np.zeros((3, n_realizations))

    for sample in range(n_total_samples):
        y_current = y_previous
        for _ in range(max_iterations):
            u = _plant_input(
                K_left,
                K_right,
                K_left_state,
                K_right_state,
                y_current,
                r_flat[sample],
                r_1_right_flat[sample],
                r_2_flat[sample],
            )
            y_next, _ = _step_model(plant, plant_state, u)
            if np.max(np.abs(y_next - y_current)) <= tolerance:
                y_current = y_next
                break
            y_current = y_next
        else:
            msg = (
                "The direct-feedthrough loop did not converge at "
                f"sample {sample} within {max_iterations} iterations."
            )
            raise ValueError(msg)

        u = _plant_input(
            K_left,
            K_right,
            K_left_state,
            K_right_state,
            y_current,
            r_flat[sample],
            r_1_right_flat[sample],
            r_2_flat[sample],
        )
        e_left = r_flat[sample] + y_current[[0], :]
        e_right = r_1_right_flat[sample] + y_current[[2], :]
        _, K_left_state = _step_model(K_left, K_left_state, e_left)
        _, K_right_state = _step_model(K_right, K_right_state, e_right)
        y_current, plant_state = _step_model(plant, plant_state, u)
        y_flat[sample] = y_current
        y_previous = y_current

    return _restore_periods(y_flat, n_samples, n_periods, r_2.ndim)


def _plant_input(
    K_left: ModelBLA | ModelNonlinearLFR,
    K_right: ModelBLA | ModelNonlinearLFR,
    K_left_state: NDArray[np.floating],
    K_right_state: NDArray[np.floating],
    y: NDArray[np.floating],
    r: NDArray[np.floating],
    r_1_right: NDArray[np.floating],
    r_2: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Evaluate controller outputs without advancing their states."""
    e_left = r + y[[0], :]
    e_right = r_1_right + y[[2], :]
    u_left, _ = _step_model(K_left, K_left_state, e_left)
    u_right, _ = _step_model(K_right, K_right_state, e_right)
    return np.concatenate((u_left + r_2[[0], :], u_right + r_2[[1], :]), axis=0)


def _step_model(
    model: ModelBLA | ModelNonlinearLFR,
    x: NDArray[np.floating],
    u: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate one model time step in unnormalized signal coordinates."""
    u_normalized = (u - model.norm.u_mean[:, None]) / model.norm.u_std[:, None]
    if isinstance(model, ModelNonlinearLFR):
        z = np.asarray(model.C_z) @ x + np.asarray(model.D_zu) @ u_normalized
        w = np.asarray(model.func_static._evaluate(z.T).T)
        y_normalized = (
            np.asarray(model.C_y) @ x
            + np.asarray(model.D_yu) @ u_normalized
            + np.asarray(model.D_yw) @ w
        )
        x_next = (
            np.asarray(model.A) @ x
            + np.asarray(model.B_u) @ u_normalized
            + np.asarray(model.B_w) @ w
        )
    else:
        y_normalized = np.asarray(model.C_y) @ x + np.asarray(model.D_yu) @ u_normalized
        x_next = np.asarray(model.A) @ x + np.asarray(model.B_u) @ u_normalized
    y = y_normalized * model.norm.y_std[:, None] + model.norm.y_mean[:, None]
    return y, x_next


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


def _validate_solver_options(tolerance: float, max_iterations: int) -> None:
    """Validate direct-feedthrough solver options."""
    if tolerance <= 0:
        msg = "`tolerance` must be positive."
        raise ValueError(msg)
    if max_iterations < 1:
        msg = "`max_iterations` must be at least 1."
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
