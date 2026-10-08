"""Generate deterministic excitation signals for all closed-loop experiments."""

from __future__ import annotations

from pathlib import Path

import multisine
import numpy as np

try:
    from .utils.load_profiles import generate_random_load_profile
except ImportError:  # Supports ``python utils/generate_excitation_signals.py``.
    from utils.load_profiles import generate_random_load_profile


# Experiment configuration
RANDOM_SEED = 42
SAMPLING_FREQUENCY = 100.0
GEARBOX_RATIO = 28.0
SETPOINT_AMPLITUDE_SHAFT_REV_PER_SEC = 3.0
SETPOINT_PERTURBATION_AMPLITUDE_SHAFT_REV_PER_SEC = 0.5
PLANT_INPUT_PERTURBATION_AMPLITUDE = 200.0
LOAD_TORQUE_AMPLITUDE = 150.0
LOAD_TORQUE_CORRELATION_TIME = 1.0
N_INPUTS = 2
N_SAMPLES = 1024
N_PERIODS = 4
F_MAX = 10.0
N_CONTROLLER_ID_REALIZATIONS = 2
N_PLANT_ID_EXPERIMENTS = 3
N_VALIDATION_EXPERIMENTS = 1

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
SIGNAL_DIRECTORY = DATA_DIRECTORY / "_signals"
SHAFT_REV_PER_SEC_TO_GEARBOX_RAD_PER_SEC = 2.0 * np.pi * GEARBOX_RATIO


def _next_seed(rng: np.random.Generator) -> int:
    """Draw a positive seed for one deterministic signal-generator call."""
    return int(rng.integers(1, np.iinfo(np.int64).max))


def _generate_controller_id_signals(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate reference and load trajectories for controller identification."""
    r_1 = multisine.random_phase_multisine(
        n_samples=N_SAMPLES,
        fs=SAMPLING_FREQUENCY,
        amplitude=SETPOINT_PERTURBATION_AMPLITUDE_SHAFT_REV_PER_SEC,
        f_max=F_MAX,
        n_realizations=N_CONTROLLER_ID_REALIZATIONS,
        seed=_next_seed(rng),
    ).u
    r_1 = (SETPOINT_AMPLITUDE_SHAFT_REV_PER_SEC + r_1) * SHAFT_REV_PER_SEC_TO_GEARBOX_RAD_PER_SEC
    tau_load = generate_random_load_profile(
        n_samples=N_SAMPLES * N_PERIODS,
        fs=SAMPLING_FREQUENCY,
        amplitude=LOAD_TORQUE_AMPLITUDE,
        correlation_time=LOAD_TORQUE_CORRELATION_TIME,
        n_realizations=N_CONTROLLER_ID_REALIZATIONS,
        seed=_next_seed(rng),
    )
    return {"r_1_controller_id.npy": r_1, "tau_load_controller_id.npy": tau_load}


def _generate_plant_id_signals(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate setpoint, orthogonal input excitation, and load trajectories."""
    n_realizations = N_INPUTS * N_PLANT_ID_EXPERIMENTS
    r_1 = np.full(
        (N_SAMPLES, 1, n_realizations),
        SETPOINT_AMPLITUDE_SHAFT_REV_PER_SEC * SHAFT_REV_PER_SEC_TO_GEARBOX_RAD_PER_SEC,
    )
    r_2 = np.ascontiguousarray(
        multisine.random_phase_orthogonal_multisine(
            n_samples=N_SAMPLES,
            fs=SAMPLING_FREQUENCY,
            nu=N_INPUTS,
            amplitude=PLANT_INPUT_PERTURBATION_AMPLITUDE,
            f_max=F_MAX,
            n_experiments=N_PLANT_ID_EXPERIMENTS,
            seed=_next_seed(rng),
        ).u.reshape((N_SAMPLES, N_INPUTS, n_realizations), order="F")
    )
    tau_load = generate_random_load_profile(
        n_samples=N_SAMPLES * N_PERIODS,
        fs=SAMPLING_FREQUENCY,
        amplitude=LOAD_TORQUE_AMPLITUDE,
        correlation_time=LOAD_TORQUE_CORRELATION_TIME,
        n_realizations=n_realizations,
        seed=_next_seed(rng),
    )
    return {
        "r_1_plant_id.npy": r_1,
        "r_2_plant_id.npy": r_2,
        "tau_load_plant_id.npy": tau_load,
    }


def _generate_validation_signals(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate repeated validation references without a load trajectory."""
    n_realizations = N_INPUTS * N_VALIDATION_EXPERIMENTS
    r_1_one_period = np.full(
        (N_SAMPLES, 1, n_realizations),
        SETPOINT_AMPLITUDE_SHAFT_REV_PER_SEC * SHAFT_REV_PER_SEC_TO_GEARBOX_RAD_PER_SEC,
    )
    r_2_one_period = np.ascontiguousarray(
        multisine.random_phase_orthogonal_multisine(
            n_samples=N_SAMPLES,
            fs=SAMPLING_FREQUENCY,
            nu=N_INPUTS,
            amplitude=PLANT_INPUT_PERTURBATION_AMPLITUDE,
            f_max=F_MAX,
            n_experiments=N_VALIDATION_EXPERIMENTS,
            seed=_next_seed(rng),
        ).u.reshape((N_SAMPLES, N_INPUTS, n_realizations), order="F")
    )
    return {
        "r_1_validation.npy": np.repeat(r_1_one_period[..., None], N_PERIODS, axis=3),
        "r_2_validation.npy": np.repeat(r_2_one_period[..., None], N_PERIODS, axis=3),
    }


def generate_all() -> dict[str, tuple[int, ...]]:
    """Generate and save all experiment inputs, returning their shapes by file name."""
    rng = np.random.default_rng(RANDOM_SEED)
    signals = (
        _generate_controller_id_signals(rng)
        | _generate_plant_id_signals(rng)
        | _generate_validation_signals(rng)
    )
    SIGNAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename, signal in signals.items():
        np.save(SIGNAL_DIRECTORY / filename, signal)
    return {filename: signal.shape for filename, signal in signals.items()}


if __name__ == "__main__":
    for filename, shape in generate_all().items():
        print(f"Wrote {SIGNAL_DIRECTORY / filename} with shape {shape}")
