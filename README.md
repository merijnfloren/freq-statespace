# freq-statespace
A flexible [JAX](https://docs.jax.dev/en/latest/index.html)-based package for nonlinear state-space identification using frequency-domain optimization techniques, focusing on the *nonlinear Linear Fractional Representation* (NL-LFR) model structure, which combines an LTI system with a static feedback nonlinearity. This internal feedback formulation allows for capturing many complex real-world dynamics, with other popular block-oriented structures such as Wiener, Hammerstein, and Wiener-Hammerstein models arising as special cases. Identification of standard linear state-space models is also supported.

<div align="center">
  <img src="https://github.com/merijnfloren/freq-statespace/raw/main/docs/model_structure.svg" width="500px" />
</div>

## Basic usage

The package works with (multiple periods and realizations of) input–output data sequences $u(n)$ and $y(n)$ for $n = 0, \ldots, n_{\mathrm{samples}}-1$, assuming periodic excitation and an integer number of steady-state output periods. The specific NL-LFR structure is defined as:

```math
  \begin{align*}
    x(n+1) &= A x(n) + B_u u(n) + B_w w(n),\\
    y(n) &= C_y x(n) + D_{yu} u(n) + D_{yw} w(n),\\
    z(n) &= C_z x(n) + D_{zu} u(n),\\ 
    w(n) &= f\big(z(n)\big),
  \end{align*}
```

consisting of linear state-space matrices and a static nonlinear function approximator $f(\cdot)$.

A typical step-wise identification procedure is as follows:

1. *Best Linear Approximation (BLA) parametrization.* Initializes the matrices $A$, $B_u$, $C_y$ and $D_{yu}$ using the [frequency-domain subspace method](https://github.com/tomasmckelvey/fsid), and refines these estimates through iterative optimization. If you're only interested in linear state-space models, you can stop the identification process here.
2. *NL-LFR initialization.* Applies the [frequency-domain inference and learning method](https://arxiv.org/abs/2503.14409) to efficiently initialize the remaining model parameters while keeping the BLA parameters fixed. This step requires that $f(\cdot)$ is linear in the parameters, i.e., $f(\cdot)=\beta^\top\phi(\cdot)$, with $\beta$ the parameter matrix and $\phi(\cdot)$ the nonlinear feature mapping (e.g., polynomial features).
3. *NL-LFR optimization.* Performs iterative refinement of all model parameters based on time-domain simulations. This is the most demanding step, mainly due to the sequential nature of the forward simulations and the corresponding non-convexity of the optimization problem. However, the previous steps aim to mitigate this difficulty by providing an initialization that is already close to a good local minimum.

It is also possible to skip the inference and learning step and go straight to nonlinear optimization. An advantage of this approach is that it puts no restriction on the structure of $f(\cdot)$, i.e., it does not require a model that is linear in its parameters.

## Features

- Provides a user-friendly interface for identifying _linear_ state-space models using frequency-domain subspace estimation based on the nonparametric BLA.
- Offers two workflows for identifying _nonlinear_ LFR state-space models by primarily exploiting a frequency-domain formulation that enables inherent parallelism.
- Uses JAX for automatic differentiation, JIT compilation, and GPU/TPU acceleration.
- Supports [Optimistix](https://docs.kidger.site/optimistix/) solvers (Levenberg-Marquardt, BFGS, ...) for typical system identification problems.
- Supports [Optax](https://optax.readthedocs.io/en/latest/) optimizers (Adam, SGD, ...) for large-scale optimization.

## Installation

Requires Python 3.12 or newer:

```bash
pip install freq-statespace
```

Or add it to a `uv` project:

```bash
uv add freq-statespace
```

If JAX isn't already installed in your environment, the above commands will install the CPU-only version. For GPU/TPU support (strongly recommended, often many times faster for mid-size to large problems), follow the [JAX installation guide](https://github.com/google/jax#installation).

## Quick example

We show an exemplary training pipeline on the [Silverbox benchmark dataset](https://www.nonlinearbenchmark.org/benchmarks/silverbox), containing input-output measurements from an electronic circuit that mimics a mass-spring-damper system with a cubic spring nonlinearity. Note that running this example will download the benchmark data to the local cache when it is not already available.

We first estimate the BLA:

```python
import freq_statespace as fss

data = fss.load_and_preprocess_silverbox_data()  # 8192 x 6 samples

# Step 1: BLA estimation
nx = 2  # state dimension
bla = fss.lin.subspace_id(data, nx)  # NRMSE 18.36%, non-iterative
bla = fss.lin.optimize(bla, data)  # NRMSE 13.17%, 6 iters, 1.32ms/iter
```

Next, we proceed with inference and learning, followed by full nonlinear optimization:

```python
# Step 2: Inference and learning
nw, nz = 1, 1  # internal signal dimensions
phi = fss.static.basis.Polynomial(nz, degree=3)
nllfr = fss.nonlin.inference_and_learning(bla, data, phi, nw)  # NRMSE 1.11%, 45 iters, 18.4ms/iter

# Step 3: Nonlinear optimization
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.44%, 100 iters, 387ms/iter
```

> **Note:** Inference and learning requires substantially less computation time per iteration (measured on an NVIDIA T600 Laptop GPU), as these operations are largely parallelizable.

Alternatively, we could skip inference and learning and jump straight to nonlinear optimization. In this example we use a neural network:

```python
import jax

# Step 2: Nonlinear optimization
nw, nz = 1, 1  # internal signal dimensions
neural_net = fss.static.NeuralNetwork(nw, nz, layers=1, neurons_per_layer=10, activation=jax.nn.relu)
nllfr = fss.nonlin.connect(bla, neural_net)
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.55%, 100 iters, 354ms/iter
```

Serialization of models can be achieved like so:

```python
path = "models/nllfr.zip"
fss.save_model(nllfr, path)
nllfr_loaded = fss.load_model(path)
```

The `examples/` folder also contains Jupyter notebooks covering more challenging benchmark systems and a closed-loop identification scenario, with additional guidance on hyperparameter tuning and solver configuration.

## Preparing your data

Choose one of the following entry points to create an `InputOutputData` object:

- `fss.create_data_object(...)` accepts time-domain input-output data and minimal frequency metadata; only the sampling frequency is strictly required.
- `fss.create_data_object_from_bla(...)` accepts a BLA estimate produced by the [`best-linear-approximation`](https://github.com/merijnfloren/best-linear-approximation) package. Use this when a specialized method, such as a closed-loop BLA method, has already computed the BLA and you want to fit a parametric BLA model to it.

Once the data object is instantiated by either function, the workflow proceeds exactly as in the quick example above.

## Related packages

Looking for:

- multisine excitation signals? See [multisine](https://github.com/merijnfloren/multisine).
- nonparametric estimation of the best linear approximation? See [best-linear-approximation](https://github.com/merijnfloren/best-linear-approximation).

## Citation

If you use this package in your work, please cite:

```bibtex
@inproceedings{floren2026freq,
  title={Freq-statespace: a Python package for identification of (non)linear state-space models from periodic input-output data},
  author={Floren, Merijn and Swevers, Jan},
  booktitle={45th Benelux Meeting on Systems and Control},
  year={2026}
}
```

If you use the inference and learning method, please also cite the corresponding [paper](https://arxiv.org/abs/2503.14409):

```bibtex
@article{floren2025inference,
  title={Inference and Learning of Nonlinear LFR State-Space Models},
  author={Floren, Merijn and No{\"e}l, Jean-Philippe and Swevers, Jan},
  journal={IEEE Control Systems Letters},
  year={2025},
  publisher={IEEE}
}
```

<!-- BEGIN: reusable-result capability description -->
## For AI-assisted method selection

<!--
Maintained for the CoMoDO reusable result.
Template: comodo-reusable-result/templates/capability-section.md
This section is read by the `find-methods` skill to match this repository
against an external user's application. Keep it accurate; update it when
examples are added or removed.
-->

### Identity

- **One-line summary:** Fit linear or nonlinear, black-box state-space models to excited input-output measurements, primarily periodic multisine experiments, including data collected while the system stays in closed loop.
- **Role:** method library
- **Maturity:** validated on lab hardware through recorded experimental data; does not deploy models or control hardware.
- **Maintainer / lab:** Merijn Floren / KU Leuven MECO
- **Licence / availability:** GPL-3.0; Python package on PyPI as `freq-statespace`

### Physical setup

<!-- Use-case demonstrators only. Method libraries and middleware: write `n/a`. -->

- **System under control:** n/a — offline identification library; no plant is controlled by this package.
- **Actuation:** n/a — it consumes recorded input signals and does not command actuators.
- **Sensing:** n/a — it consumes recorded input-output data and does not acquire measurements.
- **Sample rates:** user-supplied sampling frequency is required; input and output data contain an equal number of samples per period.
- **Hardware & fieldbus:** n/a — no hardware or fieldbus interface.
- **Scale of the dynamics:** n/a — determined by the sampling rate, excitation spectrum, record length, selected state dimension, and nonlinear model structure. JAX can use GPU/TPU acceleration for mid-size to large problems.

### Problems this repository can help with

- Fit a compact linear dynamic model from excited, sampled input-output tests, so it can be simulated, analysed, or used by another control-design workflow (frequency-domain BLA and state-space identification).
- Fit a nonlinear state-space model when a linear model misses nonlinear input-output behaviour, including static nonlinearities and behaviour that can be represented by increasing the state dimension (NL-LFR identification).
- Identify a plant or unknown controller from dedicated experiments while it remains in feedback, using a closed-loop BLA as the starting point (closed-loop BLA followed by state-space identification).

### Examples

<!--
One row per runnable example. Entry point = the file someone opens first.
Keep this table complete: an example missing here is invisible to the tool.
-->

| Example | What it demonstrates | Entry point | Sim | Hardware |
| --- | --- | --- | --- | --- |
| Silverbox with neural nonlinearity | Linear BLA identification followed by nonlinear NL-LFR fitting with a neural network. | `examples/nonlinear_benchmarks/silverbox_neural_network.py` | No | Recorded experimental data; offline only |
| Silverbox with inference and learning | BLA, polynomial NL-LFR inference and learning, nonlinear refinement, and test-set evaluation. | `examples/nonlinear_benchmarks/silverbox_inference_and_learning.py` | No | Recorded experimental data; offline only |
| F-16 ground-vibration test | Linear model fitting with and without stability enforcement. | `examples/nonlinear_benchmarks/f16.ipynb` | No | Recorded experimental data; offline only |
| Fine Steering Mirror | Linear and nonlinear modelling of measured mirror data, including distortion analysis. | `examples/nonlinear_benchmarks/fine_steering_mirror.ipynb` | No | Recorded experimental data; offline only |
| Parallel Wiener–Hammerstein system | Linear and nonlinear identification and benchmark reporting from measured data. | `examples/nonlinear_benchmarks/parallel_wiener_hammerstein.ipynb` | No | Recorded experimental data; offline only |
| Dual-motor drivetrain, closed loop | Multisine excitation-signal generation, controller and plant identification from closed-loop records, then validation by closed-loop simulation. | `examples/dual_motor_drivetrain/closed_loop_identification.ipynb` | No | Recorded experimental data; offline identification and simulation only |

### Preconditions

<!--
What must be true of the user's application before any of this transfers.
Be concrete: data you must be able to collect, signals you must be able to
inject, things you must be able to measure, compute you must have.
-->

- Supply sampled input and output recordings with a known sampling frequency. The signals must be persistently exciting and are expected to come from multisine excitation covering the dynamics of interest.
- The preferred workflow uses an integer number of steady-state periods; remove start-up transients before identification. Imperfect periodicity caused by disturbances, measurement noise, residual transients, or closed-loop effects does not by itself rule out use.
- To estimate a robust nonparametric BLA with uncertainty for a multi-input system, collect at least as many independent realizations as there are input channels. If this is unavailable, identification can still proceed from the input-output spectra rather than a BLA, but this is not preferred.
- For closed-loop controller identification, record a measured reference or other valid instrumental-variable signal to estimate the BLA with `best-linear-approximation`; use that BLA to initialize the linear fit. Nonlinear residual modelling subsequently uses the input-output recordings.
- Provide enough CPU memory and, where useful, a JAX-compatible GPU/TPU environment for optimisation; first-order optimizers are available for memory-efficient large-scale fitting.

### Not suitable when

<!--
The honesty valve. Concrete situations where a user should look elsewhere,
and, where you know one, the name of what they should look at instead.
-->

- The only available data are completely non-periodic, such as a measured step response. Use a time-domain identification method designed for transient or step-response data instead.
- The result must have states or parameters with immediate physical meaning. This package fits black-box, data-driven input-output models; its states and parameters are not directly physical quantities.
- The available experiment does not excite the dynamics to be identified, or start-up transients cannot be removed sufficiently to obtain an identification record. Run a dedicated excitation experiment instead.

### Dependencies beyond this repository

| Dependency | Why it is needed | Hard requirement? |
| --- | --- | --- |
| Python 3.12 or later | Runtime environment. | Yes |
| JAX | Automatic differentiation, JIT compilation, and CPU/GPU/TPU execution. | Yes |
| NumPy, SciPy, Equinox, and Optimistix | Array handling, numerical routines, model representation, and default optimisation. | Yes |
| `best-linear-approximation` | Nonparametric BLA estimation and its uncertainty information; also provides the closed-loop BLA handoff. | Yes |
| JAX GPU/TPU installation | Accelerates mid-size and large identification problems. | No — CPU JAX works |

### Where to look deeper

<!-- Files and directories worth reading, with one line each on what is in them. -->

- `README.md` — installation, the three-stage linear/nonlinear workflow, and the Silverbox quick example.
- `src/freq_statespace/_data_manager.py` — accepted signal layout, periodic-data metadata, BLA creation, and data validation.
- `src/freq_statespace/_best_linear_approximation.py` and `src/freq_statespace/_nonlin_lfr.py` — linear and nonlinear identification entry points.
- `examples/nonlinear_benchmarks/` — measured-system examples for linear and nonlinear identification.
- `examples/dual_motor_drivetrain/closed_loop_identification.ipynb` — closed-loop controller and plant workflow.

<!-- END: reusable-result capability description -->
