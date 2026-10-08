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

> **Note:** Inference and learning requires substantially less computation time per iteration, as these operations are largely parallelizable. Iteration timings were measured on an NVIDIA T600 Laptop GPU.

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
