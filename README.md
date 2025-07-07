# freq-statespace
A flexible [JAX](https://docs.jax.dev/en/latest/index.html)-based package for nonlinear state-space identification using frequency-domain optimization techniques. 

It’s built around the **nonlinear Linear Fractional Representation (NL-LFR)** model structure, a powerful block-oriented framework that connects an LTI system with a static feedback nonlinearity. This internal feedback setup is key to capturing complex behaviors found in many real-world systems.
<div align="center">
  <img src="https://github.com/merijnfloren/freq-statespace/raw/main/model_structure.svg" width="500px" />
</div>

### Basic usage

The package works with input–output data sequences $u(n)$ and $y(n)$ for $n = 0, \ldots, N-1$, assuming the system is excited by a periodic input and that an integer number of steady-state output periods has been recorded. The specific NL-LFR structure is defined as:
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

1. **Best Linear Approximation (BLA) estimation**.
   Initializes the matrices $A$, $B_u$, $C_y$ and $D_{yu}$ using the [frequency-domain subspace method](https://github.com/tomasmckelvey/fsid), and refines these estimates through iterative optimization. If you're only interested in linear state-space models, you can stop the identification process here.

2. **NL-LFR initialization**.
  Applies the [frequency-domain inference and learning method](https://arxiv.org/abs/2503.14409) to efficiently initialize the remaining model parameters while keeping the BLA parameters fixed. This step requires $f(\cdot)$ to be a linear-in-the-parameters model, for example one based on polynomial basis functions.

3. **NL-LFR optimization**. Performs iterative refinement of all model parameters using time-domain simulations. This is the most computationally demanding step, mainly due to the sequential nature of the forward simulations. Fortunately, the previous steps should have provided an initialization that is already close to a good local minimum.

It is also possible to skip the inference and learning step and go straight to nonlinear optimization. An advantage of this approach is that it puts no restriction on the structure of $f(\cdot)$, i.e., it does not require a model that is linear in the parameters.

### Features
- Provides two workflows for identifying nonlinear LFR state-space models by primarily exploiting a frequency-domain formulation that enables inherent parallelism.
- Leverages JAX for automatic differentiation, JIT compilation, and GPU/TPU acceleration.
- Supports [Optimistix](https://docs.kidger.site/optimistix/) solvers (Levenberg–Marquardt, BFGS, ...) for structured system identification problems.
- Supports [Optax](https://optax.readthedocs.io/en/latest/) optimizers (Adam, SGD, ...) for large-scale or stochastic optimization.

## Installation

```bash
pip install freq-statespace
```

If JAX isn't already installed in your environment, this will install the CPU-only version. For GPU/TPU support (strongly recommended, often many times faster for mid-size to large problems), follow the [JAX installation guide](https://github.com/google/jax#installation).

## Quick example

We show an exemplary training pipeline on the [Silverbox benchmark dataset](https://www.nonlinearbenchmark.org/benchmarks/silverbox), containing input-output measurements from an electronic circuit that mimics a mass-spring-damper system with a cubic spring nonlinearity.

We first estimate the BLA:

```python
import freq_statespace as fss

data = fss.load_and_preprocess_silverbox_data()  # 8192 x 6 samples

# Step 1: BLA estimation
nx = 2  # state dimension
q = nx + 1  # subspace dimensioning parameter
bla = fss.lin.subspace_id(data, nx, q)  # NRMSE 18.36%, non-iterative
bla = fss.lin.optimize(bla, data)  # NRMSE 13.17%, 6 iters, 1.97ms/iter
```
Next, we proceed with inference and learning, followed by full nonlinear optimization:

```python
# Step 2: Inference and learning
phi = fss.f_static.basis.Polynomial(nz=1, degree=3)
nllfr = fss.nonlin.inference_and_learning(
    bla, data, phi=phi, nw=1, lambda_w=1e-2, fixed_point_iters=5
)  # NRMSE 1.11%, 42 iters, 13.2ms/iter

# Step 3: Nonlinear optimization
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.44%, 100 iters, 387ms/iter
```

Alternatively, we could skip inference and learning and jump straight to nonlinear optimization. In this example we use a neural network:
```python
import jax

# Step 2: Nonlinear optimization
neural_net = fss.f_static.NeuralNetwork(
    nw=1, nz=1, num_layers=1, num_neurons_per_layer=10, activation=jax.nn.relu
)
nllfr = fss.nonlin.construct(bla, neural_net)
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.54%, 100 iters, 356ms/iter
```
> **Note:** Iteration timings were measured on an NVIDIA T600 Laptop GPU.

## Citation
If you use this code in your work, please cite it as ([arXiv link](https://arxiv.org/abs/2503.14409)):
```bibtex
@article{floren2025inference,
  title={Inference and Learning of Nonlinear LFR State-Space Models},
  author={Floren, Merijn and No{\"e}l, Jean-Philippe and Swevers, Jan},
  journal={IEEE Control Systems Letters},
  year={2025},
  publisher={IEEE}
}
```
