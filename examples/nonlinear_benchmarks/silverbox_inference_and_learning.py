"""Quick example on the Silverbox benchmark dataset using inference and learning."""

import time

import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE

import freq_statespace as fss


# ========== SYSTEM IDENTIFICATION ==========
data = fss.load_and_preprocess_silverbox_data()

start_time = time.time()

# Step 1: BLA estimation
nx = 2  # state dimension
bla = fss.lin.subspace_id(data, nx)
bla = fss.lin.optimize(bla, data)

# Step 2: Inference and learning
phi = fss.static.basis.Polynomial(nz=1, degree=3)

nllfr = fss.nonlin.inference_and_learning(
    bla, data, phi=phi, nw=1)

# Step 3: Nonlinear optimization
nllfr = fss.nonlin.optimize(nllfr, data, device="cpu")

# NOTE: My CPU is faster here because the optimization problem is recurrent in nature
# and has very little work per step, so GPU overhead dominates. Larger models perform 
# enough computation per step for GPUs to become advantageous.

total_time = time.time() - start_time
print(f"\nTotal time for training: {total_time:.2f} seconds")


# ========== TEST PERFORMANCE ==========

test = nonlinear_benchmarks.Silverbox()[1]
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test

# -- Multisine test --
y_model = nllfr.simulate(test_multisine.u)[0]
n = test_multisine.state_initialization_window_length
rmse_ms = RMSE(test_multisine.y[n:], y_model[n:].flatten())
print(f"\nRMSE on multisine test: {1000 * rmse_ms:.3e} mV")

# -- Arrowhead full --
y_model = nllfr.simulate(test_arrow_full.u)[0]
n = test_arrow_full.state_initialization_window_length
rmse_af = RMSE(test_arrow_full.y[n:], y_model[n:].flatten())
print(f"RMSE on arrowhead full test: {1000 * rmse_af:.3e} mV")

# -- Arrowhead no extrapolation --
y_model = nllfr.simulate(test_arrow_no_extrapolation.u)[0]
n = test_arrow_no_extrapolation.state_initialization_window_length
rmse_an = RMSE(test_arrow_no_extrapolation.y[n:], y_model[n:].flatten())
print(f"RMSE on arrowhead no extrapolation: {1000 * rmse_an:.3e} mV")
