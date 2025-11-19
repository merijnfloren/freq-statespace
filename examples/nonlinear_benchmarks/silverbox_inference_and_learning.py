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
bla = fss.lin.subspace_id(data, nx)  # NRMSE 18.36%, non-iterative
bla = fss.lin.optimize(bla, data)  # NRMSE 13.17%, 6 iters, 1.97ms/iter

# Step 2: Inference and learning
phi = fss.static.basis.Polynomial(nz=1, degree=3)
nllfr = fss.nonlin.inference_and_learning(
    bla, data, phi=phi, nw=1)  # NRMSE 1.11%, 42 iters, 13.2ms/iter

# Step 3: Nonlinear optimization
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.44%, 100 iters, 387ms/iter

total_time = time.time() - start_time
print(f"\nTotal time for training: {total_time:.2f} seconds")


# ========== TEST PERFORMANCE ==========

test = nonlinear_benchmarks.Silverbox()[1]
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test

# -- Multisine test --
y_model = nllfr.simulate(test_multisine.u)[0]
n = test_multisine.state_initialization_window_length
rmse_ms = RMSE(test_multisine.y[n:], y_model[n:])
print(f"\nRMSE on multisine test: {1000 * rmse_ms:.3e} mV")

# -- Arrowhead full --
y_model = nllfr.simulate(test_arrow_full.u)[0]
n = test_arrow_full.state_initialization_window_length
rmse_af = RMSE(test_arrow_full.y[n:], y_model[n:])
print(f"RMSE on arrowhead full test: {1000 * rmse_af:.3e} mV")

# -- Arrowhead no extrapolation --
y_model = nllfr.simulate(test_arrow_no_extrapolation.u)[0]
n = test_arrow_no_extrapolation.state_initialization_window_length
rmse_an = RMSE(test_arrow_no_extrapolation.y[n:], y_model[n:])
print(f"RMSE on arrowhead no extrapolation: {1000 * rmse_an:.3e} mV")
