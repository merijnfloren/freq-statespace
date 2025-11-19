"""Quick example on the Silverbox benchmark dataset using inference and learning."""

import time

import nonlinear_benchmarks
import numpy as np
from nonlinear_benchmarks.error_metrics import RMSE

import freq_statespace as fss


# ========== LOAD AND PREPROCESS DATA ==========

# Load Silverbox benchmark dataset
train, test = nonlinear_benchmarks.Silverbox()
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test

# Sampling info
N = 8192             # Samples per period
R = 6                # Number of realizations
P = 1                # Periods per realization
nu, ny = 1, 1        # SISO system
fs = 1e7 / 2**14     # Sampling frequency [Hz]
f_idx = np.arange(1, 2 * 1342, 2)  # Excited odd harmonics

# Extract input and output signals
u, y = train.u, train.y

# Process multisine data (discard transients, reshape)
N_init = 164    # Initial samples to discard (first realization)
N_z = 100       # Zero samples between blocks
N_tr = 400      # Transient samples to discard

u_train = np.zeros((N, R))
y_train = np.zeros((N, R))

for k in range(R):
    if k == 0:
        u = u[N_init:]
        y = y[N_init:]
    else:
        u = u[N_z + N_tr:]
        y = y[N_z + N_tr:]

    u_train[:, k] = u[:N]
    y_train[:, k] = y[:N]

    u = u[N:]
    y = y[N:]

# Reshape to match expected dimensions: (N, nu, R, P)
u_train = u_train.reshape(N, nu, R, P)
y_train = y_train.reshape(N, ny, R, P)

data = fss.create_data_object(u_train, y_train, f_idx, fs)


# ========== SYSTEM IDENTIFICATION ==========

start_time = time.time()

# Step 1: BLA estimation
nx = 2  # state dimension
q = nx + 1  # subspace dimensioning parameter
bla = fss.lin.subspace_id(data, nx, q)  # NRMSE 18.36%, non-iterative
bla = fss.lin.optimize(bla, data)  # NRMSE 13.17%, 6 iters, 1.97ms/iter

# Step 2: Inference and learning
phi = fss.f_static.basis.Polynomial(nz=1, degree=3)
nllfr = fss.nonlin.inference_and_learning(
    bla, data, phi=phi, nw=1, lambda_w=1e-2, fixed_point_iters=3
)  # NRMSE 1.11%, 42 iters, 13.2ms/iter

# Step 3: Nonlinear optimization
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.44%, 100 iters, 387ms/iter

total_time = time.time() - start_time
print(f"\nTotal time for training: {total_time:.2f} seconds")


# ========== TEST PERFORMANCE ==========

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
