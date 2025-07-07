"""Quick example on the Silverbox benchmark dataset using inference and learning."""

import freq_statespace as fss


data = fss.load_and_preprocess_silverbox_data()  # 8192 x 6 samples

# Step 1: BLA estimation
nx = 2  # state dimension
q = nx + 1  # subspace dimensioning parameter
bla = fss.lin.subspace_id(data, nx, q)  # NRMSE 18.36%, non-iterative
bla = fss.lin.optimize(bla, data)  # NRMSE 13.17%, 6 iters, 1.97ms/iter

# Step 2: Inference and learning
phi = fss.f_static.basis.Polynomial(nz=1, degree=3)
nllfr = fss.nonlin.inference_and_learning(
    bla, data, phi=phi, nw=1, lambda_w=1e-2, fixed_point_iters=5
)  # NRMSE 1.11%, 42 iters, 13.2ms/iter

# Step 3: Nonlinear optimization
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.44%, 100 iters, 387ms/iter
