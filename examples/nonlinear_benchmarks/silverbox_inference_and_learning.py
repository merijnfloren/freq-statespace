import freq_statespace as fss


data = fss.load_and_preprocess_silverbox_data()  # 8192 x 6 samples

# Step 1: BLA estimation
nx = 2  # state dimension
q = nx + 1  # subspace dimensioning parameter
model = fss.bla.subspace_id(data, nx, q)  # NRMSE 18.36%, non-iterative
model = fss.bla.optimize(model, data)  # NRMSE 13.17%, 5 iters, 1.43ms/iter

# Step 2: Inference and learning
phi = fss.feature_map.Polynomial(nz=1, degree=3)
model = fss.nonlin_lfr.inference_and_learning(
    model, data, phi=phi, nw=1, lambda_w=1e-2, fixed_point_iters=5
)  # NRMSE 1.11%, 47 iters, 12.9ms/iter

# Step 3: Nonlinear optimization
model = fss.nonlin_lfr.optimize(model, data)  # NRMSE 0.44%, 100 iters, 436ms/iter
