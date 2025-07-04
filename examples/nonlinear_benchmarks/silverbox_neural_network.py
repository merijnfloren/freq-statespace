import jax

import freq_statespace as fss


data = fss.load_and_preprocess_silverbox_data()  # 8192 x 6 samples

# Step 1: BLA estimation
nx = 2  # state dimension
q = nx + 1  # subspace dimensioning parameter
model = fss.bla.subspace_id(data, nx, q)  # NRMSE 18.36%, non-iterative
model = fss.bla.optimize(model, data)  # NRMSE 13.17%, 5 iters, 1.54ms/iter

# Step 2: Nonlinear optimization
f_static = fss.nonlin_func.NeuralNetwork(
    nw=1, nz=1, num_layers=1, num_neurons_per_layer=10, activation=jax.nn.relu
)
model = fss.nonlin_lfr.construct(bla=model, f_static=f_static)
model = fss.nonlin_lfr.optimize(model, data)  # NRMSE 0.68%, 100 iters, 401ms/iter
