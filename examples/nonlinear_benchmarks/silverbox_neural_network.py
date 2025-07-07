"""Quick example on the Silverbox benchmark dataset using a neural network."""

import jax

import freq_statespace as fss


data = fss.load_and_preprocess_silverbox_data()  # 8192 x 6 samples

# Step 1: BLA estimation
nx = 2  # state dimension
q = nx + 1  # subspace dimensioning parameter
bla = fss.lin.subspace_id(data, nx, q)  # NRMSE 18.36%, non-iterative
bla = fss.lin.optimize(bla, data)  # NRMSE 13.17%, 6 iters, 1.67ms/iter

# Step 2: Nonlinear optimization
neural_net = fss.f_static.NeuralNetwork(
    nw=1, nz=1, num_layers=1, num_neurons_per_layer=10, activation=jax.nn.relu
)
nllfr = fss.nonlin.construct(bla, neural_net)
nllfr = fss.nonlin.optimize(nllfr, data)  # NRMSE 0.54%, 100 iters, 356ms/iter
