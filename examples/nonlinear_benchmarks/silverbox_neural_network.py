"""Quick example on the Silverbox benchmark dataset using a neural network."""

import jax

import freq_statespace as fss


data = fss.load_and_preprocess_silverbox_data()  # 8192 x 6 samples

# Step 1: BLA estimation
nx = 2  # state dimension
bla = fss.lin.subspace_id(data, nx)
bla = fss.lin.optimize(bla, data)

# Step 2: Nonlinear optimization
neural_net = fss.static.NeuralNetwork(
    nw=1, nz=1, layers=1, neurons_per_layer=10, activation=jax.nn.relu
)
nllfr = fss.nonlin.connect(bla, neural_net)
nllfr = fss.nonlin.optimize(nllfr, data, device="cpu")

# NOTE: CPU is faster here because the optimization problem is recurrent in nature
# and has very little work per step, so GPU overhead dominates. Larger models perform 
# enough computation per step for GPUs to become advantageous.