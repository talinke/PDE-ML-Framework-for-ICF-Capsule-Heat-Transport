import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax  #Adam optimizer

ICFdata = np.array([
    [0.001000,  1.000000,  420.301513, 420.301513],
    [0.001000,  10.000000,  603.015126, 603.015126],
    [0.001000,  100.000000,  2430.151261, 2430.151261],
    [0.001000,  1000.000000,  20701.512613, 20701.512613],
    [0.001000,  10000.000000,  203415.126134, 203415.126127],
    [0.001000,  100000.000000,  2030551.261333, 2030551.261277],
    [0.000100,  1.000000,  421.122660, 419.480313],
    [0.000100,  10.000000,  611.226603, 594.803128],
    [0.000100,  100.000000,  2512.266029, 2348.031282],
    [0.000100,  1000.000000,  21522.660287, 19880.312820],
    [0.000100,  10000.000000,  211626.602868, 195203.128196],
    [0.000100,  100000.000000,  2112666.028686, 1948431.281960],
    [0.000010,  1.000000,  456.863264, 400.314857],
    [0.000010,  10.000000,  968.632638, 403.148567],
    [0.000010,  100.000000,  6086.326379, 431.485674],
    [0.000010,  1000.000000,  57263.263791, 714.856737],
    [0.000010,  10000.000000,  569032.637911, 3548.567369],
    [0.000010,  100000.000000,  5686726.379061, 31885.673692],
    [0.000001,  1.000000,  579.560684, 400.000000],
    [0.000001,  10.000000,  2195.606841, 400.000000],
    [0.000001,  100.000000,  18356.068406, 400.000000],
    [0.000001,  1000.000000,  179960.684060, 400.000000],
    [0.000001,  10000.000000,  1796006.840535, 400.000000],
    [0.000001,  100000.000000,  17956468.405825, 400.000000],
    [0.0000001,  1.000000,  959.486063, 400.000000],
    [0.0000001,  10.000000,  5994.860629, 400.000000],
    [0.0000001,  100.000000,  56348.606294, 400.000000],
    [0.0000001,  1000.000000,  559886.062949, 400.000000],
    [0.0000001,  10000.000000,  5595260.629854, 400.000000],
    [0.0000001,  100000.000000,  55949006.296277, 400.000000],
    [0.00000001,  1.000000,  1846.525600, 400.000000],
    [0.00000001,  10.000000,  14865.256001, 400.000000],
    [0.00000001,  100.000000,  145052.560018, 400.000000],
    [0.00000001,  1000.000000,  1446925.599960, 400.000000],
    [0.00000001,  10000.000000,  14465656.002127, 400.000000],
    [0.00000001,  100000.000000,  144652960.010369, 400.000000]
])

# Inputs and outputs
x = ICFdata[:, [0,1]]
y = ICFdata[:, 2].reshape(-1,1)  # T[0]
x = np.log10(x)
y = np.log10(y)

# Train/Validation split
N = x.shape[0]
split = int(0.8 * N)
train_x, val_x = x[:split], x[split:]
train_y, val_y = y[:split], y[split:]

# Standardize inputs (mean=0, std=1)
mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x - mean) / std
val_x = (val_x - mean) / std

# Neural network
def sigmoid_layer(w, x):
    return 1 / (1 + jnp.exp(-jnp.dot(x, w)))

def relu_layer(w, x):
    return jnp.maximum(0, jnp.dot(x, w))


def linear_layer(w, x):
    return jnp.dot(x, w)

def nn_model(params, x):
    w1, w2 = params
    h = sigmoid_layer(w1, x)
    #h = relu_layer(w1,x)
    y_pred = linear_layer(w2, h)
    return y_pred

def mse_loss(params, x, y, l2=0.0):
    y_pred = nn_model(params, x)
    loss = jnp.mean((y_pred - y)**2)
    # L2 regularization
    loss += l2 * sum([jnp.sum(w**2) for w in params])
    return loss

# initialize weights
key = jax.random.PRNGKey(0)
num_features = train_x.shape[1]
num_hidden = 5
num_outputs = train_y.shape[1]

w1 = jax.random.normal(key, (num_features, num_hidden)) * 0.1
key, subkey = jax.random.split(key)
w2 = jax.random.normal(subkey, (num_hidden, num_outputs)) * 0.1
params = [w1, w2]

# Adam optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

grad_fn = jax.grad(mse_loss)

num_epochs = 5000
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    grads = grad_fn(params, train_x, train_y, 1e-4) 
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    if epoch % 50 == 0:
        train_loss = mse_loss(params, train_x, train_y, 1e-4)
        val_loss = mse_loss(params, val_x, val_y, 1e-4)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

# Plots
plt.plot(np.arange(0,len(train_losses))*50, train_losses, label='Train')
plt.plot(np.arange(0,len(val_losses))*50, val_losses, label='Validation')
plt.xlabel("Epoch")
plt.ylabel("MSE (log space)")
#plt.ylim([0, 0.2])
plt.legend()
plt.show()

print("Final training loss:", train_losses[-1])
print("Final validation loss:", val_losses[-1])



# Test data (alpha, flux)
test_data = np.array([
    [5.0e-6, 5.0e3],        #region of high certainty
    [5.0e-6, 5.0e2],        # |
    [5.0e-6, 5.0e1],        # |
    [5.0e-7, 5.0e2],        # |
    [5.0e-5, 5.0e2],        # |
    [5.0e-8, 5.0e0],        # region of high uncertainty
    [5.0e-8, 5.0e4],        # |
    [5.0e-4, 5.0e0],        # |
    [5.0e-4, 5.0e4]         # |
])
test_x = np.log10(test_data)
test_x = (test_x - mean) / std

# Make NN predictions
y_pred_log = nn_model(params, test_x)
y_pred = 10 ** y_pred_log # convert back to original T scale

# PDE solutions
PDEsolution_shell = np.array([402420.329983, 40602.032998, 4420.203300, 127165.908330, 13333.423275, 4286.468977, 38865089.767167, 501.507568, 1015475.678072])
PDEsolution_fuel = [410.148219, 401.014822, 400.101482, 400.000000, 7789.746749, 400.000000, 400.000000, 501.507558, 1015475.583229]

rel_error = (PDEsolution_shell - np.array(y_pred).flatten())/PDEsolution_shell
print("Relative error:", rel_error)



