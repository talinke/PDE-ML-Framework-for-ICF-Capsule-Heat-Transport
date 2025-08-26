import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax  # Adam optimizer

# load data
ICFdata = np.loadtxt("ICFdata.txt", comments='#', delimiter=None, usecols=(0,1,2,3))

# inputs and outputs
x = ICFdata[:, [0,1]]
y = ICFdata[:, 2].reshape(-1,1)  # train on outer ablator temperature T[0]
x = np.log10(x)
y = np.log10(y)

# train/validation split
N = x.shape[0]
split = int(0.8 * N)
train_x, val_x = x[:split], x[split:]
train_y, val_y = y[:split], y[split:]

# standardize inputs (mean=0, std=1)
mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x - mean) / std
val_x = (val_x - mean) / std

# neural network:
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

# randomly initialize weights
key = jax.random.PRNGKey(0)
num_features = train_x.shape[1]
num_hidden = 5
num_outputs = train_y.shape[1]

w1 = jax.random.normal(key, (num_features, num_hidden)) * 0.1
key, subkey = jax.random.split(key)
w2 = jax.random.normal(subkey, (num_hidden, num_outputs)) * 0.1
params = [w1, w2]

# use Adam optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

grad_fn = jax.grad(mse_loss)

num_epochs = 2000
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

# plot
plt.plot(np.arange(0,len(train_losses))*50, train_losses, label='Train')
plt.plot(np.arange(0,len(val_losses))*50, val_losses, label='Validation')
plt.xlabel("Epoch")
plt.ylabel("MSE (log space)")
#plt.ylim([0, 0.2])
plt.legend()
plt.show()

# print final loss
print("Final training loss:", train_losses[-1])
print("Final validation loss:", val_losses[-1])
