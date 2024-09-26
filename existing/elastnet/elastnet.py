import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import grad, jit, random
import optax # optimization library for JAX

import numpy as np
import time

from sklearn import preprocessing

# Setup

num_neuron = 128 # Number of neurons in hidden layers (how many features the model should learn)
learn_rate = 0.001  # Learning rate (how fast the model should learn)

x_disp = np.loadtxt('C:/Users/ryanm/OneDrive/Desktop/research/existing/elastnet/data_incompressible/m_z5_nu_05/disp_coord')  # Displacement coordinates
y_disp = np.loadtxt('C:/Users/ryanm/OneDrive/Desktop/research/existing/elastnet/data_incompressible/m_z5_nu_05/disp_data') 
x_elas = np.loadtxt('C:/Users/ryanm/OneDrive/Desktop/research/existing/elastnet/data_incompressible/m_z5_nu_05/strain_coord') # Elasticity coordinates
y_elas = np.loadtxt('C:/Users/ryanm/OneDrive/Desktop/research/existing/elastnet/data_incompressible/m_z5_nu_05/m_data')

ss_x = preprocessing.StandardScaler() # Standardizes the features by removing the mean and scaling to unit variance
x_disp = ss_x.fit_transform(x_disp.reshape(-1, 2)) # Reshape the data to 2D array
x_elas = ss_x.fit_transform(x_elas.reshape(-1, 2))

key = random.PRNGKey(0)
def weight_variable(shape, key):
    return random.truncated_normal(key, lower=-0.1, upper=0.1, shape=shape) * 0.1
def bias_variable(shape):
    return jnp.full(shape, 0.1)

logs_path = 'TensorBoard/'

# Define elasticity network
# The layers of the neural network:
# The network is a feedforward neural network with 17 hidden layers and 128 neurons in each layer

xs_disp = jnp.array(x_disp)  #xs_disp = tf.compat.v1.placeholder(tf.float32, [None, 2])
ys_disp = jnp.array(y_disp)  #ys_disp = tf.compat.v1.placeholder(tf.float32, [None, 2]) # u & v
xs_elas = jnp.array(x_elas)  #xs_elas = tf.compat.v1.placeholder(tf.float32, [None, 2])

W_fc1a = weight_variable([2, num_neuron])
b_fc1a = bias_variable([num_neuron])
h_fc1a = jax.nn.relu(jnp.dot(xs_elas, W_fc1a) + b_fc1a)

W_fc2a = weight_variable([num_neuron, num_neuron])
b_fc2a = bias_variable([num_neuron])
h_fc2a = jax.nn.relu(jnp.dot(h_fc1a, W_fc2a) + b_fc2a)

W_fc3a = weight_variable([num_neuron, num_neuron])
b_fc3a = bias_variable([num_neuron])
h_fc3a = jax.nn.relu(jnp.dot(h_fc2a, W_fc3a) + b_fc3a)

W_fc4a = weight_variable([num_neuron, num_neuron])
b_fc4a = bias_variable([num_neuron])
h_fc4a = jax.nn.relu(jnp.dot(h_fc3a, W_fc4a) + b_fc4a)

W_fc5a = weight_variable([num_neuron, num_neuron])
b_fc5a = bias_variable([num_neuron])
h_fc5a = jax.nn.relu(jnp.dot(h_fc4a, W_fc5a) + b_fc5a)

W_fc6a = weight_variable([num_neuron, num_neuron])
b_fc6a = bias_variable([num_neuron])
h_fc6a = jax.nn.relu(jnp.dot(h_fc5a, W_fc6a) + b_fc6a)

W_fc7a = weight_variable([num_neuron, num_neuron])
b_fc7a = bias_variable([num_neuron])
h_fc7a = jax.nn.relu(jnp.dot(h_fc6a, W_fc7a) + b_fc7a)

W_fc8a = weight_variable([num_neuron, num_neuron])
b_fc8a = bias_variable([num_neuron])
h_fc8a = jax.nn.relu(jnp.dot(h_fc7a, W_fc8a) + b_fc8a)

W_fc9a = weight_variable([num_neuron, num_neuron])
b_fc9a = bias_variable([num_neuron])
h_fc9a = jax.nn.relu(jnp.dot(h_fc8a, W_fc9a) + b_fc9a)

W_fc10a = weight_variable([num_neuron, num_neuron])
b_fc10a = bias_variable([num_neuron])
h_fc10a = jax.nn.relu(jnp.dot(h_fc9a, W_fc10a) + b_fc10a)

W_fc11a = weight_variable([num_neuron, num_neuron])
b_fc11a = bias_variable([num_neuron])
h_fc11a = jax.nn.relu(jnp.dot(h_fc10a, W_fc11a) + b_fc11a)

W_fc12a = weight_variable([num_neuron, num_neuron])
b_fc12a = bias_variable([num_neuron])
h_fc12a = jax.nn.relu(jnp.dot(h_fc11a, W_fc12a) + b_fc12a)

W_fc13a = weight_variable([num_neuron, num_neuron])
b_fc13a = bias_variable([num_neuron])
h_fc13a = jax.nn.relu(jnp.dot(h_fc12a, W_fc13a) + b_fc13a)

W_fc14a = weight_variable([num_neuron, num_neuron])
b_fc14a = bias_variable([num_neuron])
h_fc14a = jax.nn.relu(jnp.dot(h_fc13a, W_fc14a) + b_fc14a)

W_fc15a = weight_variable([num_neuron, num_neuron])
b_fc15a = bias_variable([num_neuron])
h_fc15a = jax.nn.relu(jnp.dot(h_fc14a, W_fc15a) + b_fc15a)

W_fc16a = weight_variable([num_neuron, num_neuron])
b_fc16a = bias_variable([num_neuron])
h_fc16a = jax.nn.relu(jnp.dot(h_fc15a, W_fc16a) + b_fc16a)

W_fc17a = weight_variable([num_neuron, 1])
b_fc17a = bias_variable([1])
y_preda = jnp.dot(h_fc16a, W_fc17a) + b_fc17a
y_pred_m = y_preda[:, 0]

# Define displacement network

W_fc1b = weight_variable([2, num_neuron])
b_fc1b = bias_variable([num_neuron])
h_fc1b = jax.nn.swish(jnp.dot(xs_disp, W_fc1b) + b_fc1b)

W_fc2b = weight_variable([num_neuron, num_neuron])
b_fc2b = bias_variable([num_neuron])
h_fc2b = jax.nn.swish(jnp.dot(h_fc1b, W_fc2b) + b_fc2b)

W_fc3b = weight_variable([num_neuron, num_neuron])
b_fc3b = bias_variable([num_neuron])
h_fc3b = jax.nn.swish(jnp.dot(h_fc2b, W_fc3b) + b_fc3b)

W_fc4b = weight_variable([num_neuron, num_neuron])
b_fc4b = bias_variable([num_neuron])
h_fc4b = jax.nn.swish(jnp.dot(h_fc3b, W_fc4b) + b_fc4b)

W_fc5b = weight_variable([num_neuron, num_neuron])
b_fc5b = bias_variable([num_neuron])
h_fc5b = jax.nn.swish(jnp.dot(h_fc4b, W_fc5b) + b_fc5b)

W_fc6b = weight_variable([num_neuron, num_neuron])
b_fc6b = bias_variable([num_neuron])
h_fc6b = jax.nn.swish(jnp.dot(h_fc5b, W_fc6b) + b_fc6b)

W_fc7b = weight_variable([num_neuron, num_neuron])
b_fc7b = bias_variable([num_neuron])
h_fc7b = jax.nn.swish(jnp.dot(h_fc6b, W_fc7b) + b_fc7b)

W_fc8b = weight_variable([num_neuron, num_neuron])
b_fc8b = bias_variable([num_neuron])
h_fc8b = jax.nn.swish(jnp.dot(h_fc7b, W_fc8b) + b_fc8b)

W_fc9b = weight_variable([num_neuron, num_neuron])
b_fc9b = bias_variable([num_neuron])
h_fc9b = jax.nn.swish(jnp.dot(h_fc8b, W_fc9b) + b_fc9b)

W_fc10b = weight_variable([num_neuron, num_neuron])
b_fc10b = bias_variable([num_neuron])
h_fc10b = jax.nn.swish(jnp.dot(h_fc9b, W_fc10b) + b_fc10b)

W_fc11b = weight_variable([num_neuron, num_neuron])
b_fc11b = bias_variable([num_neuron])
h_fc11b = jax.nn.swish(jnp.dot(h_fc10b, W_fc11b) + b_fc11b)

W_fc12b = weight_variable([num_neuron, num_neuron])
b_fc12b = bias_variable([num_neuron])
h_fc12b = jax.nn.swish(jnp.dot(h_fc11b, W_fc12b) + b_fc12b)

W_fc13b = weight_variable([num_neuron, num_neuron])
b_fc13b = bias_variable([num_neuron])
h_fc13b = jax.nn.swish(jnp.dot(h_fc12b, W_fc13b) + b_fc13b)

W_fc14b = weight_variable([num_neuron, num_neuron])
b_fc14b = bias_variable([num_neuron])
h_fc14b = jax.nn.swish(jnp.dot(h_fc13b, W_fc14b) + b_fc14b)

W_fc15b = weight_variable([num_neuron, num_neuron])
b_fc15b = bias_variable([num_neuron])
h_fc15b = jax.nn.swish(jnp.dot(h_fc14b, W_fc15b) + b_fc15b)

W_fc16b = weight_variable([num_neuron, num_neuron])
b_fc16b = bias_variable([num_neuron])
h_fc16b = jax.nn.swish(jnp.dot(h_fc15b, W_fc16b) + b_fc16b)

W_fc17b = weight_variable([num_neuron, 1])
b_fc17b = bias_variable([1])
y_predb = jnp.dot(h_fc16b, W_fc17b) + b_fc17b
y_pred_v = y_predb[:, 0]

# Read displacements

y_u = ys_disp[:, 0] # the actual displacement in the x-direction
y_v = y_pred_v # the predicted displacement in the x-direction

# Calculate strains
# The strains are calculated by taking the derivative of the displacement field
# The derivative is calculated using a convolutional neural network
# The derivative of the displacement field is the strain field

# Convolution: Convolves the displacement matrices to calculate strains.
# y_e_xx: The strain in the x-direction
# y_e_yy: The strain in the y-direction
# y_r_xy: The strain in the xy-direction (shear strain)

def conv2d(x, W):
    return lax.conv_general_dilated(x, W, window_strides=(1, 1), padding='VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

u_matrix = jnp.reshape(y_u, [257, 257]) # u is the displacement components
v_matrix = jnp.reshape(y_v, [257, 257])
u_matrix_4d = jnp.reshape(u_matrix, [-1, 257, 257, 1])
v_matrix_4d = jnp.reshape(v_matrix, [-1, 257, 257, 1])
conv_x = np.array([[[[-0.5]], [[-0.5]]], [[[0.5]], [[0.5]]], ])
conv_y = np.array([[[[0.5]], [[-0.5]]], [[[0.5]], [[-0.5]]], ])

conv_x = jnp.array(conv_x, dtype = jnp.float32)
conv_y = jnp.array(conv_y, dtype = jnp.float32)
y_e_xx = conv2d(u_matrix_4d, conv_x)
y_e_yy = conv2d(v_matrix_4d, conv_y)
y_r_xy = conv2d(u_matrix_4d, conv_y) + conv2d(v_matrix_4d, conv_x)
y_e_xx = 100*jnp.reshape(y_e_xx, [-1])
y_e_yy = 100*jnp.reshape(y_e_yy, [-1])
y_r_xy = 100*jnp.reshape(y_r_xy, [-1])

# Define elastic tensors
# The elastic tensor is a 3x3 matrix that describes the relationship between stress and strain in a material
# Figure (2) in the inverse problem paper

c_matrix = (1/(1-(1/2.0)**2))*np.array([[1, 1/2.0, 0], [1/2.0, 1, 0], [0, 0, (1-1/2.0)/2.0]])
c_matrix = jnp.array(c_matrix, dtype = jnp.float32)

# Calculate stresses
# The stresses are calculated by multiplying the elastic tensor by the strain field
# The stress field is the output of the neural network

strain    = jnp.stack([y_e_xx, y_e_yy, y_r_xy], axis = 1)
modulus   = jnp.stack([y_pred_m, y_pred_m, y_pred_m], axis = 1)
stress    = jnp.multiply(jnp.dot(strain, c_matrix), modulus)
stress_xx = stress[:, 0]
stress_yy = stress[:, 1]
stress_xy = stress[:, 2]

# Calculate sum of stresses
# The sum of the stresses is calculated by taking the sum of the sub-stresses

stress_xx_matrix = jnp.reshape(stress_xx, [256, 256])
stress_yy_matrix = jnp.reshape(stress_yy, [256, 256])
stress_xy_matrix = jnp.reshape(stress_xy, [256, 256])

# Calculate sum of sub stresses

sum_conv = np.array([[[[1.0]], [[1.0]], [[1.0]]], [[[1.0]], [[1.0]], [[1.0]]],
[[[1.0]], [[1.0]], [[1.0]]], ])
y_pred_m_matrix = jnp.reshape(y_pred_m, [256, 256])
y_pred_m_matrix_4d = jnp.reshape(y_pred_m_matrix, [-1, 256, 256, 1])
y_pred_m_conv = conv2d(y_pred_m_matrix_4d, sum_conv)
stress_xx_matrix_4d = jnp.reshape(stress_xx_matrix, [-1, 256, 256, 1])
stress_yy_matrix_4d = jnp.reshape(stress_yy_matrix, [-1, 256, 256, 1])
stress_xy_matrix_4d = jnp.reshape(stress_xy_matrix, [-1, 256, 256, 1])
wx_conv_xx = np.array([[[[-1.0]], [[-1.0]], [[-1.0]]], [[[0.0]], [[0.0]], [[0.0]]],
[[[1.0]], [[1.0]], [[1.0]]], ])
wx_conv_xy = np.array([[[[1.0]], [[0.0]], [[-1.0]]], [[[1.0]], [[0.0]], [[-1.0]]],
[[[1.0]], [[0.0]], [[-1.0]]], ])
wy_conv_yy = np.array([[[[1.0]], [[0.0]], [[-1.0]]], [[[1.0]], [[0.0]], [[-1.0]]],
[[[1.0]], [[0.0]], [[-1.0]]], ])
wy_conv_xy = np.array([[[[-1.0]], [[-1.0]], [[-1.0]]], [[[0.0]], [[0.0]], [[0.0]]],
[[[1.0]], [[1.0]], [[1.0]]], ])

wx_conv_xx = jnp.array(wx_conv_xx, dtype = jnp.float32)
wx_conv_xy = jnp.array(wx_conv_xy, dtype = jnp.float32)
wy_conv_yy = jnp.array(wy_conv_yy, dtype = jnp.float32)
wy_conv_xy = jnp.array(wy_conv_xy, dtype = jnp.float32)
fx_conv_xx = conv2d(stress_xx_matrix_4d, wx_conv_xx)
fx_conv_xy = conv2d(stress_xy_matrix_4d, wx_conv_xy)
fx_conv_sum = fx_conv_xx + fx_conv_xy
fy_conv_yy = conv2d(stress_yy_matrix_4d, wy_conv_yy)
fy_conv_xy = conv2d(stress_xy_matrix_4d, wy_conv_xy)
fy_conv_sum = fy_conv_yy + fy_conv_xy
fx_conv_sum_norm = jnp.divide(fx_conv_sum, y_pred_m_conv)
fy_conv_sum_norm = jnp.divide(fy_conv_sum, y_pred_m_conv)

# Calculate loss
# The loss is calculated by taking the mean absolute error (MAE) between the predicted and actual stresses
# The loss is minimized using the Adam optimizer

# Computes the loss based on the mean absolute error between the predicted and actual stresses

mean_modu = jnp.array(np.mean(y_elas), dtype = jnp.float32)
loss_x = jnp.mean(jnp.abs(fx_conv_sum_norm))
loss_y = jnp.mean(jnp.abs(fy_conv_sum_norm))
loss_m = jnp.abs(jnp.mean(y_pred_m) - mean_modu)
loss_v = jnp.abs(jnp.mean(y_pred_v))
loss = loss_x + loss_y + loss_m/100 + loss_v/100
err = jnp.sum(jnp.abs(y_elas - y_pred_m))


# Having trouble here converting the optimizer to JAX
train_step = tf.compat.v1.train.AdamOptimizer(learn_rate).minimize(loss) # Optimizes the loss function using the Adam optimizer
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training process
# The model is trained using the training data
# Runs the training step 200,001 times, printing the loss and error every 100 iterations
# Saves the predicted stresses to a file
# Prints the elapsed time

start_time  = time.time()
for i in range(200001):
    sess.run(train_step, feed_dict = {xs_elas: x_elas, xs_disp: x_disp, ys_disp: y_disp})
    if i % 100 == 0:
        err_vale = np.array(sess.run(err, feed_dict = {xs_elas: x_elas}))
        loss_vale = np.array(sess.run(loss, feed_dict = {xs_elas: x_elas, xs_disp: x_disp, ys_disp: y_disp}))
        print(i, loss_vale, err_vale)
y_pred_m_value = sess.run(y_pred_m, feed_dict = {xs_elas: x_elas})
y_pred_v_value = sess.run(y_pred_v, feed_dict = {xs_disp: x_disp})
np.savetxt('y_pred_m_final', y_pred_m_value)
np.savetxt('y_pred_v_final', y_pred_v_value)
print("--- %s Elapsed time ---" % (time.time() - start_time))
