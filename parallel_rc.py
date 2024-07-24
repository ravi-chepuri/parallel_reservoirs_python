"""
Aims to implement a parallelized parallel RC architecture for forecasting spatiotemporal dynamics.

To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

import sys

from mpi4py import MPI
import numpy as np

import rc


comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # process number
size = comm.Get_size()  # total number of processes

# data
data = None  # TODO
T, Q, d = data.shape  # number of time steps, number of spatial 1D grid points, dimension of system at each grid point
num_training_timesteps = 10000
train_data, test_data = np.vsplit(data, [num_training_timesteps])

# parallel RC specifications
q = 2  # number of contiguous grid points per reservoir
l = 1  # number of spatial points in the contiguous input overlap regions

# checks
assert Q % q == 0, "Number of grid points is not a multiple of group size"
g = Q // q  # number of reservoirs needed
assert size >= g, "Number of processes is less than number of reservoirs needed"
assert l <= q, "Size of input overlap region is too large"

# use only the necessary processes
if rank >= g: sys.exit(0)  # close unnecessary processes
new_group = comm.group.Incl(range(g))
new_comm = comm.Create_group(new_group)
reservoir_id = new_comm.Get_rank()
assert new_comm.Get_size() == g

## Training ##
grid_points = np.arange(reservoir_id * q, (reservoir_id+1) * q)  # q grid points that reservoir predicts
input_grid_points = np.arange(reservoir_id * q - l, (reservoir_id+1) * q + l)  # q+2l grid points that are used as input
train_inputs = train_data[:, input_grid_points, :]
train_targets = train_data[:, grid_points, :]
# flatten the system dimensions so now every d entries in the 2nd dimension is a grid point
train_inputs = train_inputs.reshape((-1, (q+2*l)*d))
train_targets = train_targets.reshape((-1, q*d))

h = rc.Hyperparameters(num_inputs=(q+2*l)*d,
                        N=500, degree=2.6667, sigma=0.5, beta=0.00003, dt=0.02,
                        train_length=num_training_timesteps, prediction_steps=1000)  # TODO: change these

W_out, A, W_in, training_res_states = rc.train_RC(train_inputs, h, train_targets=train_targets)

## Forecasting ##
left_neighbor_id = (reservoir_id + 1) % Q  # periodic boundary conditions
right_neighbor_id = (reservoir_id - 1) % Q
left_input_buffer = np.empty(l*d)
right_input_buffer = np.empty(l*d)

local_predictions = np.zeros((h.prediction_steps, q*d))
res_state = training_res_states[-1]
for t in range(h.prediction_steps):  # closed loop with parallel reservoirs
    # Make prediction from past reservoir state
    local_predictions[t] = W_out @ res_state
    # Send predictions that are in neighbors' overlap regions to the neighbors
    new_comm.Send(local_predictions[t, :l*d], dest=left_neighbor_id, tag=t)
    new_comm.Send(local_predictions[t, -l*d:], dest=right_neighbor_id, tag=t)
    # Receive predictions in our own overlap regions from the neighbors
    new_comm.Recv(left_input_buffer, source=left_neighbor_id, tag=t)
    new_comm.Recv(right_input_buffer, source=right_neighbor_id, tag=t)
    # Combine received predictions with own predictions
    inputs = np.hstack([left_input_buffer, local_predictions[t], right_input_buffer])
    # Update res_state
    res_state = (1 - h.leakage) * res_state \
                + h.leakage * h.activation_func(A @ res_state \
                                                + W_in @ inputs \
                                                + h.bias)

## Gathering predictions in the root (rank 0) process ##
predictions = np.empty((g, h.prediction_steps, q*d)) if (reservoir_id == 0) else None  # shape (g x T x q*d)
new_comm.Gather(local_predictions, predictions, root=0)

if reservoir_id == 0:
    predictions = np.swapaxes(predictions, 0, 1).reshape((h.prediction_steps, Q, d))  # reshape to (T x Q x d) (Q=gq)
    np.save('predictions.npy', predictions)