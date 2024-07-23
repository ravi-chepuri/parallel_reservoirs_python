"""
Aims to implement a parallelized parallel RC architecture for forecasting spatiotemporal chaos.

To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # process number
size = comm.Get_size()  # total number of processes


# input data specifications
Q = 64  # number of spatial 1D grid points
d = 1  # dimension of system at each spatial point
T = 100  # number of time steps of training data

# parallel RC specifications
q = 2  # number of contiguous grid points per reservoir
l = 1  # number of spatial points in the contiguous buffer regions


def train_vanilla_RC(resparams, data):
    """TODO"""
    N = resparams['N']
    radius = resparams['radius']
    degree = resparams['degree']
    num_inputs = resparams['num_inputs']
    sigma = resparams['sigma']
    train_length = resparams['train_length']

    A = generate_reservoir.erdos_renyi_reservoir(N, degree, radius)
    W_in = generate_reservoir.generate_W_in(num_inputs, N, sigma)

    res_states = rc.reservoir_layer(A, W_in, data, resparams)

    W_out = rc.fit_output_weights(resparams, res_states, data[:, :train_length])

    return W_out, A, W_in, res_states