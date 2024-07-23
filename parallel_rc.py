"""
Aims to implement a parallelized parallel RC architecture for forecasting spatiotemporal chaos.

To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

from mpi4py import MPI
import numpy as np

import rc


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

assert Q % q == 0, "Number of grid points is not a multiple of group size"
assert size >= Q//q, "Number of processes is less than number of grid points"
assert l <= q, "Size of buffer region is too large"

