"""
From: https://mpi4py.readthedocs.io/en/latest/tutorial.html
To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # process number
size = comm.Get_size()  # total number of processes
# print(f'Size: {size}, Rank: {rank}')

# passing MPI datatypes explicitly
# if rank == 0:
#     data = np.arange(1000, dtype='i')
#     comm.Send([data, MPI.INT], dest=1, tag=77)
# elif rank == 1:
#     data = np.empty(1000, dtype='i')
#     comm.Recv([data, MPI.INT], source=0, tag=77)

# automatic MPI datatype discovery
# if rank == 0:
#     data = np.arange(100, dtype=np.float64)
#     comm.Send(data, dest=1, tag=13)
# elif rank == 1:
#     data = np.empty(100, dtype=np.float64)
#     comm.Recv(data, source=0, tag=13)

# messing around
if rank == 0:
    data = np.arange(100, dtype=np.float64)
    print(f'[Process {rank}]: Created numpy array: `data = np.arange(100, dtype=np.float64)`')
    comm.Send(data, dest=1, tag=13)
    print(f'[Process {rank}]: Sent `data` to process 1 with tag 13')
elif rank == 1:
    data = np.empty(100, dtype=np.float64)
    comm.Recv(data, source=0, tag=13)
    print(f'[Process {rank}]: Received `data` from process 0 with tag 13: {data}')