"""
From: https://mpi4py.readthedocs.io/en/latest/tutorial.html
To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.zeros(10, dtype='i') + rank
print(f'[Process {rank}]: My messsage to send is {sendbuf}')

recvbuf = np.empty([size, 10], dtype='i') if (rank == 0) else None
comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
    for i in range(size):
        assert np.allclose(recvbuf[i,:], i)
    print(f'[Process {rank}]: I gathered all the messages into this array:\n{recvbuf}')