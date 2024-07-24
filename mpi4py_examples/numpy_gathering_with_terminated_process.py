"""
From: modified from https://mpi4py.readthedocs.io/en/latest/tutorial.html with ChatGPT's advice
To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

import sys

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 2: sys.exit(0)

new_group = comm.group.Excl(np.array([2]))  # a group excluding the terminated process(es)
# new_group = comm.group.Incl(np.array([0, 1, 3]))  # alternatively, a group including only the non-terminated process(es)
new_comm = comm.Create_group(new_group)  # an Intracomm (just like comm) corresponding to the new group
new_rank = new_comm.Get_rank()
new_size = new_comm.Get_size()
print(f'[Process {rank}]: My rank in new_group of size {new_size} is {new_rank}')

sendbuf = np.zeros(10, dtype='i') + rank
print(f'[Process {rank}]: My messsage to send is {sendbuf}')

recvbuf = np.empty([new_size, 10], dtype='i') if (rank == 0) else None
new_comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
    print(f'[Process {rank}]: I gathered all the messages into this array:\n{recvbuf}')