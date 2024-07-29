"""
From: https://mpi4py.readthedocs.io/en/latest/tutorial.html
To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

from mpi4py import MPI
import numpy as np
import time

start_time = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # process number
size = comm.Get_size()  # total number of processes

if rank == 0:
    data = np.arange(100, dtype=np.float64)
    print(f'[Process {rank}]: Created numpy array: `data = np.arange(100, dtype=np.float64)`')
    time.sleep(1.)
    comm.Send(data, dest=1, tag=13)
    print(f'[Process {rank}]: Sent `data` to process 1 with tag 13')
elif rank == 1:
    data = np.empty(100, dtype=np.float64)
    before_receive_time = MPI.Wtime()
    comm.Recv(data, source=0, tag=13)
    after_receive_time = MPI.Wtime()
    print(f'[Process {rank}]: Received `data` from process 0 with tag 13: {data}')
    print(f'[Process {rank}]: Total time: {after_receive_time - start_time}\nReceive waiting time: {after_receive_time-before_receive_time}\nFraction receive waiting: {(after_receive_time-before_receive_time)/(after_receive_time - start_time)}')