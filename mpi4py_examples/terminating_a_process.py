"""
From: ChatGPT
To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`
"""

from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Example condition: terminate process with rank 2
if rank == 2:
    print(f"Process {rank} terminating.")
    sys.exit(0)  # This will terminate only the current process. 
                 # 0 indicates successful termination; nonzero value is considered abnormal termination

# Continue with the rest of the program for other processes
print(f"Process {rank} continues.")

# Finalize the MPI environment
# MPI.Finalize()  # not sure why needed
