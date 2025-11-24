"""
To run, first install mpi4py in your conda environment (`conda install -c conda-forge mpi4py openmpi`), then run
`mpiexec -n 4 python filename.py`

GOAL:
Consider agents located on a directed network as shown:

A-->B
^   |
|   |
V   V
C<->D   

Each agent holds a real number. Time moves forward in discrete steps. At every time step, each agent changes their
number to be the average of all of their upstream neighbors' numbers at the previous time step.

I aim to implement this using separate CPU processes for each agent, using mpi4py.

"""

from mpi4py import MPI
import numpy as np
import networkx as nx

import socket, sys
print("Hello from host", socket.gethostname(), "python:", sys.executable, flush=True)

comm = MPI.COMM_WORLD
my_agent_id = comm.Get_rank()  # process number. Thruout, `my_...` indicates variable that I expect to be different for
                               # each process
num_agents = comm.Get_size()  # total number of processes

adjacency_matrix = np.array([[0, 1, 1, 0],
                             [0, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 0, 1, 0]])
graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# directed adjacency lists
downstream_neighbors_lists = {node: list(neighbors) for node, neighbors in graph.adjacency()}
upstream_neighbors_lists = {node: list(neighbors) for node, neighbors in graph.reverse().adjacency()}
my_downstream_neighbors = downstream_neighbors_lists[my_agent_id]
my_upstream_neighbors = upstream_neighbors_lists[my_agent_id]

initial_numbers = {
    0: 10,
    1: 20,
    2: 30,
    3: 40
}
my_number = np.array(initial_numbers[my_agent_id], dtype=np.float64)  # 0d numpy array

my_incoming_buffers = [np.empty([]) for upstream_neighbor in my_upstream_neighbors]

sim_length = 4
my_log = np.empty(sim_length+1)
my_log[0] = my_number
print(f'Agent {my_agent_id}, time 0: My number is {my_number}')
for t in range(1, sim_length+1):
    for downstream_neighbor in my_downstream_neighbors:
        comm.Send(my_number, dest=downstream_neighbor, tag=t)
        print(f'Agent {my_agent_id}, time {t}: Sent my number {my_number} to agent {downstream_neighbor}')
    for i, upstream_neighbor in enumerate(my_upstream_neighbors):
        comm.Recv(my_incoming_buffers[i], source=upstream_neighbor, tag=t)
        print(f'Agent {my_agent_id}, time {t}: Received number {my_number} from agent {upstream_neighbor}')
    received = np.array(my_incoming_buffers)
    my_number[...] = np.mean(received)
    my_log[t] = my_number
    print(f'Agent {my_agent_id}, time {t}: My number is {my_number}')

np.savetxt(f'{my_agent_id}.csv', my_log)  # logs match the result of working the game out by hand!
