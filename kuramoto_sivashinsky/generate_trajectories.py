import numpy as np

from ks_etdrk4 import kursiv_predict


timestep = 0.25
prediction_steps = 50000
system_size = 22
grid_points = 64

num_trajectories = 16

for i in range(num_trajectories):

    u0 = np.random.uniform(low=-0.6, high=0.6, size=grid_points)
    u0 = u0 - np.mean(u0)

    u_arr, params = kursiv_predict(u0, tau=timestep, N=grid_points, d=system_size, T=prediction_steps, 
                                params = np.array([[],[]], dtype = np.complex128),
                                int_steps = 1, noise = np.zeros((1,1), dtype = np.double))  # what is params??
    u_arr = np.ascontiguousarray(u_arr) / (1.1876770355823614)  # I took the number from the class Numerical Model,
                                                                # but not really sure where it came from

    u_arr_for_saving = np.expand_dims(u_arr.T, axis=-1)  # T x Q x d (with d=1)

    np.save(f'kuramoto_sivashinsky/trajectories/trajectory_{i}.npy', u_arr_for_saving)
    print(i)