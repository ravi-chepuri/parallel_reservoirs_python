"""
A parallel RC architecture for forecasting spatiotemporal dynamics, parallelized using mpi4py.

To run, launch on a cluster with `sbatch run_parallel_rc.sh`.
"""

import sys

import numpy as np
from mpi4py import MPI

from rc import make_input_matrix, make_reservoir, run_open_loop, fit_output_weights, Hyperparameters


def forecast_spatiotemporal_system(train_data, hyperparams, Q=128, d=1, q=8, l=6):
    """Given a trajectory of a spatiotemporal system, makes a forecast of the future trajectory using a parallel
    reservoir computing scheme. Utilizes one process for each reservoir.

    Args:
        train_data (ndarray): Past trajectory of the system that serves as training data. Of shape (timesteps x number
            of grid points x system dimension)
        hyperparams (Hyperparameters): Hyperparameters of the individual reservoirs.
        Q (int, optional): Number of spatial grid points of the system. Should equal train_data.shape[1].
            Defaults to 128.
        d (int, optional): System dimension. Ex. for a scalar field, d=1; for a vector field in 3D, d=3. Should equal 
            train_data.shape[2].Defaults to 1.
        q (int, optional): Number of spatial points per reservoir. Defaults to 8.
        l (int, optional): Number of grid points in overlap regions. Defaults to 6.

    Returns:
        ndarray: Forecasted trajectory of the system that continues the training trajectory. Of shape (timesteps x 
            number of grid points x system dimension)
    """

    # checks
    assert (Q, d) == (train_data.shape[1], train_data.shape[2]), "System size and dimension do not match data"
    assert Q % q == 0, "Number of grid points is not a multiple of group size"
    g = Q // q  # number of reservoirs needed
    assert l <= q, "Size of input overlap region is too large"
    assert hyperparams.num_inputs == (q+2*l)*d, "num_inputs in hyperparameters must be (q+2*l)*d"

    # use only the necessary processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # process number
    size = comm.Get_size()  # total number of processes
    assert size >= g, "Number of processes is less than number of reservoirs needed"
    if rank >= g: 
        sys.exit(0)  # close unnecessary processes
    new_group = comm.group.Incl(range(g))
    new_comm = comm.Create_group(new_group)
    reservoir_id = new_comm.Get_rank()

    ## Training ##
    start_train_time = MPI.Wtime()
    grid_points = np.arange(reservoir_id * q, (reservoir_id+1) * q)  # q grid points that reservoir predicts
    input_grid_points = np.arange(reservoir_id * q - l, (reservoir_id+1) * q + l) % Q  # q+2l grid points used as input
    train_inputs = train_data[:, input_grid_points, :]
    train_targets = train_data[:, grid_points, :]
    # flatten the system dimensions so now every d entries in the 2nd dimension is a grid point
    train_inputs = train_inputs.reshape((-1, (q+2*l)*d))
    train_targets = train_targets.reshape((-1, q*d))

    A = make_reservoir(hyperparams)
    W_in = make_input_matrix(hyperparams)
    start_open_loop_time = MPI.Wtime()
    training_res_states = run_open_loop(A, W_in, train_inputs, hyperparams)
    end_open_loop_time = MPI.Wtime()
    res_states_for_fit = training_res_states[hyperparams.discard_transient_length:-1]
    targets_for_fit = train_targets[hyperparams.discard_transient_length+1:hyperparams.train_length]
    start_fit_time = MPI.Wtime()
    W_out = fit_output_weights(res_states_for_fit, targets_for_fit, hyperparams)
    end_fit_time = MPI.Wtime()
    end_train_time = MPI.Wtime()

    new_comm.Barrier()

    ## Forecasting ##
    start_forecast_time = MPI.Wtime()
    communication_wait_times = np.empty(hyperparams.prediction_steps)
    left_neighbor_id = (reservoir_id - 1) % g  # periodic boundary conditions
    right_neighbor_id = (reservoir_id + 1) % g
    left_input_buffer = np.empty(l*d)
    right_input_buffer = np.empty(l*d)

    local_predictions = np.zeros((hyperparams.prediction_steps, q*d))
    res_state = training_res_states[-1]
    for t in range(hyperparams.prediction_steps):  # closed loop with parallel reservoirs
        # Make prediction from past reservoir state
        local_predictions[t] = W_out @ res_state
        # Send predictions that are in neighbors' overlap regions to the neighbors
        start_communication_time = MPI.Wtime()
        new_comm.Send(local_predictions[t, :l*d], dest=left_neighbor_id, tag=t)
        new_comm.Send(local_predictions[t, -l*d:], dest=right_neighbor_id, tag=t)
        # Receive predictions in our own overlap regions from the neighbors
        new_comm.Recv(left_input_buffer, source=left_neighbor_id, tag=t)
        new_comm.Recv(right_input_buffer, source=right_neighbor_id, tag=t)
        end_communication_time = MPI.Wtime()
        communication_wait_times[t] = end_communication_time - start_communication_time
        # Combine received predictions with own predictions
        inputs = np.hstack([left_input_buffer, local_predictions[t], right_input_buffer])
        # Update res_state
        res_state = (1 - hyperparams.leakage) * res_state \
                    + hyperparams.leakage * hyperparams.activation_func(A @ res_state \
                                                                        + W_in @ inputs \
                                                                        + hyperparams.bias)

    # gather predictions in the root (rank 0) process
    predictions = np.empty((g, hyperparams.prediction_steps, q*d)) if (reservoir_id == 0) else None  # (g x T x q*d)
    new_comm.Gather(local_predictions, predictions, root=0)
    end_forecast_time = MPI.Wtime()

    # gather timing info in the root process
    local_timing_info = np.array([end_train_time - start_train_time,
                                  end_open_loop_time - start_open_loop_time,
                                  end_fit_time - start_fit_time,
                                  end_forecast_time - start_forecast_time,
                                  np.sum(communication_wait_times)])
    timing_info = np.empty((g, *local_timing_info.shape)) if (reservoir_id == 0) else None
    new_comm.Gather(local_timing_info, timing_info, root=0)

    if reservoir_id == 0:
        mean_timing_info = np.mean(timing_info, axis=0)
        print(f'Mean total train time (s): {mean_timing_info[0]:.6f}')
        print(f'\tMean total open loop time (s): {mean_timing_info[1]:.6f}')
        print(f'\tMean total fitting time (s): {mean_timing_info[2]:.6f}')
        print(f'Mean total forecast time (s): {mean_timing_info[3]:.6f}')
        print(f'\tMean total communication wait time (s): {mean_timing_info[4]:.6f}')

    if reservoir_id == 0:
        # reshape to (T x Q x d) (Q=gq)
        return np.swapaxes(predictions, 0, 1).reshape((hyperparams.prediction_steps, Q, d))
    else:
        return None


if __name__ == '__main__':

    # load data and set up train/test split
    data = np.load('kuramoto_sivashinsky/trajectories/trajectory_0.npy')
    discard_system_transient_length = 500
    data = data[discard_system_transient_length:]
    num_training_timesteps = 20000
    train_data, test_data = np.vsplit(data, [num_training_timesteps])
    _, Q, d = train_data.shape  # _, number of spatial 1D grid points, dimension of system at each grid point

    # parallel RC specifications
    q = 8  # number of contiguous grid points per reservoir
    l = 6  # number of spatial points in the contiguous input overlap regions

    # set forecasting hyperparameters
    hyperparams = Hyperparameters(num_inputs=(q+2*l)*d,
                       N=4000, degree=3, radius=0.6, leakage=1., bias=1., sigma=0.1, beta=1e-6, 
                       discard_transient_length=100, activation_func=np.tanh,
                       dt=0.25,
                       train_length=num_training_timesteps, prediction_steps=2000)
    
    predictions = forecast_spatiotemporal_system(train_data, hyperparams, Q=Q, d=d, q=q, l=l)

    if predictions is not None:
        np.save('predictions/predictions.npy', predictions)