# Parallel reservoir computing in Python

A parallelized Python implementation of the parallel reservoir computing architecture, as presented in Pathak et al., 2018, using the `mpi4py` package.

**This is a work in progress!** Currently, the code runs for forecasting the Kuramoto Sivashinsky system but produces poor predictions. The forecasts are reasonable for the first few steps, which I beleive indicates that the code is functioning correctly. However, the predictions quickly lose accuracy and fail to capture the system's climate, which I believe is likely due to poorly chosen hyperparameters.

## Usage

Ensure you're in a conda environment with `mpi4py` installed (`conda install -c conda-forge mpi4py openmpi`), as well as other packages like numpy. On a cluster with slurm, queue up the parallel RC code with `sbatch run_parallel_rc.sh`. When the job is done, it will write the prediction to `predictions`; use the notebook inside to visualize the prediction.

To run some of the scripts in `mpi4py_examples`, run `mpiexec -n 4 python filename.py` (again, make sure you're in a conda environment with mpi4py installed).


## Contents

* `kuramoto_sivashinky` contains code for generating trajectores of the Kuramoto-Sivashinsky system, adapted from [Alex Wikner's code](https://github.com/awikner/res-noise-stabilization/blob/master/src/res_reg_lmnt_awikner/ks_etdrk4.py)
* `mpi4py_examples` contains illustrative examples of using `mpi4py` to do relevant tasks
    - `numpy_point-to-point.py`: Direct message passing of a numpy array from one process to another
    - `numpy_gathering.py`: Many-to-one message passing of a numpy array
    - `terminating_a_process.py`: Ending a process that's no longer needed. `numpy_gathering_with_terminated_process.py` shows how to do many-to-one communication after some processes have been terminated
    - `averaging_game.py`: A "game" in which agents on a network pass messages along the network using point-to-point communication, over many time steps. Done as an illustrative example while preparing to make the parallel RC code. Can be run on a cluster with `sbatch run_averaging_game.sh`
* `rc.py` contains vanilla reservoir computing code (not very optimized). `single_RC_forecasting.ipynb` shows how to use this code to forecast the Lorenz system (also uses `lorenz.py` and `prediction_analysis.py`)
* `parallel_rc.py` contains the parallelized parallel reservoir computing code. It calls code from `rc.py`. It must be run on a cluster with at least 32 available processes; this can be done by running `sbatch run_parallel_rc.sh`
* Predictions from `parallel_rc.py` get written to `predictions`, which also contains a jupyter notebook to visualize the predictions
