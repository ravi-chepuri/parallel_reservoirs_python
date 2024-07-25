#!/bin/bash
#
#SBATCH --job-name=parallel_rc          ## name of job
#SBATCH --partition=standard            ## no special requirements -> standard
#SBATCH --time=0-00:01:00               ## wall time (D-HH:MM:SS)
#
#SBATCH --ntasks=16                     ## number of tasks
#SBATCH --cpus-per-task=1               ## number of CPUs per task
#
#SBATCH --mem-per-cpu=4096              ## memory per CPU (MiB)
#SBATCH --oversubscribe                 ## willing to share node with other jobs
#SBATCH --export=NONE                   ## don't inherit environment from when sbatch was run
#
#SBATCH --output=%A_%a.out              ## output file to redirect stdout to (jobID_taskID.out)
#SBATCH --error=%A_%a.err               ## error file to redirect stderr to (jobID_taskID.err)

. ~/.bashrc

conda activate parallel_env  ## have installed mpi4py in this env with `conda install -c conda-forge mpi4py openmpi`

mpiexec -n $SLURM_NTASKS python parallel_rc.py