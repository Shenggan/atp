#!/bin/bash

#SBATCH -J myjob           # Job name
#SBATCH -o %j.o       # Name of stdout output file
#SBATCH -e %j.e       # Name of stderr error file
#SBATCH -p development
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 00:05:00

module load mvapich2

source $HOME/.pyrc
source activate torch-1.13

oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

HEAD_HOST=`hostname -I | awk '{print $2}'`
export CCL_HOST=$HEAD_HOST
export CCL_MNIC_NAME=mlx5_0
export CCL_WORKER_COUNT=4

export OMP_NUM_THREADS=16

ibrun python test.py
