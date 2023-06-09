#!/bin/bash
#SBATCH -A m636
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --qos="regular"
#SBATCH --gpus-per-task=0

export SLURM_CPU_BIND="cores"
module load python
conda activate losslandscapes
srun sh run_n_dim_loss_calculate.sh
conda deactivate
