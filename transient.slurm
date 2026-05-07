#!/bin/bash
#SBATCH --job-name={jobname_it}
#SBATCH -A <account>@h100
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module purge
module load arch/h100
module load anaconda-py3/2024.06 openmpi/4.1.5-cuda
conda activate <path_to_conda_env>

export OMP_NUM_THREADS=1

srun --mpi=pmi2 python3 -u -m pyfr run -b cuda partitiondir/cylinder.pyfrm transient.ini

python3 solution/transient_check.py
