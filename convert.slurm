#!/bin/bash
#SBATCH --job-name={jobname_c}
#SBATCH -A <account>@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module purge
module load anaconda-py3/2024.06
conda activate <path_to_conda_env>

python convert_to_h5.py
