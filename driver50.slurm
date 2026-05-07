#!/bin/bash
#SBATCH --job-name=driver50
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

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYFR_XSMM_LIBRARY_PATH=$WORK/software/libxsmm/lib/libxsmm.so
export LD_LIBRARY_PATH=$WORK/software/scotch-install/lib64:$LD_LIBRARY_PATH
export PYFR_SCOTCH_LIBRARY_PATH=$WORK/software/scotch-install/lib64/libscotch.so

python driver.py --setup_nr 50
