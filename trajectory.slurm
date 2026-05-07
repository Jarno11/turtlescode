#!/bin/bash
#SBATCH --job-name={jobname_t}
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

LATEST_SOL=$(ls -v solution/{simname}_t*.pyfrs | tail -n 1)
T0=$(echo $LATEST_SOL | grep -oP '(?<=_t)[\d\.]+(?=\.pyfrs)')
TEND=$(echo "scale=3; $T0 + 80.1" | bc)

sed -i "s/^t0 =.*/t0 = $T0/" trajectory.ini
sed -i "s/^tend =.*/tend = $TEND/" trajectory.ini

srun --mpi=pmi2 python3 -u -m pyfr -p restart -b cuda partitiondir/cylinder.pyfrm "$LATEST_SOL" trajectory.ini
