#!/bin/sh

# --output=./result_normalizer50.out

#SBATCH	-A jgmakin-n
#SBATCH --constraint=F|G|I|K|D|B|H|J|N 

# High Mem GPUs: F|G|I|K|D
# very Fast GPUs: F|K
# Fast GPUs: B|D
# Slow GPUs: E


#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
# --mem=0
#SBATCH --time=04:00:00

hostname
NUMBA_DISABLE_INTEL_SVML=1
echo $CUDA_VISIBLE_DEVICES
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5



python ../scripts/normalizer_save_results.py $@

