#!/bin/sh

#SBATCH --output=./result_glm_ds2_6.out

#SBATCH	-A standby
#SBATCH --constraint=F|G|I|K|D|B|H|J
#F|G|I|K|D|B

# High Mem GPUs: F|G|I|K|D|B
# very Fast GPUs: F|G|K
# Fast GPUs: D
# Slow GPUs: E
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=0
#SBATCH --time=4:00:00

hostname
NUMBA_DISABLE_INTEL_SVML=1
echo $CUDA_VISIBLE_DEVICES
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5
# module load gcc/9.3.0
#module load conda-env/wav2letter_pretrained-py3.8.5


python ../scripts/save_glm.py $@
# python deepspeech2_testing.py
