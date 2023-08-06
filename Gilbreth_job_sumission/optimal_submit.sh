#!/bin/sh

#SBATCH --output=./result_optimal.out

#SBATCH	-A standby

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=4:00:00

hostname
echo $CUDA_VISIBLE_DEVICES
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5
#module load conda-env/wav2letter_pretrained-py3.8.5

python ./../scripts/save_optimal_inputs.py $@
