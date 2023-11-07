#!/bin/sh

#SBATCH --output=./result_RSA3.out

#SBATCH	-A standby
#	-A debug


#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=0
#SBATCH --time=4:0:00

hostname
echo $CUDA_VISIBLE_DEVICES
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5
#module load conda-env/wav2letter_pretrained-py3.8.5

python ./../scripts/RSA_save_results.py $@
