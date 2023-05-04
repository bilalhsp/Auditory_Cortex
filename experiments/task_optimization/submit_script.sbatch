#!/bin/sh

#SBATCH --output=./result.out

#SBATCH	-A standby
#SBATCH --constraint=E|F

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
 
hostname
echo $CUDA_VISIBLE_DEVICES
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5


python task_optimization.py
