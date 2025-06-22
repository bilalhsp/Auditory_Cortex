#!/bin/sh

#SBATCH	-A jgmakin-n
#standby
#jgmakin-n
#training
#debug
# --constraint=F|G|I|K|D|B|H|J 
# --constraint=~n003

# High Mem GPUs: F|G|I|K|D
# very Fast GPUs: F|K
# Fast GPUs: B|D
# Slow GPUs: E

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH --time=8:00:00

# activate virtual environment
source ./env_setup.sh


python ../scripts/save_qualitative_prediction.py $@

