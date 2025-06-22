#!/bin/sh

# --output=./result_cache_features_5.out

#SBATCH	-A standby
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
#SBATCH --mem=40GB
#SBATCH --time=4:00:00

# activate virtual environment
source ./env_setup.sh

python ../scripts/cache_features.py $@
# python deepspeech2_testing.py
