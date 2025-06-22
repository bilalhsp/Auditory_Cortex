#!/bin/sh

#SBATCH	-A jgmakin-n 
# --constraint=F|G|I|K|D|B|H|J|N 

# standby
# jgmakin-n 
# training
# High Mem GPUs: F|G|I|K|D
# very Fast GPUs: F|K
# Fast GPUs: B|D
# Slow GPUs: E

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-04:00:00
# --mem=40GB
# --output=./result_normalizer50.out
#SBATCH --job-name=normalizers    # Job name

# activate virtual environment
source ./env_setup.sh

python ../scripts/normalizer_save_results2.py $@

