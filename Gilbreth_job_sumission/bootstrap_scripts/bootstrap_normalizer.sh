#!/bin/sh

#SBATCH	-A jgmakin-n 

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
#SBATCH --time=3-04:00:00
# --mem=40GB
#SBATCH --output=./output_logs/%j.out
#SBATCH --job-name=bootstrap_norm    # Job name

# activate virtual environment
source ../env_setup.sh

python ../../scripts/bootstrap_normalizer.py $@


