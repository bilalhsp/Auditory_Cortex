#!/bin/sh

# --output=./result_train_STRF50.out

#SBATCH	-A jgmakin-n
# standby, jgmakin-n
#SBATCH --constraint=F|G|I|K|D|B|H|J|N
#|B|H|J
#F|G|I|K|D|B|H|J

# High Mem GPUs: F|G|I|K|D
# very Fast GPUs: F|K
# Fast GPUs: D
# Slow GPUs: E

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
# --mem=0
#SBATCH --time=4:00:00
# activate virtual environment
source ./env_setup.sh

python ../scripts/train_STRF.py $@