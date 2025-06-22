#!/bin/sh

# --output=./result_regression_w2l_spect.out

#SBATCH	-A jgmakin-n
# --constraint=I|J|K|N|G|F|H|C

#F|G|I|K|D|B

# jgmakin-n, standby, training
# all valid: I|J|K|N|G|F|H|C|D|B
# very high mem: I|J|K      # 80GB
# High Mem GPUs: I|J|K|N|G|F|H|C
# very Fast GPUs: F|G|K
# Fast GPUs: D
# Slow GPUs: E
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=2-04:00:00

#SBATCH --job-name=multi-layer    # Job name

# activate virtual environment
source ./env_setup.sh

python ../scripts/run_trf_all_layers.py $@

