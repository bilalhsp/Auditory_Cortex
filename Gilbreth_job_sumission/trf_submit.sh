#!/bin/sh

# --output=./result_regression_w2l_spect.out

#SBATCH	-A standby
#SBATCH --constraint=I|J|K|N|G|F|H|C

#F|G|I|K|D|B
# jgmakin-n, standby, training
# all valid: I|J|K|N|G|F|H|C|D|B
# very high mem: I|J|K      # 80GB
# High Mem GPUs: I|J|K|N|G    # 40GB
# High Mem GPUs: I|J|K|N|G|F|H|C    # 24GB
# very Fast GPUs: F|G|K
# Fast GPUs: D
# Slow GPUs: E
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50GB
#SBATCH --time=04:00:00
#SBATCH --output=./output_logs/%j.out

#SBATCH --job-name=run_trf    # Job name
# activate virtual environment
source ./env_setup.sh

python ../scripts/run_trf.py $@

