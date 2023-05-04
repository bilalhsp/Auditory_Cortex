#!/bin/sh

#SBATCH --output=./result_pca_topo.out

#SBATCH	-A standby

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=0
#SBATCH --time=4:00:00


module load use.own
module load conda-env/wav2letter-py3.8.5

python pca_topography_results.py