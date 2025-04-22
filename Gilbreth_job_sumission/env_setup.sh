#!/bin/bash

# Environment setup for Slurm jobs
hostname
export NUMBA_DISABLE_INTEL_SVML=1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

module purge
module load external
module load conda 
conda activate /home/ahmedb/.conda/envs/cent7/2020.11-py38/wav2letter
