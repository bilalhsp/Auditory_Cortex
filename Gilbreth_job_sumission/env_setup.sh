#!/bin/bash

# Environment setup for Slurm jobs
echo "Hostname: $(hostname)"
if [[ -n "$SLURM_MEM_PER_NODE" && "$SLURM_MEM_PER_NODE" =~ ^[0-9]+$ ]]; then
    echo "Allocated memory per node: $((${SLURM_MEM_PER_NODE} / 1024)) GB"
else
    echo "SLURM_MEM_PER_NODE is not set or not a valid number."
fi
echo "Number of GPUs: $SLURM_GPUS_PER_NODE"
export NUMBA_DISABLE_INTEL_SVML=1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU Info:"
nvidia-smi

module purge
module load external
module load conda 
conda activate /home/ahmedb/.conda/envs/cent7/2020.11-py38/wav2letter
