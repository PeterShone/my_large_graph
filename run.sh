#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu_24h
#SBATCH --gres=gpu:1
#SBATCH --output=out5.log
eval "$(/research/dept8/fyp21/cwf2101/peter/miniconda3/bin/conda shell.bash hook)"
conda activate graph
CUDA_LAUNCH_BLOCKING=1 python3 scripts/train_network.py

