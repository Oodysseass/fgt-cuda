#!/bin/bash
#SBATCH --job-name=DeviceQuery
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

nvidia-smi
nvprof ./deviceQuery
