#!/bin/bash
#SBATCH --job-name=GraphFreq
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

module load gcc/7.3.0 cuda/10.0.130
nvcc src/main.cu src/mtx.cu src/matrixOps.cu -O3 -o main

./main auto.mtx
