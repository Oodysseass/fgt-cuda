#!/bin/bash
#SBATCH --job-name=GraphFreq
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=59:00

module load gcc/7.3.0 cuda/10.0.130
nvcc src/main.cu src/mtx.cu src/fglt.cu -O3 -o main

./main test.mtx 
mv freq_net.csv results/resultstest/

./main auto.mtx 
mv freq_net.csv results/resultsauto/

./main great-britain_osm_coord.mtx 
mv freq_net.csv results/resultsgreat/

./main delaunay_n22.mtx
mv freq_net.csv results/resultsdel/
