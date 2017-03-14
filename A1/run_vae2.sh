#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=zachzhang

module purge
module load python3/intel/3.5.2
module load pytorch/intel/20170125

cd /home/zz1409/DeepLearningAssignments

python vae_train2.py
