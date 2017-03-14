#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=zachzhang
#SBATCH --time=8:00:00


module purge
module load python3/intel/3.5.2
module load pytorch/intel/20170125

cd /home/zz1409/DeepLearningAssignments


python ladder_train.py
