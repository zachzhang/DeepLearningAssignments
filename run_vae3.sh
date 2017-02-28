#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=zachzhang

module purge
module load python3/intel/3.5.2
module load pytorch/intel/20170125

cd /home/zz1409/DeepLearningAssignments

##python vae_train3.py 0.
##python vae_train3.py .2
##python vae_train3.py .5
##python vae_train3.py 1.

python vae_train3.py 1.

