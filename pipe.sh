#!/bin/sh
#SBATCH -N 1
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH -n 6
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -o embed_output%j.out

module avail
module load cuda cudnn
module load python/3.11

source ../../env/bin/activate


python ./hierarchical_trial.py