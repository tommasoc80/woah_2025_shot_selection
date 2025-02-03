#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=llama3-8_random
#SBATCH --mem=50G
#SBATCH --output=brexit_random_shuffle.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python llama3_8b_brexit_random_shuffle.py