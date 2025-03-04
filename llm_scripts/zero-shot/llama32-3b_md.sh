#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu:2
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=llama32_zero
#SBATCH --mem=50G
#SBATCH --output=md3b_zero_shot.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python llama32_3b_md.py