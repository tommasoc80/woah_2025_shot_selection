#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=llama3-8
#SBATCH --mem=50G
#SBATCH --output=md_difficult_random.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python llama3_8b_md_difficult_random.py