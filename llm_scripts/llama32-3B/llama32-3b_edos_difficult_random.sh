#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=llama32-3_difficult_random
#SBATCH --mem=50G
#SBATCH --output=edos_difficult.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python llama32_3b_edos_difficult_random.py