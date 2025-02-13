#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --job-name=qwen-7b
#SBATCH --mem=80G
#SBATCH --output=edos_ambiguous_random.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python qwen_7b_edos_ambiguous_random.py