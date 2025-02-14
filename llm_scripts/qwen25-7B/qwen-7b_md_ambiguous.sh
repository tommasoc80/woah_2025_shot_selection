#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --job-name=llama3-8_ambiguous
#SBATCH --mem=50G
#SBATCH --output=md_ambiguous.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python qwen_7b_md_ambiguous.py