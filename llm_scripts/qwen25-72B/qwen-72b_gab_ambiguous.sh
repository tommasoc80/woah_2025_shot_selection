#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --job-name=qwen_ambiguous
#SBATCH --mem=50G
#SBATCH --output=gab_ambiguous.out

module load Python/3.11.3-GCCcore-12.3.0
source ./bin/activate
#python llama3_3b.py
python qwen_72b_gab_ambiguous.py --max-model-len 4096 --n-gpus 4