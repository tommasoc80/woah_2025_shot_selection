#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --job-name=llama370
#SBATCH --mem=50G
#SBATCH --output=edos_ambiguous70B.out

module load Python/3.11.3-GCCcore-12.3.0
source ./bin/activate
#python llama3_3b.py
python llama3_70b_edos_ambiguous.py --max-model-len 4096 --n-gpus 4
