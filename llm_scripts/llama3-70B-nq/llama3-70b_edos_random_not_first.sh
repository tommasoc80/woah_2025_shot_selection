#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00-08:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G
#SBATCH --output=edos_random70B.out

module load Python/3.11.3-GCCcore-12.3.0
source ./bin/activate
#python llama3_3b.py
python llama3_70b_edos_random_not_first.py --max-model-len 4096 --n-gpus 4