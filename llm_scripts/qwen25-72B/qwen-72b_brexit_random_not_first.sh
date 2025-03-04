#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00-08:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G
#SBATCH --output=brexit_random.out

module load Python/3.11.3-GCCcore-12.3.0
source ./bin/activate
#python llama3_3b.py
python qwen_72b_brexit_random.py --max-model-len 4096 --n-gpus 4