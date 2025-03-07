#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00-08:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G
#SBATCH --output=edos70b_zero_shot.out

source ./bin/activate
#python llama3_3b.py
python llama3_70b_edos.py --max-model-len 4096 --n-gpus 4