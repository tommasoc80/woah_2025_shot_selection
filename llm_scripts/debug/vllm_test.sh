#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --job-name=llama270-vllm
#SBATCH --mem=80G
#SBATCH --output=prova-batch.out

module load Python/3.11.3-GCCcore-12.3.0
source ./bin/activate
#python llama3_3b.py
python run_vllm.py