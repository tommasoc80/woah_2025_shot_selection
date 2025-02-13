#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --job-name=llama270-batch
#SBATCH --mem=80G
#SBATCH --output=prova-batch.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python batch_processing_llm-v2.py