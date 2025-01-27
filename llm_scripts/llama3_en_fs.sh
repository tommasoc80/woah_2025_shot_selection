#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --job-name=llama3fs
#SBATCH --mem=50G

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
python llama3_en_base_fs.py
