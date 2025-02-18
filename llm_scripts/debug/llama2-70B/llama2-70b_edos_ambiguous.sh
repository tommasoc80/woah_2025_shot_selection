#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --job-name=llama2-70
#SBATCH --mem=80G
#SBATCH --output=edos_ambiguous_70.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
#python llama3_3b.py
python llama2_70b_edos_ambiguous.py
