#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=sb_frozen
#SBATCH --mem=50G
#SBATCH --output=sbic_hb_frozen.out

module load Python/3.9.6-GCCcore-11.2.0
source ./bin/activate
python hatebert_sbic_frozen.py