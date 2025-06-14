#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:2
#SBATCH --job-name=md_ft
#SBATCH --mem=50G
#SBATCH --output=md_hb_ft.out

source ./bin/activate
python hatebert_md.py