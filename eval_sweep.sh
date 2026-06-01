#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -q standby

#SBATCH --mem=32G

#SBATCH -p a100-80gb
#SBATCH --gres=gpu:1

. ~/.bashrc
module purge

source .venv/bin/activate
# for RUN_NAME in peachy-brook-69 graceful-mountain-68 firm-serenity-65 autumn-pyramid-67; do
#     RUN_NAME=$RUN_NAME python eval.py
# done
python eval.py
