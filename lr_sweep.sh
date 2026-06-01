#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -q standby

#SBATCH --mem=32G

#SBATCH -p a100-80gb
#SBATCH --gres=gpu:1

. ~/.bashrc
module purge

source .venv/bin/activate
python train.py --learning_rate=0.00002
