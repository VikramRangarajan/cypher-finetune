#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -q standby

#SBATCH --mem=16G

#SBATCH -p a100-80gb,a10
#SBATCH --gres=gpu:1

. ~/.bashrc
module purge

uv run train.py --learning_rate=0.000002
