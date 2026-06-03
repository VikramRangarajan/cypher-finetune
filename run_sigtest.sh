#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 0:2:30
#SBATCH -A gpu

#SBATCH --constraint=J
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@30

echo "Job started!"
source .venv/bin/activate
echo "Here"
srun python signaltest.py
echo "Job ended!"
