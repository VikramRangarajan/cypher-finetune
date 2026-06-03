#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 0:15:00
#SBATCH -A gpu

#SBATCH --constraint=J
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@90

. ~/.bashrc
module purge

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    export TRAINER_RESUME=1
fi

cd ~/cypherbench/docker
bash start_neo4j_train_apptainer.sh
cd ~/cypher-finetune
source .venv/bin/activate
python wait_until_train_db_up.py
RUN_NAME=requeuetest srun --export=ALL python grpo.py
