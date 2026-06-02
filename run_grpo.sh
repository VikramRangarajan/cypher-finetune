#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -A gpu

#SBATCH --constraint=J
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

. ~/.bashrc
module purge

cd ~/cypherbench/docker
bash start_neo4j_train_apptainer.sh
cd ~/cypher-finetune
source .venv/bin/activate
python wait_until_train_db_up.py
RUN_NAME=basemodel python grpo.py
