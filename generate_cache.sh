#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -A gpu

#SBATCH --mem=128G
#SBATCH --gres=gpu:1

. ~/.bashrc
module purge

cd ~/cypherbench/docker
INSTANCE_DIR=$HOME/cypherbench/.cache/neo4j-instances-2 bash start_neo4j_train_apptainer.sh
cd ~/cypher-finetune
uv run wait_until_train_db_up.py
uv run query_cache.py
