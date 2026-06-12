#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -A gpu

#SBATCH --constraint=J
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@300

max_restarts=400      # tweak this number to fit your needs
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
outfile=$(scontrol show job ${SLURM_JOB_ID} | grep 'StdOut=' | cut -d= -f2)

##                                                          ##
##############################################################
##  Build a term-handler function to be executed            ##
##      when the job gets the SIGTERM                       ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ $restarts -lt $max_restarts ]];then
        # Copy the log file because it will be overwriten
        echo "Requeueing!"
        cp -v "${outfile}" "${outfile%.out}_${restarts}.out"
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximun restarts limit"
        exit 1
    fi
}

## Call the function when the jobs recieves the SIGTERM     ##
trap 'term_handler' SIGTERM

. ~/.bashrc
module purge

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
export TRAINER_RESUME=1
fi

cd ~/cypherbench/docker
INSTANCE_DIR=$HOME/cypherbench/.cache/neo4j-instances-2 bash start_neo4j_train_apptainer.sh
cd ~/cypher-finetune
uv run wait_until_train_db_up.py
# If we reach timeout before run ends, wait returns immediately, goes into trap
RUN_NAME=cypherbench-grpo-4 uv run grpo2.py --lora_rank=null --per_device_train_batch_size=2 --gradient_accumulation_steps=64 --num_generations=8 --steps_per_generation=32 &
wait $!
