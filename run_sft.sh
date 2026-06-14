#!/bin/bash

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -A gpu

#SBATCH --constraint=J
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@300

#SBATCH -J cypherbench-sft-1
#SBATCH -o %x.out

export RUN_NAME=$SLURM_JOB_NAME
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

# If we reach timeout before run ends, wait returns immediately, goes into trap

uv run sft.py --lora_rank=null &
wait $!
uv run fix_checkpoint.py output/$RUN_NAME
# Start vllm async
uv run vllm serve output/$RUN_NAME --max_model_len=2048 &
VLLM_PID=$!
cd ~/cypherbench/docker
# Start neo4j async
INSTANCE_DIR=$HOME/cypherbench/.cache/neo4j-instances bash start_neo4j_test_apptainer.sh
# Wait for vllm startup
until curl -sf "http://0.0.0.0:8000/v1/models" >/dev/null; do
    sleep 2
    echo "vllm not up"
done
# Wait for neo4j startup
cd $SCRATCH/cypher-finetune
uv run wait_until_train_db_up.py --test
cd ~/cypherbench
uv run cypherbench/baseline/zero_shot_nl2cypher.py  --llm output/$RUN_NAME --api_base http://0.0.0.0:8000/v1 --api_key dummy  --result_dir output/$RUN_NAME --overwrite  --batch_size=128
kill -15 $!
uv run cypherbench/evaluate.py --result_dir output/$RUN_NAME --num_threads 32
