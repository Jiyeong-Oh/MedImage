#!/bin/bash
#SBATCH -p gpu
#SBATCH -A r02144
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ojyhi010402@gmail.com
#SBATCH -D /geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls

PYTHON=/N/slate/ohjiye/envs/medvit/bin/python3
WORKDIR=/geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls

if [ -z "$SLURM_JOB_ID" ]; then
    NAME=${1:-eval_threshold}
    EXTRA=${2:-}
    mkdir -p $WORKDIR/logs/eval
    sbatch \
        --job-name=$NAME \
        --output=$WORKDIR/logs/eval/${NAME}_%j.out \
        --error=$WORKDIR/logs/eval/${NAME}_%j.err \
        --export=ALL,EVAL_ARGS="$EXTRA" \
        $0
    echo "▶ Submitted: $NAME  args=${EXTRA}"
    exit 0
fi

echo "Job: $SLURM_JOB_ID  Start: $(date)"
$PYTHON $WORKDIR/eval_threshold.py $EVAL_ARGS
echo "End: $(date)"
