#!/bin/bash
#SBATCH -p gpu
#SBATCH -A c02008
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ojyhi010402@gmail.com
#SBATCH -D /geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/mil_abmil

PYTHON=/N/slate/ohjiye/envs/medvit/bin/python3
WORKDIR=/geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/mil_abmil

# Usage: bash 0_submit.sh <run-name> [extra-args]
# e.g.  bash 0_submit.sh baseline
#        bash 0_submit.sh focal "--focal-gamma 2.0"
#        bash 0_submit.sh attn  "--attn-lambda 0.1"
if [ -z "$SLURM_JOB_ID" ]; then
    NAME=${1:?Usage: bash 0_submit.sh <run-name> [extra-args]}
    EXTRA=${2:-}
    mkdir -p $WORKDIR/logs/$NAME $WORKDIR/output/$NAME $WORKDIR/figures/$NAME
    sbatch \
        --job-name=mil_$NAME \
        --output=$WORKDIR/logs/$NAME/%j.out \
        --error=$WORKDIR/logs/$NAME/%j.err \
        --export=NONE,RUN_NAME=$NAME,EXTRA_ARGS="$EXTRA" \
        $0
    echo "▶ Submitted: mil_$NAME  extra=${EXTRA:-none}"
    exit 0
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Run:    $RUN_NAME"
echo "Extra:  ${EXTRA_ARGS:-none}"
echo "Start:  $(date)"

$PYTHON $WORKDIR/train.py \
    --epochs 150 \
    --batch-size 8 \
    --lr-backbone 1e-5 \
    --lr-head 3e-4 \
    --weight-decay 1e-4 \
    --patience 30 \
    --seed 42 \
    --n-slices 32 \
    --val-size 0.15 \
    --test-size 0.15 \
    --freeze-bn \
    --output-dir $WORKDIR/output/$RUN_NAME \
    $EXTRA_ARGS

echo "End: $(date)"
