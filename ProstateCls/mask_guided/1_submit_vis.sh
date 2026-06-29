#!/bin/bash
#SBATCH -p gpu
#SBATCH -A r02144
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ojyhi010402@gmail.com
#SBATCH -D /geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/mask_guided

PYTHON=/N/slate/ohjiye/envs/medvit/bin/python3
WORKDIR=/geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/mask_guided

# ── Launcher (run locally: bash 1_submit_vis.sh <name>) ──────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    NAME=${1:-$(ls -td $WORKDIR/output/*/ 2>/dev/null | head -1 | xargs basename)}
    if [ -z "$NAME" ]; then
        echo "Usage: bash 1_submit_vis.sh <run-name>"
        exit 1
    fi
    LOG=$(ls $WORKDIR/logs/$NAME/*.out 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        echo "Error: no log found in logs/$NAME/"
        exit 1
    fi
    HEAD_DEPTH=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print(c.get('training', {}).get('head_depth', 2))
except: print(2)
" 2>/dev/null)
    HEAD_DEPTH=${HEAD_DEPTH:-2}
    BACKBONE=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print(c.get('training', {}).get('backbone', 'small'))
except: print('small')
" 2>/dev/null)
    BACKBONE=${BACKBONE:-small}
    mkdir -p $WORKDIR/figures/$NAME
    sbatch \
        --job-name=vis_$NAME \
        --output=$WORKDIR/logs/$NAME/%j.out \
        --error=$WORKDIR/logs/$NAME/%j.err \
        --export=ALL,RUN_NAME=$NAME,TRAIN_LOG=$LOG,HEAD_DEPTH=$HEAD_DEPTH,BACKBONE=$BACKBONE \
        $0
    echo "▶ Submitted vis: $NAME"
    exit 0
fi

# ── SLURM job body ────────────────────────────────────────────────────────────
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Run:    $RUN_NAME"
echo "Start:  $(date)"

$PYTHON $WORKDIR/visualize.py \
    --log        $TRAIN_LOG \
    --ckpt       $WORKDIR/output/$RUN_NAME/best.pth \
    --output-dir $WORKDIR/figures/$RUN_NAME \
    --n-slices 32 \
    --seed 42 \
    --val-size 0.15 \
    --test-size 0.15 \
    --head-depth ${HEAD_DEPTH:-2} \
    --backbone   ${BACKBONE:-small}

echo "End: $(date)"
