#!/bin/bash
#SBATCH -p gpu
#SBATCH -A c02008
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ojyhi010402@gmail.com
#SBATCH -D /geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/mil_abmil

PYTHON=/N/slate/ohjiye/envs/medvit/bin/python3
WORKDIR=/geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/mil_abmil

# Usage: bash 1_submit_vis.sh <run-name>   (or no arg → latest run)
if [ -z "$SLURM_JOB_ID" ]; then
    NAME=${1:-$(ls -td $WORKDIR/output/*/ 2>/dev/null | head -1 | xargs basename)}
    if [ -z "$NAME" ]; then echo "Usage: bash 1_submit_vis.sh <run-name>"; exit 1; fi
    LOG=$(for f in $(ls -t $WORKDIR/logs/$NAME/*.out 2>/dev/null); do grep -q "Epoch " "$f" && echo "$f" && break; done)
    [ -z "$LOG" ] && echo "Warning: no training log found" && LOG=""

    BACKBONE=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print(c.get('backbone', 'small'))
except: print('small')
" 2>/dev/null)
    BACKBONE=${BACKBONE:-small}

    ADD_GLAND_CH=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print('1' if c.get('add_gland_ch', False) else '0')
except: print('0')
" 2>/dev/null)
    ADD_GLAND_CH=${ADD_GLAND_CH:-0}

    ABMIL_HIDDEN=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print(c.get('abmil_hidden', 256))
except: print(256)
" 2>/dev/null)
    ABMIL_HIDDEN=${ABMIL_HIDDEN:-256}

    SPATIAL_ATTN=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print('1' if c.get('spatial_attn', False) else '0')
except: print('0')
" 2>/dev/null)
    SPATIAL_ATTN=${SPATIAL_ATTN:-0}

    mkdir -p $WORKDIR/figures/$NAME
    sbatch \
        --job-name=vis_mil_$NAME \
        --output=$WORKDIR/logs/$NAME/%j.out \
        --error=$WORKDIR/logs/$NAME/%j.err \
        --export=ALL,RUN_NAME=$NAME,TRAIN_LOG=$LOG,BACKBONE=$BACKBONE,ADD_GLAND_CH=$ADD_GLAND_CH,ABMIL_HIDDEN=$ABMIL_HIDDEN,SPATIAL_ATTN=$SPATIAL_ATTN \
        $0
    echo "▶ Submitted vis: mil_$NAME  log=${LOG:-none}  spatial_attn=$SPATIAL_ATTN"
    exit 0
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Run:    $RUN_NAME"
echo "Start:  $(date)"

GLAND_FLAG=""
[ "${ADD_GLAND_CH:-0}" = "1" ] && GLAND_FLAG="--add-gland-ch"
SPATIAL_FLAG=""
[ "${SPATIAL_ATTN:-0}" = "1" ] && SPATIAL_FLAG="--spatial-attn"

$PYTHON $WORKDIR/visualize.py \
    --log           "$TRAIN_LOG" \
    --ckpt          $WORKDIR/output/$RUN_NAME/best.pth \
    --output-dir    $WORKDIR/figures/$RUN_NAME \
    --n-slices      32 \
    --seed          42 \
    --val-size      0.15 \
    --test-size     0.15 \
    --backbone      ${BACKBONE:-small} \
    --abmil-hidden  ${ABMIL_HIDDEN:-256} \
    $GLAND_FLAG $SPATIAL_FLAG

echo "End: $(date)"
