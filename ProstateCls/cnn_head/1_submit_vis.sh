#!/bin/bash
#SBATCH -p gpu
#SBATCH -A r02144
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ojyhi010402@gmail.com
#SBATCH -D /geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/cnn_head

PYTHON=/N/slate/ohjiye/envs/medvit/bin/python3
WORKDIR=/geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/cnn_head

if [ -z "$SLURM_JOB_ID" ]; then
    NAME=${1:-$(ls -td $WORKDIR/output/*/ 2>/dev/null | head -1 | xargs basename)}
    if [ -z "$NAME" ]; then
        echo "Usage: bash 1_submit_vis.sh <run-name>"; exit 1
    fi
    LOG=$(for f in $(ls -t $WORKDIR/logs/$NAME/*.out 2>/dev/null); do grep -q "Epoch " "$f" && echo "$f" && break; done)
    if [ -z "$LOG" ]; then echo "Warning: no training log found in logs/$NAME/ — learning curve will be skipped"; LOG=""; fi
    CNN_HEAD=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print(c.get('training', {}).get('cnn_head', 'cbam'))
except: print('cbam')
" 2>/dev/null)
    CNN_HEAD=${CNN_HEAD:-cbam}
    BACKBONE=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print(c.get('training', {}).get('backbone', 'small'))
except: print('small')
" 2>/dev/null)
    BACKBONE=${BACKBONE:-small}
    ADD_GLAND_CH=$(python3 -c "
import json
try:
    c = json.load(open('$WORKDIR/output/$NAME/config.json'))
    print('1' if c.get('training', {}).get('add_gland_ch', False) else '0')
except: print('0')
" 2>/dev/null)
    ADD_GLAND_CH=${ADD_GLAND_CH:-0}
    mkdir -p $WORKDIR/figures/$NAME
    sbatch \
        --job-name=vis_$NAME \
        --output=$WORKDIR/logs/$NAME/%j.out \
        --error=$WORKDIR/logs/$NAME/%j.err \
        --export=ALL,RUN_NAME=$NAME,TRAIN_LOG=$LOG,CNN_HEAD=$CNN_HEAD,BACKBONE=$BACKBONE,ADD_GLAND_CH=$ADD_GLAND_CH \
        $0
    echo "▶ Submitted vis: $NAME  (head=$CNN_HEAD)"
    exit 0
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Run:    $RUN_NAME  head=$CNN_HEAD"
echo "Start:  $(date)"

GLAND_FLAG=""
if [ "${ADD_GLAND_CH:-0}" = "1" ]; then GLAND_FLAG="--add-gland-ch"; fi

$PYTHON $WORKDIR/visualize.py \
    --log        "$TRAIN_LOG" \
    --ckpt       $WORKDIR/output/$RUN_NAME/best.pth \
    --output-dir $WORKDIR/figures/$RUN_NAME \
    --cnn-head   ${CNN_HEAD:-cbam} \
    --backbone   ${BACKBONE:-small} \
    --n-slices 32 \
    --seed 42 \
    --val-size 0.15 \
    --test-size 0.15 \
    $GLAND_FLAG

echo "End: $(date)"
