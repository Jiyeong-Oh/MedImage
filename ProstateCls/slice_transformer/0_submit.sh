#!/bin/bash
#SBATCH -p gpu
#SBATCH -A r02144
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ojyhi010402@gmail.com
#SBATCH -D /geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/slice_transformer

PYTHON=/N/slate/ohjiye/envs/medvit/bin/python3
WORKDIR=/geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls/slice_transformer

# ── Launcher ──────────────────────────────────────────────────────────────────
# Default: CE loss, head_depth=2, backbone=small, 2-layer Transformer (8 heads)
# batch_size=4 (backbone processes B*32 images per step — memory heavy)
# e.g.  bash 0_submit.sh slice_tf_small
#        bash 0_submit.sh slice_tf_4layer "--num-layers 4"
#        bash 0_submit.sh slice_tf_focal  "--focal-gamma 2.0"
if [ -z "$SLURM_JOB_ID" ]; then
    NAME=${1:?Usage: bash 0_submit.sh <run-name> [extra-args]}
    EXTRA=${2:-}
    mkdir -p $WORKDIR/logs/$NAME $WORKDIR/output/$NAME $WORKDIR/figures/$NAME
    sbatch \
        --job-name=$NAME \
        --output=$WORKDIR/logs/$NAME/%j.out \
        --error=$WORKDIR/logs/$NAME/%j.err \
        --export=ALL,RUN_NAME=$NAME,EXTRA_ARGS="$EXTRA" \
        $0
    echo "▶ Submitted: $NAME  extra=${EXTRA:-none}"
    exit 0
fi

# ── SLURM job body ────────────────────────────────────────────────────────────
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Run:    $RUN_NAME"
echo "Extra:  ${EXTRA_ARGS:-none}"
echo "Start:  $(date)"

$PYTHON $WORKDIR/train.py \
    --epochs 150 \
    --lr-backbone 1e-5 \
    --lr-head 3e-4 \
    --lr-factor 0.5 \
    --lr-patience 10 \
    --weight-decay 1e-4 \
    --patience 30 \
    --batch-size 1 \
    --seed 42 \
    --n-slices 32 \
    --val-size 0.15 \
    --test-size 0.15 \
    --output-dir $WORKDIR/output/$RUN_NAME \
    $EXTRA_ARGS

echo "End: $(date)"
