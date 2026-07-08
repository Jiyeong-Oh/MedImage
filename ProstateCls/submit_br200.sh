#!/bin/bash
# BigRed200 launcher — does NOT modify existing scripts
# Usage: bash submit_br200.sh <method> <run-name> [extra-args]
#
# <method>: weight_tiling | channel_adapter | cnn_head | mask_guided | seg_cls | mil_abmil
# Example:  bash submit_br200.sh seg_cls seg_fbn_fbal
#           bash submit_br200.sh weight_tiling wt_os5 "--cs-oversample 5"

METHOD=${1:?Usage: bash submit_br200.sh <method> <run-name> [extra-args]}
NAME=${2:?Usage: bash submit_br200.sh <method> <run-name> [extra-args]}
EXTRA=${3:-}

BR200_DIR="$HOME/MedImage/ProstateCls"
QUARTZ_DIR="/geode3/home/u070/ohjiye/Quartz/MedImage/ProstateCls"

ORIG="$BR200_DIR/$METHOD/0_submit.sh"
if [ ! -f "$ORIG" ]; then
    echo "ERROR: $ORIG not found"
    exit 1
fi

mkdir -p "$BR200_DIR/$METHOD/logs/$NAME" \
         "$BR200_DIR/$METHOD/output/$NAME" \
         "$BR200_DIR/$METHOD/figures/$NAME"

TMPSCRIPT=$(mktemp /tmp/br200_submit_XXXX.sh)
sed \
    -e 's/-A r02144/-A c02008/' \
    -e "s|$QUARTZ_DIR|$BR200_DIR|g" \
    "$ORIG" > "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"

sbatch \
    --job-name="$NAME" \
    --output="$BR200_DIR/$METHOD/logs/$NAME/%j.out" \
    --error="$BR200_DIR/$METHOD/logs/$NAME/%j.err" \
    --export=NONE,RUN_NAME="$NAME",EXTRA_ARGS="$EXTRA" \
    "$TMPSCRIPT"

rm "$TMPSCRIPT"
echo "▶ Submitted $METHOD/$NAME on BigRed200  extra=${EXTRA:-none}"
