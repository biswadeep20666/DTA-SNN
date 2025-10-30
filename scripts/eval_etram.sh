#!/bin/bash
# Evaluate DTA-SNN on ETram test set
# Usage: bash scripts/eval_etram.sh /path/to/checkpoint.pth

CHECKPOINT=$1

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash scripts/eval_etram.sh /path/to/checkpoint.pth"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=1

DATA_ROOT="/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs"

python evaluate_etram.py \
    --checkpoint ${CHECKPOINT} \
    --data_root ${DATA_ROOT} \
    --batch_size 8 \
    --num_workers 4 \
    --device cuda:1 \
    --visualize \
    --num_vis_samples 10

echo "Evaluation completed!"
