#!/bin/bash
# Train DTA-SNN on ETram dataset (GPU 1)
# Usage: bash scripts/train_etram.sh

# Select GPU
export CUDA_VISIBLE_DEVICES=1

# Dataset path
DATA_ROOT="/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_with_BG"

# Experiment name
EXP_NAME="etram_dta_snn_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="./checkpoints/${EXP_NAME}"

# Training configuration
BATCH_SIZE=8
EPOCHS=100
LR=1e-3
PRE_SEQ=10
AFT_SEQ=10

# Run training
python train_etram.py \
    --data_root ${DATA_ROOT} \
    --save_dir ${SAVE_DIR} \
    --pre_seq ${PRE_SEQ} \
    --aft_seq ${AFT_SEQ} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay 1e-5 \
    --use_focal \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    --DTA_ON \
    --use_amp \
    --num_workers 4 \
    --train_stride 1 \
    --val_stride 5 \
    --device cuda:1 \
    --log_interval 1 \
    --save_interval 10

echo "Training completed! Results saved to ${SAVE_DIR}"
