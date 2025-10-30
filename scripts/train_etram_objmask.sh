#!/bin/bash
# Train DTA-SNN on ETram dataset with object masking (GPU 1)
# This version applies object masks to history frames (no label leakage)

export CUDA_VISIBLE_DEVICES=1

DATA_ROOT="/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs"
EXP_NAME="etram_dta_snn_objmask_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="./checkpoints/${EXP_NAME}"

python train_etram.py \
    --data_root ${DATA_ROOT} \
    --save_dir ${SAVE_DIR} \
    --pre_seq 10 \
    --aft_seq 10 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-3 \
    --use_focal \
    --DTA_ON \
    --use_amp \
    --use_obj_mask \
    --num_workers 4 \
    --device cuda:1

echo "Training with object masking completed! Results saved to ${SAVE_DIR}"
