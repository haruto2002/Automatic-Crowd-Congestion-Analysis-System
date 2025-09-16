#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

WEIGHT_PATH="cutout.pth"
IMG_DIR="outputs/demo/2025-09-01_14-15-28/results/img"
OUT_DIR="demo_output"
FULL_DET_DIR="demo_detection"
LOG_LEVEL="INFO"

python p2pnet/run_fullsize_inference_ddp.py p2p \
    out_dir=$OUT_DIR \
    default.finetune=True \
    network.init_weight=$WEIGHT_PATH \
    optimizer.batch_size.test=2 \
    default.bar=True \
    default.num_workers=2 \
    img_dir=$IMG_DIR \
    full_det_dir=$FULL_DET_DIR \
    log_level=$LOG_LEVEL

