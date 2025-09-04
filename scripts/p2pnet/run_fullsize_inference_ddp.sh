#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

WEIGHT_PATH=$1
IMG_DIR=$2
OUT_DIR=$3
FULL_DET_DIR=$4
LOG_LEVEL=$5

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
