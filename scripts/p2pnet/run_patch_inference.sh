#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

# PATCH_OUT_DIR="demo/patch_detection/"
# WEIGHT_PATH="cutout.pth"
# IMG_DIR="demo/img"
# FULL_OUT_DIR="demo/full_detection"

# PATCH_OUT_DIR="error_analysis/patch_detection/"
# WEIGHT_PATH="cutout.pth"
# IMG_DIR="error_analysis/img"
# FULL_OUT_DIR="error_analysis/full_detection"


PATCH_OUT_DIR=$1
WEIGHT_PATH=$2
IMG_DIR=$3
FULL_OUT_DIR=$4

python p2pnet/run_patch_inference_ddp.py p2p \
    out_dir=$PATCH_OUT_DIR \
    default.finetune=True \
    network.init_weight=$WEIGHT_PATH \
    optimizer.batch_size.test=4 \
    default.bar=True \
    default.num_workers=4 \
    img_dir=$IMG_DIR \

python p2pnet/prediction/merge_detections.py \
    --source_dir $PATCH_OUT_DIR \
    --save_dir $FULL_OUT_DIR

echo "Done"
echo "Time: $SECONDS seconds"
