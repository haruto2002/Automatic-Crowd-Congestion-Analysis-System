#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

IMG_DIR=$1
DET_DIR=$2
SAVE_DIR=$3
FREQ=$4
NODE_TYPE=$5
LOG_LEVEL=$6
MULTI_VIDEO_MODE=$7

python visualize/detection/plot_detection.py \
        --detection_dir ${DET_DIR} \
        --img_dir ${IMG_DIR} \
        --save_dir ${SAVE_DIR} \
        --freq ${FREQ} \
        --node_type ${NODE_TYPE} \
        --log_level ${LOG_LEVEL} \
        --multi_video_mode ${MULTI_VIDEO_MODE}