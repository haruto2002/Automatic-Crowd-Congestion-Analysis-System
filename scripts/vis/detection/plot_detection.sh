#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

IMG_DIR=$1
DET_DIR=$2
SAVE_DIR=$3
FREQ=$4
MULTI_VIDEO_MODE=$5
NODE_TYPE=$6
LOG_LEVEL=$7

python visualize/detection/plot_detection.py \
        --detection_dir ${DET_DIR} \
        --img_dir ${IMG_DIR} \
        --save_dir ${SAVE_DIR} \
        --freq ${FREQ} \
        --node_type ${NODE_TYPE} \
        --log_level ${LOG_LEVEL} \
        --multi_video_mode ${MULTI_VIDEO_MODE}