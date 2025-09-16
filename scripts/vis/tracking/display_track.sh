#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

# TRACK_DIR=demo/track
# IMG_DIR=demo/img
# VIS_DIR=demo/track_vis

TRACK_DIR=$1
IMG_DIR=$2
VIS_DIR=$3
FREQ=$4
NODE_TYPE=$5
LOG_LEVEL=$6
MULTI_VIDEO_MODE=$7

python visualize/tracking/display_track.py \
        --track_dir ${TRACK_DIR} \
        --img_dir ${IMG_DIR} \
        --save_base_dir ${VIS_DIR} \
        --freq ${FREQ} \
        --node_type ${NODE_TYPE} \
        --log_level ${LOG_LEVEL} \
        --multi_video_mode ${MULTI_VIDEO_MODE}
