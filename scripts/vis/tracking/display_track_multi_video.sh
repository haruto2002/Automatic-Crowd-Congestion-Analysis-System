#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

IO_INFO_FILE=$1
TRACK_SUB_DIR=$2
IMG_SUB_DIR=$3
TRACK_VIS_SUB_DIR=$4
TRACK_VIS_FREQ=$5
NODE_TYPE=$6
LOG_LEVEL=$7

python scripts/vis/tracking/display_track_multi_video.py \
        --io_info_file ${IO_INFO_FILE} \
        --img_sub_dir ${IMG_SUB_DIR} \
        --track_sub_dir ${TRACK_SUB_DIR} \
        --track_vis_sub_dir ${TRACK_VIS_SUB_DIR} \
        --freq ${TRACK_VIS_FREQ} \
        --node_type ${NODE_TYPE} \
        --log_level ${LOG_LEVEL}
