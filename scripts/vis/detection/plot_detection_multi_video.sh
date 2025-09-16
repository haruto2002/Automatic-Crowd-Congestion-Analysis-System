#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR
        
IO_INFO_FILE=$1
IMG_SUB_DIR=$2
DET_SUB_DIR=$3
DET_VIS_SUB_DIR=$4
FREQ=$5
NODE_TYPE=$6
LOG_LEVEL=$7

python scripts/vis/detection/plot_detection_multi_video.py \
        --io_info_file ${IO_INFO_FILE} \
        --img_sub_dir ${IMG_SUB_DIR} \
        --detection_sub_dir ${DET_SUB_DIR} \
        --detection_vis_sub_dir ${DET_VIS_SUB_DIR} \
        --freq ${FREQ} \
        --node_type ${NODE_TYPE} \
        --log_level ${LOG_LEVEL} \