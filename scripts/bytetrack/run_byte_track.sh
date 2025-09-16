#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

DETECTION_DIR=$1
TRACK_SAVE_DIR=$2
LOG_LEVEL=$3

python bytetrack/run_point_tracker.py \
        --save_dir ${TRACK_SAVE_DIR} \
        --source_dir ${DETECTION_DIR} \
        --img_h_size 4320 \
        --img_w_size 7680 \
        --track_thresh 0.5 \
        --track_buffer 30 \
        --match_thresh 10.0 \
        --distance_metric euclidean \
        --log_level ${LOG_LEVEL}
