#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

DETECTION_DIR="output_pre/DSC_3809/detection/frame_detection"
TRACK_SAVE_DIR="demo_track"

python bytetrack/run_point_tracker.py \
        --save_dir ${TRACK_SAVE_DIR} \
        --source_dir ${DETECTION_DIR} \
        --img_h_size 4320 \
        --img_w_size 7680 \
        --track_thresh 0.5 \
        --track_buffer 30 \
        --match_thresh 10.0 \
        --distance_metric euclidean \
        --max_frame 30 \
        --log_level ERROR
