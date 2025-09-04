#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

# IMG_DIR=demo/img
# DET_DIR=demo/full_detection
# SAVE_DIR=demo/detection_plot

# IMG_DIR=error_analysis/img
# DET_DIR=error_analysis/full_detection
# SAVE_DIR=error_analysis/detection_plot

IMG_DIR=$1
DET_DIR=$2
SAVE_DIR=$3
FREQ=$4
LOG_LEVEL=$5

python visualize/detection/plot_detection.py \
        --detection_dir ${DET_DIR} \
        --img_dir ${IMG_DIR} \
        --save_dir ${SAVE_DIR} \
        --freq ${FREQ} \
        --log_level ${LOG_LEVEL}