#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

PATH2VIDEO=$1
SAVE_DIR=$2
IMG_EXTENSION=$3
NODE_TYPE=$4
LOG_LEVEL=$5

python preprocess/get_all_frame.py --path2video $PATH2VIDEO --save_dir $SAVE_DIR --img_extension $IMG_EXTENSION --node_type $NODE_TYPE --log_level $LOG_LEVEL
