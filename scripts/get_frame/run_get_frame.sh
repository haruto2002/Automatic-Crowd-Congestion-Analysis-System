#!/bin/bash

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

PATH2VIDEO=$1
SAVE_DIR=$2
LOG_LEVEL=$3

python get_all_frame.py --path2video $PATH2VIDEO --save_dir $SAVE_DIR --log_level $LOG_LEVEL
