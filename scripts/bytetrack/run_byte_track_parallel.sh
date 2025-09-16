#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -P gaa50073
#PBS -e /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/
#PBS -o /home/aag16599bn/research/Automatic-Crowd-Congestion-Analysis-System/log/

source ~/miniconda3/bin/activate hnakayama

WORKDIR=~/research/Automatic-Crowd-Congestion-Analysis-System
cd $WORKDIR

IO_INFO_FILE=${IO_INFO_FILE:-$1}
DETECTION_DATA_SUB_DIR=${DETECTION_DATA_SUB_DIR:-$2}
TRACK_SAVE_SUB_DIR=${TRACK_SAVE_SUB_DIR:-$3}
NODE_TYPE=${NODE_TYPE:-$4}
LOG_LEVEL=${LOG_LEVEL:-$5}

python bytetrack/run_point_tracker_parallel.py \
        --io_info_file ${IO_INFO_FILE} \
        --detection_data_sub_dir ${DETECTION_DATA_SUB_DIR} \
        --track_save_sub_dir ${TRACK_SAVE_SUB_DIR} \
        --img_h_size 4320 \
        --img_w_size 7680 \
        --track_thresh 0.5 \
        --track_buffer 30 \
        --match_thresh 10.0 \
        --distance_metric euclidean \
        --node_type ${NODE_TYPE} \
        --log_level ${LOG_LEVEL}
